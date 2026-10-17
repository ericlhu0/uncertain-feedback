"""Start poses for a care task, kept when they provoke feedback.

Each slot is either a persona, whose fixed bounds the start must provoke, or a
generated user, whose bound is sampled against the start's own rollout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluation.benchmarks.base import Benchmark
from evaluation.benchmarks.informative_scenarios import ScenarioCriteria
from evaluation.benchmarks.sampled_bounds import sample_bound
from evaluation.benchmarks.scenarios import user_from_dict
from evaluation.benchmarks.structs import InteractionTask
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    sample_arm_q,
)
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.goal_spaces import (
    GoalStallConfig,
    as_goal_region,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    compute_violations,
    first_violation_step,
    get_persona,
)
from uncertain_feedback.simulated_users.personas import UNRESTRICTED

# Resolved against the repo root: the MDM loader os.chdir()s into its submodule
# before the runner asks for tasks.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def generate_task_starts(
    config_path: Path,
    out_dir: Path,
    personas: Sequence[str],
    per_persona: int,
    seed: int,
    max_attempts: int,
    sampled_bounds: int = 0,
) -> None:
    """Keep up to ``per_persona`` start poses per persona for the task config's goal.

    A start is kept when it is comfortable, costs nothing under the task's base
    costs, the base-cost rollout from it to the config's first goal triggers
    feedback (as ``run_episode`` tests it, with the default stall stop), and the
    oracle (base + hidden cost) still reaches the goal, so the persona has
    something to correct and the task is solvable. A start-relative goal
    (``DisplacementRegion``) is bound to each start.

    ``sampled_bounds > 0`` replaces ``personas`` with that many generated users
    (``sampled_000``, ...; alternating constant and coupled bounds), one start
    each: ``sample_bound`` places a bound the base-cost rollout crosses after
    comfortable history, with no goal witness (the oracle test is the region's)
    and the crossing window cut at the rollout's end, as task rollouts are
    often shorter than it. The same trigger and oracle tests then apply, and
    the user is stored in its row.

    Every attempt is recorded in ``starts.json``; accepted rows have
    ``rejected == ""``.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg = load_mpc_config(config_path)
    assert cfg.cartesian is not None
    goal = as_goal_region(cfg.cartesian.goals[0])
    cfg = replace(
        cfg,
        seed=seed,
        cartesian=replace(
            cfg.cartesian, stall=cfg.cartesian.stall or GoalStallConfig()
        ),
    )
    human = Human(pose=cfg.pose, arm=cfg.arm)
    threshold = cfg.corrections.trigger_threshold
    criteria = ScenarioCriteria()
    names = (
        [f"sampled_{index:03d}" for index in range(sampled_bounds)]
        if sampled_bounds
        else list(personas)
    )
    quota = 1 if sampled_bounds else per_persona
    rows: list[dict[str, object]] = []
    audit: dict[str, object] = {
        "config": str(config_path.resolve()),
        "seed": seed,
        "per_persona": quota,
        "sampled_bounds": sampled_bounds,
        "attempts": rows,
    }
    for slot_index, name in enumerate(names):
        family = "constant" if slot_index % 2 == 0 else "coupled"
        # Base costs depend on the user only through the shared joint box.
        base = base_extra_costs(
            cfg.costs, human, UNRESTRICTED if sampled_bounds else get_persona(name)
        )
        rng = np.random.default_rng([seed, slot_index])
        kept = 0
        for attempt in range(max_attempts):
            if kept == quota:
                break
            q0 = sample_arm_q(rng, human)
            row: dict[str, object] = {
                "persona": name,
                "attempt": attempt,
                "start_q": q0.tolist(),
            }
            rows.append(row)
            start = human.reset_human_with_q(q0)
            start_goal = goal.bind(human.arm_aa_from_q(q0), human)
            # The task's own costs (e.g. a mattress floor) must hold at the start.
            if float(base(human.arm_aa_from_q(q0)[None, None])[0]) > 0.0:
                row["rejected"] = "infeasible_start"
                continue
            if sampled_bounds:
                nominal = rollout_to_goal(
                    cfg, start, start_goal, base, stop_on_stall=True
                ).history
                sampled = sample_bound(
                    rng,
                    nominal,
                    None,
                    human,
                    name,
                    family,
                    criteria.min_history,
                    criteria.window,
                    criteria.min_nominal_violation,
                    clip_window=True,
                )
                if sampled is None:
                    row["rejected"] = "no_separating_bound"
                    continue
                user = sampled
                row["user"] = asdict(user)
            else:
                user = get_persona(name)
                if compute_violations(user, human, q0[None]).max() > 0.0:
                    row["rejected"] = "uncomfortable_start"
                    continue
                nominal = rollout_to_goal(
                    cfg, start, start_goal, base, stop_on_stall=True
                ).history
            trigger = first_violation_step(user, human, nominal, threshold)
            if trigger is None:
                row["rejected"] = "no_violation"
                continue
            oracle_costs = CompositeTrajectoryCost(
                [*base.terms(), HiddenCostTerm(user=user, human=human)]
            )
            oracle = rollout_to_goal(cfg, start, start_goal, oracle_costs).history
            if not goal_reach(human, cfg, oracle, start_goal)["reached"]:
                row["rejected"] = "oracle_did_not_reach"
                continue
            row.update(trigger=trigger, rejected="")
            kept += 1
            print(f"[task_starts] {name} attempt {attempt}: kept", flush=True)
        print(f"[task_starts] {name}: {kept}/{quota} kept", flush=True)
        (out_dir / "starts.json").write_text(json.dumps(audit, indent=2))


class TaskStartsBenchmark(Benchmark):
    """One task per accepted start in ``starts_dir/starts.json``.

    Every task pursues the run config's first goal, so pair it with the same
    ``mpc_config`` the starts were generated with. A generated user's row
    carries the user itself; a persona's is resolved by name.
    """

    def __init__(
        self,
        name: str,
        starts_dir: str,
        verbalizers: Sequence[str] = ("everyday",),
        max_rounds: int = 3,
    ) -> None:
        super().__init__(name)
        self.starts_dir = _REPO_ROOT / starts_dir
        self.verbalizers = list(verbalizers)
        self.max_rounds = max_rounds

    def generate_tasks(self, seed: int, cfg: MpcRunConfig) -> list[InteractionTask]:
        assert cfg.cartesian is not None
        goal = as_goal_region(cfg.cartesian.goals[0])
        audit = json.loads((self.starts_dir / "starts.json").read_text())
        return [
            InteractionTask(
                persona=row["persona"],
                verbalizer=verbalizer,
                goals=(goal,),
                max_rounds=self.max_rounds,
                seed=seed,
                user=user_from_dict(row["user"]) if "user" in row else None,
                start_q=tuple(float(v) for v in row["start_q"]),
            )
            for row in audit["attempts"]
            if row.get("rejected") == ""
            for verbalizer in self.verbalizers
        ]
