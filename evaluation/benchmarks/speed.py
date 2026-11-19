"""Limit benchmark: one motion under three ways of holding a hidden limit.

Per persona, care task and sampled start, the arm follows a correction (when
one is given) and then reaches for the task's goal under three arms: ``none``
(base costs only), ``hidden_soft`` (the persona's hidden limit as the old soft
planner cost, a weight times the squared excess averaged over the horizon) and
``constraint`` (the same limit hand-written as margins in the generated form,
which the planner charges hard: linear in the worst step's excess, at
``limit_weight``). Speed personas test speed limits, the others path limits.
The correction is played back as chosen, exempt from every limit. No LLM and
no motion generator is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics.speed import limit_violation, step_excess
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    sample_arm_q,
)
from uncertain_feedback.planners.mpc import ArmMPC, FeedbackConfig
from uncertain_feedback.planners.mpc.arm_features import STEP_SECONDS
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    base_extra_costs,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.goal_spaces import GoalRegion, as_goal_region
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import anchor_q_trajectory
from uncertain_feedback.planners.mpc.rollout import run_planning_loop
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    HiddenSpeedTerm,
    SimulatedUser,
    compute_violations,
    first_violation_step,
    get_persona,
    noticed_speed_limit,
    verbalize_speed,
)
from uncertain_feedback.utils.plot import ArmVisualizer

ARMS = ("none", "hidden_soft", "constraint")
SUMMARY_COLUMNS = [
    "reached",
    "playback_over_limit_share",
    "playback_peak_excess",
    "reach_over_limit_share",
    "reach_peak_excess",
    "duration_ratio",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OVER_LIMIT_COLOR = "crimson"

# One hand-written constraint per persona, its hidden limit in the generated
# form: margins in physical units, positive when violated.
CONSTRAINTS: dict[str, tuple[str, dict[str, Any]]] = {
    "moved_too_fast": (
        "def speed_margins(features, params):\n"
        "    return {\n"
        "        'hand': features['wrist_speed'] - params['hand_cap'],\n"
        "        'elbow': features['elbow_speed'] - params['elbow_cap'],\n"
        "    }\n",
        {"hand_cap": 0.25, "elbow_cap": 0.125},
    ),
    "spastic_elbow_flexors": (
        "def speed_margins(features, params):\n"
        "    cap = interp(features['elbow_flexion'], params['flexion'], params['cap'])\n"
        "    return {'elbow_extension': -features['elbow_flexion_rate'] - cap}\n",
        {"flexion": [0.79, 2.79], "cap": [0.14, 5.14]},
    ),
    "adhesive_capsulitis": (
        "def path_margins(features, params):\n"
        "    return {'elevation': features['shoulder_elevation'] - params['max']}\n",
        {"max": 1.25},
    ),
    "elbow_contracture": (
        "def path_margins(features, params):\n"
        "    return {'extension': params['min'] - features['elbow_flexion']}\n",
        {"min": 0.5},
    ),
}


@dataclass(frozen=True)
class SpeedRun:
    """One arm's executed motion from one start.

    ``executed`` is ``(T, 7)``, one planner step per frame; its first
    ``playback_steps`` steps followed the correction.
    """

    arm: str
    executed: np.ndarray
    playback_steps: int
    reached: bool

    @property
    def playback(self) -> np.ndarray:
        """The correction-playback stretch, from the start frame on."""
        return self.executed[: self.playback_steps + 1]

    @property
    def reach(self) -> np.ndarray:
        """The goal-reach stretch, from the last playback frame on."""
        return self.executed[self.playback_steps :]


def arm_costs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    arm: str,
    user: SimulatedUser,
    human: Human,
    base: CompositeTrajectoryCost,
    weight: float = 10.0,
    limit_weight: float = 1000.0,
    pullback_weight: float = 10.0,
) -> CompositeTrajectoryCost:
    """The planner costs of ``arm``: the base costs plus its limit term, if any.

    ``weight`` is the soft arm's, on the squared excess averaged over the
    horizon; ``limit_weight`` the generated form's, on the worst step's excess,
    and ``pullback_weight`` its cost for staying past a path limit.
    """
    terms = list(base.terms())
    if arm == "hidden_soft":
        terms.append(
            HiddenSpeedTerm(user=user, human=human, weight=weight)
            if user.speed_limits
            else HiddenCostTerm(user=user, human=human, weight=weight)
        )
    if arm == "constraint":
        code, params = CONSTRAINTS[user.name]
        context = build_generated_cost_context(
            human,
            human.q[None],
            window=1,
            limit_weight=limit_weight,
            pullback_weight=pullback_weight,
        )
        terms.append(GeneratedPythonCost(code, params, context))
    return CompositeTrajectoryCost(terms)


def run_arm(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    arm: str,
    cfg: MpcRunConfig,
    human: Human,
    goal: GoalRegion,
    costs: CompositeTrajectoryCost,
    correction: np.ndarray | None,
) -> SpeedRun:
    """Follow ``correction`` from ``human``'s pose, then reach for ``goal``.

    Playback bypasses ``costs``, as it does for every learned cost.
    """
    assert cfg.cartesian is not None
    mpc = ArmMPC(
        human,
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        extra_costs=costs,
        seed=cfg.seed,
        cartesian=replace(cfg.cartesian, goals=[goal]),
        feedback=(
            None
            if correction is None
            else replace(cfg.feedback or FeedbackConfig(), uq=None)
        ),
    )
    if correction is not None:
        mpc.push_trajectory(anchor_q_trajectory(correction, human.q))
    playing: list[bool] = []
    result = run_planning_loop(
        mpc,
        cfg.steps,
        on_pre_step=lambda _step, _human: playing.append(not mpc.mdm_tracking_complete),
    )
    return SpeedRun(arm, result.human.history, sum(playing), result.reached_goal)


def complaint(
    user: SimulatedUser, human: Human, executed: np.ndarray, threshold: float
) -> tuple[int, str] | None:
    """When and what the persona would say about ``executed``, if anything.

    A speed persona speaks once a speed has stayed over its cap for a window
    (``noticed_speed_limit``), any other at the first frame past its bounds by
    ``threshold``.
    """
    if user.speed_limits:
        noticed = noticed_speed_limit(user, human, executed)
        return (
            None if noticed is None else (noticed[0], verbalize_speed(noticed[1]).text)
        )
    step = first_violation_step(user, human, executed, threshold)
    return None if step is None else (step, user.feedback_text)


def speed_row(
    run: SpeedRun,
    user: SimulatedUser,
    human: Human,
    baseline: SpeedRun,
    threshold: float,
) -> dict[str, Any]:
    """One arm's metrics; ``baseline`` is the unconstrained arm from the same start.

    The violation is given for the whole motion and for its playback and reach
    stretches apart (NaN for a stretch with no step).
    """
    said = complaint(user, human, run.executed, threshold)
    parts = {"playback": run.playback, "reach": run.reach}
    unit = "rad"
    if user.speed_limits:
        unit = "m/s" if user.speed_limits[0].feature.endswith("_speed") else "rad/s"
    return {
        "arm": run.arm,
        "reached": run.reached,
        **limit_violation(user, human, list(parts.values())),
        **{
            f"{name}_{key}": value
            for name, part in parts.items()
            for key, value in limit_violation(user, human, [part]).items()
        },
        "unit": unit,
        "duration_s": (len(run.executed) - 1) * STEP_SECONDS,
        "duration_ratio": (len(run.executed) - 1) / (len(baseline.executed) - 1),
        "complaint_step": None if said is None else said[0],
        "complaint": "" if said is None else said[1],
    }


def render_run(
    run: SpeedRun, user: SimulatedUser, human: Human, goal: GoalRegion, path: Path
) -> None:
    """Rollout video: playback orange, the reach blue, steps over the hidden limit red."""
    colors = [ArmVisualizer.TARGET_COLOR] * len(run.executed)
    colors[1 : run.playback_steps + 1] = [ArmVisualizer.MDM_COLOR] * run.playback_steps
    for frame in np.flatnonzero(step_excess(user, human, run.executed) > 0.0) + 1:
        colors[frame] = _OVER_LIMIT_COLOR
    ArmVisualizer(human.fk).render_rollout_video(
        human.arm_aa_from_q(run.executed),
        path,
        spine3_pos=human.spine3_pos,
        spine3_aa=human.spine3_aa,
        body_pos=human.posture,
        cartesian_goal=goal,
        frame_colors=colors,
    )


def run_speed_eval(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_dir: Path,
    personas: Sequence[str],
    tasks: Sequence[str],
    n_starts: int,
    seed: int,
    max_attempts: int,
    correction: np.ndarray | None,
    weight: float,
    limit_weight: float,
    pullback_weight: float,
) -> pd.DataFrame:
    """Run every arm from ``n_starts`` sampled starts per persona and care task.

    A start is kept when it is inside the persona's path bounds, the task's
    base costs are zero there, the unconstrained arm reaches the goal and the
    persona complains about it, so there is something to hold back and the
    task is solvable. ``weight`` is the ``hidden_soft`` arm's, ``limit_weight``
    and ``pullback_weight`` the ``constraint`` arm's. Writes ``speed_eval.csv`` (one row per run),
    ``summary.csv`` (means per persona and arm), every kept start's executed
    motions under ``runs/<persona>_<task>_<start>.npz`` and, for each persona's
    first kept start, the ``none`` and ``constraint`` rollout videos under
    ``videos/``.
    """
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for persona_index, persona in enumerate(personas):
        user = get_persona(persona)
        rendered = False
        for task_index, task in enumerate(tasks):
            cfg = replace(
                load_mpc_config(_REPO_ROOT / f"evaluation/conf/care_tasks/{task}.yaml"),
                seed=seed,
            )
            assert cfg.cartesian is not None
            threshold = cfg.corrections.trigger_threshold
            body = Human(pose=cfg.pose, arm=cfg.arm)
            base = base_extra_costs(cfg.costs, body, user)
            region = as_goal_region(cfg.cartesian.goals[0])
            rng = np.random.default_rng([seed, persona_index, task_index])
            kept = 0
            for _ in range(max_attempts):
                if kept == n_starts:
                    break
                human = body.reset_human_with_q(sample_arm_q(rng, body))
                if (
                    float(base(human.arm_aa_from_q(human.q)[None, None])[0]) > 0.0
                    or compute_violations(user, human, human.q[None]).max() > 0.0
                ):
                    continue
                goal = region.bind(human.arm_aa_from_q(human.q), human)
                baseline = run_arm("none", cfg, human, goal, base, correction)
                if not baseline.reached or (
                    complaint(user, human, baseline.executed, threshold) is None
                ):
                    continue
                runs = [baseline] + [
                    run_arm(
                        arm,
                        cfg,
                        human,
                        goal,
                        arm_costs(
                            arm,
                            user,
                            human,
                            base,
                            weight,
                            limit_weight,
                            pullback_weight,
                        ),
                        correction,
                    )
                    for arm in ARMS[1:]
                ]
                for run in runs:
                    rows.append(
                        {
                            "persona": persona,
                            "task": task,
                            "start": kept,
                            **speed_row(run, user, human, baseline, threshold),
                        }
                    )
                np.savez_compressed(
                    out_dir / "runs" / f"{persona}_{task}_{kept}.npz",
                    **{run.arm: run.executed for run in runs},
                    playback_steps=[run.playback_steps for run in runs],
                )
                if not rendered:
                    for run in (runs[0], runs[2]):
                        render_run(
                            run,
                            user,
                            human,
                            goal,
                            out_dir / "videos" / f"{persona}_{run.arm}.mp4",
                        )
                    rendered = True
                kept += 1
                print(f"[speed] {persona} {task}: start {kept}/{n_starts}", flush=True)
                pd.DataFrame(rows).to_csv(out_dir / "speed_eval.csv", index=False)
    table = pd.DataFrame(rows)
    table.groupby(["persona", "arm"], sort=False)[SUMMARY_COLUMNS].mean().to_csv(
        out_dir / "summary.csv"
    )
    return table
