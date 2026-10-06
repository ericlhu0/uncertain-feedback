"""Smoke tests for the evaluation harness (benchmarks x approaches x episode)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.approaches import (
    Approach,
    BridgePotentialFieldGrounder,
    ConsolidateCostGen,
    ImmediateCostGen,
    LlmKeypointGrounder,
    NoCostGen,
)
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from evaluation.benchmarks.base import InteractionBenchmark
from evaluation.benchmarks.episode import run_episode
from evaluation.benchmarks.structs import Interaction, InteractionTask
from uncertain_feedback.cost_generation import CostRound
from uncertain_feedback.planners.mpc import rollout as rollout_module
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.goal_spaces import GoalStallConfig
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import propose_unblocked_path
from uncertain_feedback.simulated_users import get_persona

_SMOKE_MPC = (
    Path(__file__).resolve().parents[1] / "evaluation" / "conf" / "mpc_smoke.yaml"
)


def test_benchmark_generates_persona_verbalizer_grid() -> None:
    """Tasks enumerate personas x verbalizers with resolved goal tuples."""
    cfg = load_mpc_config(_SMOKE_MPC)
    bench = InteractionBenchmark(
        name="grid",
        personas=["elbow_contracture", "painful_arc"],
        verbalizers=["vague", "everyday"],
        max_rounds=2,
    )
    tasks = bench.generate_tasks(3, cfg)
    assert len(tasks) == 4
    assert {task.persona for task in tasks} == {"elbow_contracture", "painful_arc"}
    assert all(task.goals == ((-0.18, 0.40, 0.34),) for task in tasks)
    assert all(task.seed == 3 for task in tasks)


def test_bridge_baseline_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end on CPU with the potential-field baseline."""
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    user = get_persona("elbow_contracture")
    bench = InteractionBenchmark(
        name="smoke",
        personas=["elbow_contracture"],
        verbalizers=["joint_resolved"],
        goals=[[-0.18, 0.40, 0.34]],
        max_rounds=1,
    )
    task = bench.generate_tasks(0, cfg)[0]
    approach = Approach(
        name="bridge_baseline",
        grounder=BridgePotentialFieldGrounder(),
        cost_gen=NoCostGen(),
    )
    approach.reset(cfg, human, None, user, task.seed, tmp_path / "episode")
    result = run_episode(cfg, human, user, task, approach, tmp_path / "episode")
    assert (tmp_path / "episode" / "episode_summary.json").exists()
    assert result, "episode recorded no interactions"


class _PinnedCostGen(NoCostGen):
    """A learned cost that keeps the arm where it is, so every rollout stalls."""

    def __init__(self, human: Human) -> None:
        super().__init__()
        self.cost = GeneratedPythonCost(
            code=(
                "def cost(q_trajs, context, params):\n"
                "    return 1e3 * np.sum((q_trajs[:, 1:] - q_trajs[:, :1]) ** 2, "
                "axis=(1, 2, 3))\n"
            ),
            params={},
            context=build_generated_cost_context(
                human, mdm_traj=np.zeros((3, 3, 3)), window=3
            ),
        )

        self.retired: list[tuple[tuple[GeneratedPythonCost, ...], str]] = []

    def learned_terms(self) -> list[GeneratedPythonCost]:
        return [self.cost] if not self.retired else []

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        if ctx.retire:
            self.retired.append((ctx.retire, ctx.utterance_text))
        return LearnOutcome(cost_accepted=True, unified_installed=False)


def _pinned_episode(
    tmp_path: Path, persona: str, max_rounds: int = 1
) -> tuple[Interaction, _PinnedCostGen]:
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    assert cfg.cartesian is not None
    cfg = replace(
        cfg, cartesian=replace(cfg.cartesian, stall=GoalStallConfig(window=5))
    )
    human = Human(pose=cfg.pose, arm=cfg.arm)
    user = get_persona(persona)
    task = InteractionTask(
        persona=persona,
        verbalizer="joint_resolved",
        goals=((-0.18, 0.40, 0.34),),
        max_rounds=max_rounds,
        seed=0,
    )
    cost_gen = _PinnedCostGen(human)
    approach = Approach(
        name="pinned",
        grounder=BridgePotentialFieldGrounder(),
        cost_gen=cost_gen,
    )
    approach.reset(cfg, human, None, user, task.seed, tmp_path / "episode")
    episode = run_episode(cfg, human, user, task, approach, tmp_path / "episode")
    return episode[0], cost_gen


def test_stalled_rollout_follows_an_approved_proposal(tmp_path: Path) -> None:
    """A persona the proposal does not bother approves it and the goal resolves."""
    interaction, _ = _pinned_episode(tmp_path, "unrestricted")
    assert interaction.result == "no_violation"
    assert interaction.reached
    assert interaction.proposal_approved
    assert len(interaction.executed) > len(interaction.initial_rollout)


def test_retire_keeps_rounds_for_consolidation(tmp_path: Path) -> None:
    """Retired costs leave planning; their rounds stay, marked, for combining."""
    human = Human()
    context = build_generated_cost_context(
        human, mdm_traj=np.zeros((3, 3, 3)), window=3
    )
    costs = [
        GeneratedPythonCost(
            code="def cost(q_trajs, context, params):\n    return np.zeros(len(q_trajs))\n",
            params={},
            context=context,
        )
        for _ in range(2)
    ]
    rounds = [
        CostRound(
            index=i,
            goal=None,
            feedback_text=f"round {i}",
            trigger_step=0,
            round_dir=tmp_path,
            state_path=tmp_path / "state.pkl",
            cost_code=costs[i].code,
            params={},
            summaries={},
            image_paths=(),
        )
        for i in range(2)
    ]

    stacked = ImmediateCostGen()
    stacked._generated, stacked._cost_rounds = list(costs), list(rounds)
    stacked.retire([costs[0]], "too strict")
    assert stacked.learned_terms() == [costs[1]]
    assert [r.retired for r in stacked._cost_rounds] == ["too strict", ""]

    unified = ConsolidateCostGen()
    unified._generated, unified._cost_rounds = list(costs), list(rounds)
    unified._unified = costs[1]
    unified.retire([costs[1]], "too strict")
    assert not unified.learned_terms()
    assert all(r.retired == "too strict" for r in unified._cost_rounds)


def test_consolidation_sees_the_retired_round_note(tmp_path: Path, monkeypatch) -> None:
    """The combine after a rejection weighs the retired round, marked as such."""
    human = Human()
    context = build_generated_cost_context(
        human, mdm_traj=np.zeros((3, 3, 3)), window=3
    )
    blocking, correcting, combined = (
        GeneratedPythonCost(
            code="def cost(q_trajs, context, params):\n    return np.zeros(len(q_trajs))\n",
            params={},
            context=context,
        )
        for _ in range(3)
    )
    cost_gen = ConsolidateCostGen()
    cost_gen._generated = [blocking]
    cost_gen._cost_rounds = [
        CostRound(
            index=0,
            goal=None,
            feedback_text="keep my elbow down",
            trigger_step=0,
            round_dir=tmp_path,
            state_path=tmp_path / "state.pkl",
            cost_code=blocking.code,
            params={},
            summaries={},
            image_paths=(),
        )
    ]
    cost_gen._unified = blocking
    seen: list[CostRound] = []

    def fake_combine(_ctx, _generation):
        seen.extend(cost_gen._cost_rounds)
        return combined

    monkeypatch.setattr(cost_gen, "_combine", fake_combine)
    ctx = RoundContext(
        round_dir=tmp_path,
        goal=np.zeros(3),
        utterance_text="bring my hand in",
        grounding=None,  # type: ignore[arg-type]
        human=human,
        event_index=1,
        rejected_labels=frozenset(),
        retire=(blocking,),
    )
    generation = SimpleNamespace(
        generated_cost=correcting,
        eval_state=SimpleNamespace(save=lambda _path: None),
        summaries={},
        images={},
        description="",
        explanation="",
        interpretation="",
        grounding="",
    )

    cost_gen.record(ctx, generation)  # type: ignore[arg-type]

    assert "bring my hand in" in seen[0].retired and not seen[1].retired
    assert cost_gen.learned_terms() == [combined]


def test_rejected_proposal_becomes_the_next_rounds_plan(tmp_path: Path) -> None:
    """A rejected proposal is what the next round's correction is attributed against."""
    interaction, cost_gen = _pinned_episode(tmp_path, "elbow_contracture", max_rounds=2)
    first, second = interaction.rounds
    assert first.retrigger_step is None
    assert first.proposal_rejected and not first.resolved
    np.testing.assert_allclose(second.q_feedback, first.continuation[-1])
    np.testing.assert_allclose(second.nominal_plan[0], first.continuation[-1])
    # The pinned cost keeps a planned nominal still; the proposal leaves it out.
    assert np.linalg.norm(second.nominal_plan[-1] - second.nominal_plan[0]) > 0.1
    # The correction after the rejection is learned with the pinned cost to retire.
    assert len(cost_gen.retired) == 1
    ((retired_terms, correction_text),) = cost_gen.retired
    assert retired_terms[0] is cost_gen.cost
    assert correction_text == second.utterance.text


def test_proposal_leaves_out_only_the_blocking_costs(monkeypatch) -> None:
    """Only the learned costs that keep the arm from the goal are left out."""
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    assert cfg.cartesian is not None
    cfg = replace(
        cfg, cartesian=replace(cfg.cartesian, stall=GoalStallConfig(window=5))
    )
    human = Human(pose=cfg.pose, arm=cfg.arm)
    harmless = GeneratedPythonCost(
        code="def cost(q_trajs, context, params):\n    return np.zeros(len(q_trajs))\n",
        params={},
        context=_PinnedCostGen(human).cost.context,
    )
    pins = [_PinnedCostGen(human).cost, _PinnedCostGen(human).cost]

    loops: list[int] = []
    goal_loop = rollout_module._goal_loop  # pylint: disable=protected-access

    def counted(*args, **kwargs):
        loops.append(1)
        return goal_loop(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "_goal_loop", counted)
    older_blocks = propose_unblocked_path(
        cfg, human, CompositeTrajectoryCost([pins[0], harmless])
    )
    assert older_blocks is not None and older_blocks.reached_goal
    assert len(older_blocks.dropped) == 1 and older_blocks.dropped[0] is pins[0]
    # The pinned cost pushes back on the look-ahead, so it is tried first.
    assert len(loops) == 2

    both_block = propose_unblocked_path(
        cfg, human, CompositeTrajectoryCost([pins[0], pins[1]])
    )
    assert both_block is not None and both_block.reached_goal
    assert len(both_block.dropped) == 2

    loops.clear()
    four = [_PinnedCostGen(human).cost for _ in range(4)]
    all_block = propose_unblocked_path(
        cfg, human, CompositeTrajectoryCost([four[0], four[1], four[2], four[3]])
    )
    assert all_block is not None and all_block.reached_goal
    assert len(all_block.dropped) == 4
    # Look-ahead, 4 singles, 6 pairs, then all four: no triples searched.
    assert len(loops) == 12


def test_keypoint_baseline_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end with a stubbed keypoint interpreter."""
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    user = get_persona("elbow_contracture")
    bench = InteractionBenchmark(
        name="smoke",
        personas=["elbow_contracture"],
        verbalizers=["joint_resolved"],
        goals=[[-0.18, 0.40, 0.34]],
        max_rounds=1,
    )
    task = bench.generate_tasks(0, cfg)[0]
    grounder = LlmKeypointGrounder()
    grounder._interpret = (  # type: ignore[method-assign]
        lambda text, scene_context: {
            "joint": "wrist",
            "keypoint": np.array([0.1, 0.3, 0.2]),
        }
    )
    approach = Approach(name="llm_keypoint", grounder=grounder, cost_gen=NoCostGen())
    approach.reset(cfg, human, None, user, task.seed, tmp_path / "episode")
    result = run_episode(cfg, human, user, task, approach, tmp_path / "episode")
    assert (tmp_path / "episode" / "episode_summary.json").exists()
    assert result, "episode recorded no interactions"
