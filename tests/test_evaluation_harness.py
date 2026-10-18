"""Smoke tests for the evaluation harness (benchmarks x approaches x episode)."""

import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.approaches import (
    Approach,
    BridgePotentialFieldGrounder,
    ConsolidateCostGen,
    ImmediateCostGen,
    LlmKeypointGrounder,
    NoCostGen,
)
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from evaluation.benchmarks import episode as episode_module
from evaluation.benchmarks import structs as structs_module
from evaluation.benchmarks.base import InteractionBenchmark
from evaluation.benchmarks.episode import _Motion, _move_to_goal, run_episode
from evaluation.benchmarks.structs import Interaction, InteractionTask, crossings
from evaluation.metrics.cost_learning.success import success_at_k
from uncertain_feedback.cost_generation import CostRound
from uncertain_feedback.planners.mpc import rollout as rollout_module
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.goal_spaces import GoalStallConfig, PointRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import (
    StallProposal,
    propose_unblocked_path,
)
from uncertain_feedback.simulated_users import get_persona
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

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
    assert all(task.goals == (PointRegion((-0.18, 0.40, 0.34)),) for task in tasks)
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
        goals=(PointRegion((-0.18, 0.40, 0.34)),),
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
    np.testing.assert_allclose(interaction.executed, interaction.initial_rollout)
    ((start, end),) = interaction.approved_spans
    assert 0 < start < end == len(interaction.executed) - 1
    # The approval is the persona's one turn: no success without it.
    assert interaction.approvals == 1 and interaction.user_turns == 1
    curve = success_at_k([interaction], max_k=1)
    assert curve.set_index("k")["success"].tolist() == [0.0, 1.0]
    assert interaction.violation_count == 0
    assert interaction.motion_steps == len(interaction.executed) - 1
    assert 0.0 < interaction.robot_seconds < np.inf


def _line_world(
    monkeypatch, replans: list[list[float]], proposals: list[list[float]]
) -> None:
    """Script a 1-D world for :func:`_move_to_goal`: the goal is q[0] = 1.

    Each replan and each proposal moves q[0] by its per-step deltas from
    wherever the arm is; the persona approves every proposal.
    """
    replan_steps, proposal_steps = iter(replans), iter(proposals)

    def walk(at: Human, deltas: list[float]) -> Human:
        frames = at.q + np.outer(np.cumsum(deltas), np.eye(7)[0])
        return at.reset_human_with_q(at.q).step(frames)

    monkeypatch.setattr(
        episode_module,
        "rollout_to_goal",
        lambda cfg, at, goal, costs, **_: walk(at, next(replan_steps)),
    )
    monkeypatch.setattr(
        episode_module,
        "propose_unblocked_path",
        lambda cfg, at, costs: StallProposal(
            human=walk(at, next(proposal_steps)), dropped=(), reached_goal=False
        ),
    )
    monkeypatch.setattr(
        episode_module,
        "goal_reach",
        lambda human, cfg, rollout, goal: {
            "reached": rollout[-1][0] >= 1.0 - 1e-9,
            "distance": max(1.0 - float(rollout[-1][0]), 0.0),
        },
    )
    monkeypatch.setattr(
        episode_module, "first_violation_step", lambda *args, **kwargs: None
    )


def _line_motion(
    monkeypatch,
    replans: list[list[float]],
    proposals: list[list[float]],
    approvals_left: int = 3,
) -> tuple[_Motion, np.ndarray]:
    cfg = load_mpc_config(_SMOKE_MPC)
    assert cfg.cartesian is not None
    cfg = replace(cfg, cartesian=replace(cfg.cartesian, stall=GoalStallConfig()))
    start = Human().reset_human_with_q(np.zeros(7))
    _line_world(monkeypatch, replans, proposals)
    approach = SimpleNamespace(planning_costs=lambda: None)
    motion = _move_to_goal(
        cfg,
        start,
        PointRegion((0.0, 0.0, 0.0)),
        approach,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        approvals_left,
        "t",
    )
    return motion, motion.path[:, 0]


def test_motion_goes_on_after_an_approved_proposal_falls_short(monkeypatch) -> None:
    """Following an approval that stops short, the arm replans to the goal."""
    motion, progress = _line_motion(
        monkeypatch, replans=[[0.1, 0.1], [0.25, 0.25]], proposals=[[0.1, 0.1, 0.1]]
    )
    np.testing.assert_allclose(progress, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    assert motion.approved_spans == ((2, 5),)
    assert motion.reached and motion.trigger is None and motion.rejected is None


def test_motion_stops_approving_at_the_cap(monkeypatch) -> None:
    """A planner that keeps stalling gets at most ``approvals_left`` approvals."""
    motion, progress = _line_motion(
        monkeypatch,
        replans=[[-0.05]] * 3,
        proposals=[[0.1]] * 2,
        approvals_left=2,
    )
    np.testing.assert_allclose(progress, [0, -0.05, 0.05, 0.0, 0.1, 0.05])
    assert motion.approved_spans == ((1, 2), (3, 4))
    assert not motion.reached and motion.rejected is None


def test_no_proposal_when_leaving_costs_out_gets_no_closer(monkeypatch) -> None:
    """A proposal that does not bring the arm closer to the goal is not offered."""
    motion, progress = _line_motion(monkeypatch, replans=[[0.1]], proposals=[[0.0]])
    np.testing.assert_allclose(progress, [0, 0.1])
    assert not motion.approved_spans
    assert not motion.reached and motion.rejected is None


def test_violations_count_stops_and_each_crossing_in_a_correction(
    monkeypatch,
) -> None:
    """A stop is one crossing; a correction runs whole, so each stretch counts."""
    profiles = {
        3: np.array([0.0, 0.03, 0.03, 0.0, 0.05, 0.01]),  # two stretches
        2: np.array([0.0, 0.01]),  # under the threshold
    }
    monkeypatch.setattr(
        structs_module,
        "compute_violations",
        lambda user, human, trajectory: profiles[len(trajectory) // 2],
    )
    interaction = Interaction.__new__(Interaction)
    interaction.__dict__.update(
        trigger_step=4,
        stall_rejected=False,
        rounds=(
            SimpleNamespace(retrigger_step=7, correction_q=np.zeros((6, 7))),
            SimpleNamespace(retrigger_step=None, correction_q=np.zeros((4, 7))),
        ),
        user=None,
        human=None,
    )
    # The first motion's stop, round 0's re-trigger, and its correction's two.
    assert interaction.violation_count == 4
    interaction.__dict__["stall_rejected"] = True
    assert interaction.violation_count == 3


def test_older_pickles_count_their_approval() -> None:
    """Pickles from before approvals were counted read back with one approval."""
    legacy = Interaction.__new__(Interaction)
    legacy.__dict__.update({"rounds": (), "proposal_approved": True})
    restored = pickle.loads(pickle.dumps(legacy))
    assert restored.approvals == 1 and restored.proposal_approved
    assert restored.user_turns == 1


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
        goal=PointRegion((0.0, 0.0, 0.0)),
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
    retired_terms, correction_text = cost_gen.retired[0]
    assert retired_terms[0] is cost_gen.cost
    assert correction_text == second.utterance.text
    # Turning a proposal down is no violation; only crossings count.
    assert not interaction.stall_rejected
    in_corrections = sum(
        crossings(interaction.user, interaction.human, r.correction_q, 0.02)
        for r in interaction.rounds
    )
    stops = 1 + int(second.retrigger_step is not None)
    assert interaction.violation_count == stops + in_corrections


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
    for rnd in (r for interaction in result for r in interaction.rounds):
        assert rnd.grounding.magnitude == rnd.choice.magnitude == 1.0


@pytest.mark.parametrize("rescale, expected", [(False, 1.0), (True, 0.5)])
def test_keypoint_edit_runs_at_1x_unless_rescaled(
    rescale: bool, expected: float
) -> None:
    """The keypoint edit runs as proposed; ``rescale`` plays the user's size."""
    cfg = load_mpc_config(_SMOKE_MPC)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    nominal_plan = np.repeat(human.q[None], 12, axis=0)
    grounder = LlmKeypointGrounder(rescale=rescale)
    grounder._interpret = (  # type: ignore[method-assign]
        lambda text, scene_context: {
            "joint": "wrist",
            "keypoint": np.array([0.1, 0.3, 0.2]),
        }
    )
    judged: list[list[int]] = []

    def select(means: dict[int, np.ndarray]) -> tuple[int, float]:
        judged.append(list(means))
        return next(iter(means)), 0.5

    result = grounder.ground("move my hand up", human, nominal_plan, select)
    assert judged == [[1]], "the user should still judge the one edit"
    assert result.chosen_label == 1
    assert result.magnitude == expected
    np.testing.assert_allclose(
        result.correction_traj,
        scale_trajectory(result.candidates[1], expected),
    )


def test_keypoint_without_an_edit_holds_at_the_stop_pose() -> None:
    """No keypoint: the nominal plan is judged but not replayed."""
    cfg = load_mpc_config(_SMOKE_MPC)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    nominal_plan = np.repeat(human.q[None], 12, axis=0)
    grounder = LlmKeypointGrounder()
    grounder._interpret = (  # type: ignore[method-assign]
        lambda text, scene_context: None
    )
    judged: list[list[int]] = []

    def select(means: dict[int, np.ndarray]) -> tuple[int, float]:
        judged.append(list(means))
        return next(iter(means)), 1.0

    result = grounder.ground("hmm", human, nominal_plan, select)
    assert judged == [[0]] and result.chosen_label == 0
    np.testing.assert_allclose(
        result.correction_traj, human.arm_aa_from_q(human.q[None])
    )
