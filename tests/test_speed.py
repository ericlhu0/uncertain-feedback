"""Tests for speed limits: features, simulated users, checking and the penalty."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.approaches import Approach, HiddenLimitCostGen, NoCostGen
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.benchmarks.episode import run_episode
from evaluation.benchmarks.speed import (
    ARMS,
    CONSTRAINTS,
    arm_costs,
    run_arm,
    speed_row,
)
from evaluation.benchmarks.structs import InteractionTask
from evaluation.metrics.cost_learning.success import goal_row
from evaluation.metrics.grounding.structs import GroundingResult
from evaluation.metrics.speed import felt_segments, learned_limit_row, limit_violation
from uncertain_feedback.evaluation_mechanism.scoring import rank_candidate_cost
from uncertain_feedback.motion_generators.mdm.mdm_api import N_PREFIX_FRAMES
from uncertain_feedback.motion_generators.mdm.torch_features import (
    build_generated_cost,
    features_from_hml,
    motion_features,
)
from uncertain_feedback.planners.mpc import ArmMPC, CartesianConfig, FeedbackConfig
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    SPEED_FEATURE_NAMES,
    STEP_SECONDS,
    arm_feature_series,
    arm_motion_series,
)
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    TrajectoryCost,
    build_generated_cost_context,
    generated_cost_feature_dependencies,
)
from uncertain_feedback.planners.mpc.goal_spaces import PointRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import Q_DIM, Q_ELBOW
from uncertain_feedback.planners.mpc.rollout import run_planning_loop
from uncertain_feedback.simulated_users import (
    Complaint,
    HiddenBound,
    HiddenSpeedTerm,
    SimulatedUser,
    SpeedLimit,
    feature_series,
    first_complaint,
    first_violation_step,
    get_persona,
    noticed_speed_limit,
    speed_margins,
    verbalize_speed,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_REPO = Path(__file__).resolve().parents[1]

_ELBOW_CAP = (
    "def speed_margins(features, params):\n"
    "    return {'elbow': features['elbow_speed'] - params['cap']}\n"
)
_PATH = (
    "def path_margins(features, params):\n"
    "    return {'bend': features['elbow_flexion'] - params['cap']}\n"
)


def _motion(human: Human, n: int = 30, seed: int = 0) -> np.ndarray:
    """A smooth ``(n, 7)`` arm motion with the clavicle held, as the planner moves."""
    rng = np.random.default_rng(seed)
    start, end = rng.uniform(-0.5, 0.5, size=(2, Q_DIM))
    blend = (1.0 - np.cos(np.linspace(0.0, np.pi, n)))[:, None] / 2.0
    q = start + blend * (end - start)
    q[:, :3] = human.q[:3]
    return q


def _cost(human: Human, code: str, params: dict) -> GeneratedPythonCost:
    return GeneratedPythonCost(
        code, params, build_generated_cost_context(human, human.q[None], window=1)
    )


def _user(*limits: SpeedLimit) -> SimulatedUser:
    return SimulatedUser("test", "", "", bounds=(), speed_limits=limits)


def test_motion_series_is_fk_speed_and_feature_rate_per_second() -> None:
    human = Human()
    q = _motion(human)
    features = arm_motion_series(q, human)

    positions = human.fk_positions_from_q(q)
    for name, joint in (("elbow_speed", 3), ("wrist_speed", 4)):
        expected = (
            np.linalg.norm(np.diff(positions[:, joint], axis=0), axis=1) / STEP_SECONDS
        )
        np.testing.assert_allclose(features[name], expected, atol=1e-9)
    pose = arm_feature_series(q, human)
    for name in FEATURE_NAMES:
        np.testing.assert_allclose(features[name], pose[name][1:])
        np.testing.assert_allclose(
            features[f"{name}_rate"], np.diff(pose[name]) / STEP_SECONDS, atol=1e-9
        )
    assert set(features) == {*FEATURE_NAMES, *SPEED_FEATURE_NAMES}
    # Half the frame time is twice the speed.
    np.testing.assert_allclose(
        arm_motion_series(q, human, dt=STEP_SECONDS / 2)["wrist_speed"],
        2.0 * features["wrist_speed"],
    )


def test_torch_features_agree_with_the_planner_on_the_same_motion() -> None:
    human = Human()
    q = _motion(human, seed=3)
    expected = arm_motion_series(q, human)

    positions = torch.tensor(human.fk_positions_from_q(q))[None]
    # A swaying body carries the whole arm along and must not read as arm speed.
    sway = torch.tensor(np.random.default_rng(1).normal(0.0, 0.05, size=(len(q), 3)))
    spine3, shoulder, elbow, wrist = (
        positions[:, :, joint] + sway for joint in (0, 2, 3, 4)
    )
    features = motion_features(spine3, shoulder, elbow, wrist, STEP_SECONDS)

    for name in ("wrist_speed", "elbow_speed", "elbow_flexion_rate"):
        np.testing.assert_allclose(features[name][0].numpy(), expected[name], atol=1e-5)
    for name in (
        "shoulder_elevation_rate",
        "shoulder_abduction_adduction_rate",
        "shoulder_flexion_extension_rate",
    ):
        np.testing.assert_allclose(features[name][0].numpy(), expected[name], atol=1e-5)


def test_hml_features_count_the_jump_out_of_the_prefix_as_speed() -> None:
    mean, std = torch.zeros(263), torch.ones(263)
    x0 = torch.zeros(1, 263, 1, N_PREFIX_FRAMES + 4)
    wrist_x = 4 + 19 * 3
    x0[0, wrist_x, 0, N_PREFIX_FRAMES:] = 0.1

    features = features_from_hml(x0, mean, std)

    assert features["wrist_speed"].shape == (1, 4)
    assert features["elbow_flexion"].shape == (1, 4)
    torch.testing.assert_close(
        features["wrist_speed"][0],
        torch.tensor([0.1 * 20.0, 0.0, 0.0, 0.0]),
        atol=1e-4,
        rtol=0.0,
    )


def test_generated_cost_reads_speed_in_the_planner_and_in_steering() -> None:
    human = Human()
    cost = _cost(human, _ELBOW_CAP, {"cap": 0.1})
    assert generated_cost_feature_dependencies(_ELBOW_CAP) == ("elbow_speed",)

    q = _motion(human)
    fast = cost.speed_excess(q[None])[0]
    assert fast > 0.0
    assert cost.speed_excess(_motion(human, n=240)[None])[0] == 0.0
    # The planner charges the worst step's excess, in m/s, at the limit weight.
    features = arm_motion_series(q, human)
    assert fast == pytest.approx(features["elbow_speed"].max() - 0.1)
    assert cost(q[None])[0] == pytest.approx(cost.context.limit_weight * fast)
    assert not cost.path_excess(q[None]).any()

    steer = build_generated_cost([cost], torch.zeros(263), torch.ones(263), 1.0)
    assert steer is not None
    x0 = torch.tensor(
        np.random.default_rng(2).normal(size=(3, 263, 1, N_PREFIX_FRAMES + 6)),
        dtype=torch.float32,
    ).requires_grad_()
    values = steer(x0)
    (grad,) = torch.autograd.grad(values.sum(), x0)
    assert values.shape == (3,)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0


def test_speed_limits_are_ordered_with_timing_and_path_limits_without() -> None:
    human = Human()
    q = _motion(human)
    slow = _motion(human, n=240)
    context = build_generated_cost_context(human, slow, window=1, reference_traj=q)

    speed = rank_candidate_cost(context, _cost(human, _ELBOW_CAP, {"cap": 0.1}))
    assert speed is not None and speed.admitted
    assert speed.excess["chosen_correction"] == 0.0

    # Same path at two tempos: a path limit cannot tell them apart.
    cap = float(arm_feature_series(q, human)["elbow_flexion"].mean())
    path = rank_candidate_cost(context, _cost(human, _PATH, {"cap": cap}))
    assert path is not None and not path.admitted
    assert path.excess["chosen_correction"] == pytest.approx(
        path.excess["original_plan"], rel=0.05
    )


def test_admission_is_ordering_only_even_when_the_chosen_motion_exceeds() -> None:
    human = Human()
    q = _motion(human)
    # The chosen correction is slower than the plan but still over a tight cap.
    context = build_generated_cost_context(
        human, _motion(human, n=60), window=1, reference_traj=q
    )

    ranking = rank_candidate_cost(context, _cost(human, _ELBOW_CAP, {"cap": 0.01}))

    assert ranking is not None
    assert 0.0 < ranking.excess["chosen_correction"] < ranking.excess["original_plan"]
    assert ranking.admitted


def test_speed_limit_margin_is_signed_and_moves_with_pose() -> None:
    limit = SpeedLimit(
        "elbow_flexion_rate",
        cap=0.1,
        direction=-1,
        cond_feature="elbow_flexion",
        cond_start=1.0,
        slope=2.0,
    )
    features = {
        "elbow_flexion_rate": np.array([-0.5, 0.5, -0.5]),
        "elbow_flexion": np.array([0.5, 0.5, 1.2]),
    }
    # Extending at 0.5 rad/s: 0.4 over the floor cap; bending is never over;
    # 0.2 rad above cond_start the cap is 0.5, so the same rate is just allowed.
    np.testing.assert_allclose(limit.margin(features), [0.4, -0.6, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        SpeedLimit("wrist_speed", cap=0.0)
    with pytest.raises(ValueError):
        SpeedLimit("elbow_flexion", cap=0.1)


def test_user_notices_a_sustained_speed_not_a_single_fast_step() -> None:
    human = Human()
    user = _user(SpeedLimit("elbow_flexion_rate", cap=0.2, direction=-1))
    q = np.repeat(human.q[None], 20, axis=0)
    # 0.1 rad/s extension throughout, comfortable.
    q[:, Q_ELBOW] -= 0.005 * np.arange(20)
    assert noticed_speed_limit(user, human, q) is None

    # One 0.02 rad step (0.4 rad/s) inside it: over the cap, but too brief.
    spike = q.copy()
    spike[8:, Q_ELBOW] -= 0.015
    assert speed_margins(user, human, spike).max() > 0.0
    assert noticed_speed_limit(user, human, spike) is None

    # 0.4 rad/s from frame 8 on: noticed at frame 9, when the last five steps
    # (three at 0.1, two at 0.4) average over the cap.
    fast = q.copy()
    fast[8:, Q_ELBOW] -= 0.015 * np.arange(1, 13)
    noticed = noticed_speed_limit(user, human, fast, window=5)
    assert noticed is not None
    step, limit = noticed
    assert step == 9
    assert verbalize_speed(limit).text == (
        "slow down, you're straightening my elbow too fast"
    )
    assert noticed_speed_limit(user, human, human.q[None]) is None


def test_first_complaint_is_the_earlier_of_a_path_limit_and_a_noticed_speed() -> None:
    human = Human()
    speed = SpeedLimit("elbow_flexion_rate", cap=0.2, direction=-1)
    q = np.repeat(human.q[None], 20, axis=0)
    q[:, Q_ELBOW] = 1.0 - 0.015 * np.arange(20)  # 0.3 rad/s extension throughout
    flexion = feature_series(human, q)["elbow_flexion"]

    def straightened_past(frame: int) -> SimulatedUser:
        path = HiddenBound("elbow_flexion", "lower_bound", low=float(flexion[frame]))
        return SimulatedUser("test", "", "", bounds=(path,), speed_limits=(speed,))

    # Five steps in, the window is full and over the cap; a path limit crossed
    # earlier is complained of first, and on the same frame the path wins.
    assert first_complaint(straightened_past(12), human, q, 0.0) == Complaint(5, speed)
    assert first_complaint(straightened_past(2), human, q, 0.0) == Complaint(3)
    assert first_complaint(straightened_past(4), human, q, 0.0) == Complaint(5)
    assert first_complaint(_user(speed), human, q[:5]) is None
    path_only = replace(straightened_past(12), speed_limits=())
    assert first_complaint(path_only, human, q, 0.0) == Complaint(
        first_violation_step(path_only, human, q, 0.0) or 0
    )

    # Executed motion before it fills the window: the same motion played on
    # from a fast history is noticed at once.
    history = q[:6] - q[5] + q[0]
    assert first_complaint(_user(speed), human, q, history=history) == Complaint(
        0, speed
    )
    # Taken as played from the current pose: a candidate starting elsewhere is
    # not charged for the jump to its first frame.
    still = np.repeat(human.q[None], 8, axis=0)
    moved = still.copy()
    moved[:, Q_ELBOW] -= 1.0
    assert first_complaint(_user(speed), human, moved, history=still) is None


def test_speed_personas_have_a_phrase_for_every_limit() -> None:
    for name in ("moved_too_fast", "spastic_elbow_flexors"):
        user = get_persona(name)
        assert user.speed_limits and not user.bounds
        for limit in user.speed_limits:
            assert verbalize_speed(limit).form == "speed"


def test_playback_is_exempt_from_a_speed_term() -> None:
    human = Human()
    frames = _motion(human, n=40, seed=5)
    frames = frames - frames[0] + human.q
    user = _user(SpeedLimit("wrist_speed", cap=0.05))

    def play(costs: CompositeTrajectoryCost) -> np.ndarray:
        mpc = ArmMPC(human, extra_costs=costs, feedback=FeedbackConfig())
        mpc.push_trajectory(frames[1:])
        while not mpc.mdm_tracking_complete:
            mpc.step()
        return mpc.human.history

    limited = play(CompositeTrajectoryCost([HiddenSpeedTerm(user, human)]))

    # The chosen correction is followed as chosen, over the limit or not.
    np.testing.assert_array_equal(limited, play(CompositeTrajectoryCost()))
    np.testing.assert_allclose(limited, frames, atol=1e-9)
    assert speed_margins(user, human, limited).max() > 0.0


@pytest.mark.parametrize("persona", sorted(CONSTRAINTS))
def test_hand_written_constraint_matches_the_hidden_limit(persona: str) -> None:
    human = Human()
    user = get_persona(persona)
    (written,) = arm_costs("constraint", user, human, CompositeTrajectoryCost()).terms()
    assert isinstance(written, GeneratedPythonCost)
    q = np.stack([_motion(human, n=12, seed=seed) for seed in range(6)])

    if user.speed_limits:
        q[:, :, Q_ELBOW] = np.abs(q[:, :, Q_ELBOW]) + 0.6
        # Worst-step excess per limit, in the limit's own units.
        hidden = np.maximum(speed_margins(user, human, q), 0.0).max(axis=-1).sum(axis=0)
        assert hidden.max() > 0.0
        np.testing.assert_allclose(written.speed_excess(q), hidden, atol=1e-9)
        assert not written.path_excess(q).any()
    else:
        poses = q.reshape(-1, Q_DIM)
        hidden = user.violation_series(feature_series(human, poses))
        assert hidden.max() > 0.0
        np.testing.assert_allclose(written.pose_excess(poses), hidden, atol=1e-9)
        assert not written.speed_excess(q).any()


def test_reach_is_never_over_a_hard_limit_and_sometimes_over_a_soft_one() -> None:
    human = Human()
    goal = PointRegion(human.wrist_from_q(human.q) + np.array([-0.1, -0.12, 0.1]))
    user = _user(SpeedLimit("wrist_speed", cap=0.1))
    cartesian = CartesianConfig(goals=[goal], threshold=0.03)
    hard = _cost(
        human,
        "def speed_margins(features, params):\n"
        "    return {'hand': features['wrist_speed'] - params['cap']}\n",
        {"cap": 0.1},
    )

    def reach(*terms: TrajectoryCost) -> np.ndarray:
        mpc = ArmMPC(
            human,
            horizon=5,
            n_mpc_samples=64,
            max_angle_delta=0.004,
            extra_costs=CompositeTrajectoryCost(list(terms)),
            seed=0,
            cartesian=cartesian,
        )
        assert run_planning_loop(mpc, 600).reached_goal
        return mpc.human.history

    free = limit_violation(user, human, [reach()])
    soft_q = reach(HiddenSpeedTerm(user, human))
    soft = limit_violation(user, human, [soft_q])
    hard_q = reach(hard)

    # The old squared, time-averaged penalty trades small excesses for progress;
    # the worst-step linear one at a high weight leaves none.
    assert 0.0 < soft["peak_excess"] < 0.5 * free["peak_excess"]
    assert limit_violation(user, human, [hard_q]) == {
        "over_limit_share": 0.0,
        "peak_excess": 0.0,
    }
    assert len(hard_q) > len(soft_q)
    assert np.isnan(limit_violation(user, human, [soft_q[:1]])["peak_excess"])


def test_arms_share_the_playback_and_differ_in_the_reach(tmp_path: Path) -> None:
    config = tmp_path / "mpc.yaml"
    config.write_text(
        "pose: null\nsteps: 600\nhorizon: 5\nn_mpc_samples: 64\n"
        "max_angle_delta: 0.004\nfeedback:\n  max_playback_delta: 0.05\n"
        "cartesian:\n  goals:\n    - [0.0, 0.0, 0.0]\n  threshold: 0.03\n"
    )
    cfg = load_mpc_config(config)
    human = Human()
    user = get_persona("moved_too_fast")
    goal = PointRegion(human.wrist_from_q(human.q) + np.array([-0.1, -0.12, 0.1]))
    correction = _motion(human, n=10, seed=5)
    base = CompositeTrajectoryCost()

    runs = {
        arm: run_arm(
            arm, cfg, human, goal, arm_costs(arm, user, human, base), correction
        )
        for arm in ARMS
    }
    rows = {
        arm: speed_row(run, user, human, runs["none"], 0.02)
        for arm, run in runs.items()
    }

    assert all(run.reached and run.playback_steps > 0 for run in runs.values())
    assert rows["none"]["playback_peak_excess"] > 0.0
    assert rows["none"]["reach_over_limit_share"] > 0.0
    assert rows["none"]["complaint"].startswith("slow down")
    for arm in ("hidden_soft", "constraint"):
        # The correction is played back as chosen; only the reach is limited.
        np.testing.assert_array_equal(runs[arm].playback, runs["none"].playback)
        assert rows[arm]["playback_peak_excess"] == rows["none"]["playback_peak_excess"]
        assert (
            rows[arm]["reach_over_limit_share"] < rows["none"]["reach_over_limit_share"]
        )
    assert rows["constraint"]["reach_over_limit_share"] == 0.0


def test_path_persona_runs_through_the_same_arms() -> None:
    human = Human()
    user = get_persona("elbow_contracture")
    base = CompositeTrajectoryCost()
    soft = arm_costs("hidden_soft", user, human, base, weight=10.0)
    hard = arm_costs(
        "constraint", user, human, base, limit_weight=1000.0, pullback_weight=0.0
    )
    q = np.tile(human.q, (2, 5, 1))
    q[:, :, Q_ELBOW] = 1.0
    q[1, 1:, Q_ELBOW] = [0.8, 0.4, 0.2, 0.3]  # straightens past the limit

    assert soft(q)[0] == hard(q)[0] == 0.0
    flexion = feature_series(human, q[1])["elbow_flexion"]
    assert hard(q)[1] == pytest.approx(1000.0 * (0.5 - flexion.min()))
    assert 0.0 < soft(q)[1] < hard(q)[1]
    assert limit_violation(user, human, [q[1]])["peak_excess"] == pytest.approx(
        0.5 - flexion.min()
    )


class _SlowedDown(Grounder):
    """Offers the interrupted plan as it was and played at half its speed."""

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text
        frames = np.arange(len(nominal_plan))
        halves = np.arange(2 * len(nominal_plan) - 1) / 2.0
        slow = np.stack(
            [np.interp(halves, frames, joint) for joint in nominal_plan.T], axis=1
        )
        candidates = {
            0: human.arm_aa_from_q(nominal_plan),
            1: human.arm_aa_from_q(slow),
        }
        label, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates,
            label,
            magnitude,
            scale_trajectory(candidates[label], magnitude),
        )


class _LearnsTheCap(NoCostGen):
    """Learns the given limit from the first correction, as if generated."""

    def __init__(self, cost: GeneratedPythonCost) -> None:
        super().__init__()
        self.cost = cost
        self.rounds: list[str] = []

    def learned_terms(self) -> list[GeneratedPythonCost]:
        return [self.cost] if self.rounds else []

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        self.rounds.append(ctx.utterance_text)
        return LearnOutcome(cost_accepted=True, unified_installed=False)


def test_a_speed_complaint_runs_the_episode_loop_to_a_learned_cap(
    tmp_path: Path,
) -> None:
    config = tmp_path / "mpc.yaml"
    config.write_text(
        "pose: null\nsteps: 600\nhorizon: 5\nn_mpc_samples: 64\n"
        "max_angle_delta: 0.004\n"
        "cartesian:\n  goals:\n    - [0.0, 0.0, 0.0]\n  threshold: 0.03\n"
    )
    cfg = load_mpc_config(config)
    human = Human()
    limit = SpeedLimit("wrist_speed", cap=0.1)
    user = replace(get_persona("moved_too_fast"), speed_limits=(limit,))
    goal = PointRegion(human.wrist_from_q(human.q) + np.array([-0.1, -0.12, 0.1]))
    task = InteractionTask(user.name, "joint_resolved", (goal,), max_rounds=2, seed=0)
    learner = _LearnsTheCap(
        _cost(
            human,
            "def speed_margins(features, params):\n"
            "    return {'hand': features['wrist_speed'] - params['cap']}\n",
            {"cap": 0.1},
        )
    )
    approach = Approach("learns_the_cap", _SlowedDown(), learner)
    approach.reset(cfg, human, None, user, task.seed, tmp_path / "episode")

    goal_run = run_episode(cfg, human, user, task, approach, tmp_path / "episode")[0]

    (round_,) = goal_run.rounds
    assert goal_run.trigger_step is not None
    assert goal_run.trigger_speed == limit and goal_run.speed_complaints == 1
    assert learner.rounds == [verbalize_speed(limit).text] == [round_.utterance.text]
    # The interrupted plan is refused as too fast; a slowed one is taken.
    assert not round_.choice.acceptable[0] and round_.choice.acceptable[1]
    assert round_.grounding.chosen_label == 1
    # The learned cap holds the replan under the limit to the goal.
    assert goal_run.result == "ok" and goal_run.reached
    assert limit_violation(user, human, [round_.continuation])["peak_excess"] == 0.0
    # The persona felt the motion up to its complaint, then the rewound rest.
    felt = felt_segments(goal_run)
    assert len(felt) == 2 and len(felt[0]) == goal_run.trigger_step + 1
    np.testing.assert_array_equal(felt[1][0], round_.q_feedback)
    row = goal_row(goal_run)
    assert row["speed_complaints"] == 1 and row["felt_peak_excess"] > 0.0


def test_the_ceiling_plans_with_the_hidden_limit_from_the_start(tmp_path: Path) -> None:
    cfg = load_mpc_config(_REPO / "evaluation/conf/care_tasks/elbow_extension.yaml")
    human = Human()
    user = get_persona("spastic_elbow_flexors")
    approach = Approach("hidden_limit_ceiling", _SlowedDown(), HiddenLimitCostGen())
    approach.reset(cfg, human, None, user, 0, tmp_path)
    (limit,) = approach.cost_gen.learned_terms()
    assert (limit.code, limit.params) == CONSTRAINTS[user.name]
    assert limit in approach.planning_costs().terms()


@pytest.mark.parametrize("persona", ["moved_too_fast", "spastic_elbow_flexors"])
def test_the_hidden_limit_written_as_margins_passes_every_check(persona: str) -> None:
    user = get_persona(persona)
    code, params = CONSTRAINTS[persona]
    row = learned_limit_row(code, params, user)
    assert row["speed_cap"] and row["right_feature"] and row["catches"]
    assert row["spares_reverse"] is (
        True if persona == "spastic_elbow_flexors" else None
    )
    both_ways = (
        "def speed_margins(features, params):\n"
        "    return {'elbow': features['elbow_flexion_rate'].abs() - params['cap']}\n"
    )
    assert learned_limit_row(both_ways, {"cap": 0.3}, user)["spares_reverse"] is (
        False if persona == "spastic_elbow_flexors" else None
    )
    path = learned_limit_row(_PATH, {"cap": 1.0}, user)
    assert not (path["speed_cap"] or path["right_feature"] or path["catches"])
