"""Tests for the oracle-path cluster-and-magnitude chooser."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import (
    ATTRIBUTED_FEATURES,
    ChoiceResult,
    CorrectionIntent,
    HiddenBound,
    SimulatedUser,
    SpeedLimit,
    Utterance,
    choose_correction,
    feature_series,
)


@pytest.fixture(name="human")
def _human() -> Human:
    return Human()


def _q_traj(flexions: list[float]) -> np.ndarray:
    q = np.zeros((len(flexions), 7), dtype=np.float64)
    q[:, 6] = flexions
    return q


def _aa_mean(human: Human, end_flexion: float, n: int = 10) -> np.ndarray:
    q = _q_traj(list(np.linspace(0.0, end_flexion, n)))
    return human.arm_aa_from_q(q)


def _elbow_at(human: Human, flexion: float) -> float:
    return float(feature_series(human, _q_traj([flexion]))["elbow_flexion"][0])


def _user(human: Human, max_flexion: float | None) -> SimulatedUser:
    bounds: tuple[HiddenBound, ...] = ()
    if max_flexion is not None:
        bounds = (
            HiddenBound(
                feature="elbow_flexion",
                bound_type="upper_bound",
                high=_elbow_at(human, max_flexion),
            ),
        )
    return SimulatedUser(name="test", description="", feedback_text="", bounds=bounds)


def test_do_nothing_cluster_loses_to_progress(human: Human) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.0, 21)))
    means = {0: _aa_mean(human, 0.0), 1: _aa_mean(human, 0.5)}
    result = choose_correction(
        _user(human, None), human, means, oracle, mode="progress"
    )
    assert result.label == 1
    assert not result.no_acceptable_cluster
    assert result.acceptable == {0: True, 1: True}
    assert result.scores[1] < result.scores[0]


def test_painful_cluster_is_filtered(human: Human) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.2, 21)))
    means = {0: _aa_mean(human, 0.2), 1: _aa_mean(human, 1.2)}
    result = choose_correction(
        _user(human, 0.45), human, means, oracle, mode="progress"
    )
    assert result.scores[1] < result.scores[0]
    assert result.label == 0
    assert result.acceptable == {0: True, 1: False}
    assert not result.no_acceptable_cluster


def test_all_painful_falls_back_to_least_violating(human: Human) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.0, 21)))
    means = {0: _aa_mean(human, 0.4), 1: _aa_mean(human, 1.0)}
    result = choose_correction(
        _user(human, 0.05), human, means, oracle, mode="progress"
    )
    assert result.no_acceptable_cluster
    assert result.label == 0
    assert result.magnitude == 0.5
    assert result.acceptable == {0: False, 1: False}


def test_grid_finds_scaled_only_acceptable_variant(human: Human) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.0, 21)))
    means = {0: _aa_mean(human, 1.0)}
    result = choose_correction(_user(human, 0.6), human, means, oracle, mode="progress")
    assert not result.no_acceptable_cluster
    assert result.label == 0
    assert result.magnitude == 0.5
    assert result.acceptable == {0: True}


def test_robot_top_runs_the_first_candidate_as_proposed(human: Human) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.2, 21)))
    # Listed painful one first, as the grounder would its most likely reading.
    means = {1: _aa_mean(human, 1.2), 0: _aa_mean(human, 0.2)}
    result = choose_correction(
        _user(human, 0.45), human, means, oracle, mode="robot_top"
    )
    assert (result.label, result.magnitude) == (1, 1.0)
    assert result.acceptable == {0: True, 1: False}
    assert not result.no_acceptable_cluster

    result = choose_correction(
        _user(human, 0.05), human, means, oracle, mode="robot_top"
    )
    assert (result.label, result.magnitude) == (1, 1.0)
    assert result.no_acceptable_cluster


def _extension_user(cap: float) -> SimulatedUser:
    return SimulatedUser(
        name="test",
        description="",
        feedback_text="",
        bounds=(),
        speed_limits=(SpeedLimit("elbow_flexion_rate", cap=cap, direction=-1),),
    )


def test_a_candidate_that_moves_too_fast_is_refused_at_its_size(human: Human) -> None:
    # Straightening 0.02 rad a step (0.4 rad/s); the user tolerates 0.35 rad/s,
    # so 3/4 of the motion in the same time.
    oracle = _q_traj(list(np.linspace(1.0, 0.6, 21)))
    means = {0: human.arm_aa_from_q(_q_traj(list(np.linspace(1.0, 0.6, 21))))}
    result = choose_correction(
        _extension_user(0.35), human, means, oracle, mode="progress"
    )
    assert not result.no_acceptable_cluster
    assert (result.label, result.magnitude) == (0, 0.75)


def test_speed_is_judged_with_the_executed_history_before_it(human: Human) -> None:
    # Two fast steps alone are too brief to notice; after three fast executed
    # steps they complete a noticed window.
    oracle = _q_traj(list(np.linspace(1.0, 0.9, 6)))
    means = {0: human.arm_aa_from_q(_q_traj([1.0, 0.95, 0.9]))}
    user = _extension_user(0.3)
    alone = choose_correction(user, human, means, oracle, magnitudes=(1.0,))
    history = _q_traj([1.15, 1.1, 1.05, 1.0])
    after = choose_correction(
        user, human, means, oracle, magnitudes=(1.0,), history=history
    )
    assert alone.acceptable == {0: True}
    assert after.acceptable == {0: False} and after.no_acceptable_cluster


def test_with_nothing_acceptable_the_least_fast_candidate_is_taken(
    human: Human,
) -> None:
    oracle = _q_traj(list(np.linspace(1.0, 0.2, 21)))
    means = {
        0: human.arm_aa_from_q(_q_traj(list(np.linspace(1.0, 0.2, 9)))),
        1: human.arm_aa_from_q(_q_traj(list(np.linspace(1.0, 0.2, 21)))),
    }
    result = choose_correction(
        _extension_user(0.05), human, means, oracle, mode="progress"
    )
    assert result.no_acceptable_cluster
    assert (result.label, result.magnitude) == (1, 0.5)


def _intent(elbow_delta: float) -> CorrectionIntent:
    deltas = {name: 0.0 for name in ATTRIBUTED_FEATURES}
    deltas["elbow_flexion"] = elbow_delta
    return CorrectionIntent(0, deltas, np.zeros(3), np.zeros(3))


def test_cost_to_go_picks_fewest_steps_among_comfortable(human: Human) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.2, 21)))
    straighten = human.arm_aa_from_q(_q_traj(list(np.linspace(0.3, 0.1, 10))))
    # Bends the elbow a little but mostly raises the shoulder.
    raise_q = _q_traj(list(np.linspace(0.0, 0.1, 10)))
    raise_q[:, 2] = np.linspace(0.0, 0.6, 10)
    means = {
        0: _aa_mean(human, 0.2),
        1: _aa_mean(human, 0.4),
        2: _aa_mean(human, 1.2),
        3: straighten,
        4: human.arm_aa_from_q(raise_q),
    }

    # The further the elbow flexes or the shoulder rises the closer the goal;
    # cluster 2 gets there first but hurts at every magnitude, cluster 4 next.
    def steps(candidate: np.ndarray) -> float:
        end = human.q_from_arm_aa(candidate)[-1]
        return 10.0 - 10.0 * float(end[6] + end[2])

    def choose(intent: CorrectionIntent, text: str) -> ChoiceResult:
        return choose_correction(
            _user(human, 0.45),
            human,
            means,
            oracle,
            mode="cost_to_go",
            intent=intent,
            steps_to_goal=steps,
            utterance=Utterance(text, "joint_resolved"),
        )

    result = choose(_intent(-0.3), "bend my elbow more")
    assert result.acceptable == {0: True, 1: True, 2: False, 3: True, 4: True}
    # Cluster 4 bends the elbow too, but its largest motion is the shoulder.
    assert (result.label, result.magnitude) == (1, 1.0)
    # Asked to straighten, only cluster 3 moves the way the words say.
    assert choose(_intent(0.3), "straighten my elbow more").label == 3
    with pytest.raises(ValueError):
        choose_correction(_user(human, 0.45), human, means, oracle, mode="cost_to_go")
