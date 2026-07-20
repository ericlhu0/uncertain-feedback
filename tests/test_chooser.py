"""Tests for the oracle-path cluster-and-magnitude chooser."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa
from uncertain_feedback.simulated_users import (
    HiddenBound,
    SimulatedUser,
    choose_correction,
    feature_series,
)


@pytest.fixture(name="context")
def _context() -> MpcCostContext:
    fk = SmplLeftArmFK()
    return MpcCostContext(fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3))


def _q_traj(flexions: list[float]) -> np.ndarray:
    q = np.zeros((len(flexions), 7), dtype=np.float64)
    q[:, 6] = flexions
    return q


def _aa_mean(context: MpcCostContext, end_flexion: float, n: int = 10) -> np.ndarray:
    q = _q_traj(list(np.linspace(0.0, end_flexion, n)))
    return q_to_arm_aa(q, context.fk.elbow_hinge_axis)


def _elbow_at(context: MpcCostContext, flexion: float) -> float:
    return float(feature_series(context, _q_traj([flexion]))["elbow_flexion"][0])


def _user(context: MpcCostContext, max_flexion: float | None) -> SimulatedUser:
    bounds: tuple[HiddenBound, ...] = ()
    if max_flexion is not None:
        bounds = (
            HiddenBound(
                feature="elbow_flexion",
                bound_type="upper_bound",
                high=_elbow_at(context, max_flexion),
            ),
        )
    return SimulatedUser(name="test", description="", feedback_text="", bounds=bounds)


def test_do_nothing_cluster_loses_to_progress(context: MpcCostContext) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.0, 21)))
    means = {0: _aa_mean(context, 0.0), 1: _aa_mean(context, 0.5)}
    result = choose_correction(_user(context, None), context, means, oracle)
    assert result.label == 1
    assert not result.no_acceptable_cluster
    assert result.acceptable == {0: True, 1: True}
    assert result.scores[1] < result.scores[0]


def test_painful_cluster_is_filtered(context: MpcCostContext) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.2, 21)))
    means = {0: _aa_mean(context, 0.2), 1: _aa_mean(context, 1.2)}
    result = choose_correction(_user(context, 0.45), context, means, oracle)
    assert result.scores[1] < result.scores[0]
    assert result.label == 0
    assert result.acceptable == {0: True, 1: False}
    assert not result.no_acceptable_cluster


def test_all_painful_falls_back_to_least_violating(context: MpcCostContext) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.0, 21)))
    means = {0: _aa_mean(context, 0.4), 1: _aa_mean(context, 1.0)}
    result = choose_correction(_user(context, 0.05), context, means, oracle)
    assert result.no_acceptable_cluster
    assert result.label == 0
    assert result.magnitude == 0.5
    assert result.acceptable == {0: False, 1: False}


def test_grid_finds_scaled_only_acceptable_variant(context: MpcCostContext) -> None:
    oracle = _q_traj(list(np.linspace(0.0, 1.0, 21)))
    means = {0: _aa_mean(context, 1.0)}
    result = choose_correction(_user(context, 0.6), context, means, oracle)
    assert not result.no_acceptable_cluster
    assert result.label == 0
    assert result.magnitude == 0.5
    assert result.acceptable == {0: True}
