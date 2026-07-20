"""Tests for simulated-user feedback attribution."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import (
    CorrectionIntent,
    assert_axis_conventions,
    attribute_correction,
    has_feedback_content,
)


@pytest.fixture(name="context")
def _context() -> MpcCostContext:
    fk = SmplLeftArmFK()
    return MpcCostContext(fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3))


def _shoulder_z_traj(angles: list[float]) -> np.ndarray:
    q = np.zeros((len(angles), 7), dtype=np.float64)
    q[:, 5] = angles
    return q


def test_axis_conventions(context: MpcCostContext) -> None:
    assert_axis_conventions(context.fk)


def test_attribution_finds_join_and_dominant_feature(context: MpcCostContext) -> None:
    # Oracle lowers the arm (shoulder z-rotation 0 -> -0.6); the robot's
    # nominal plan holds it raised (+0.3), so nominal - oracle elevation is
    # positive and the nominal wrist sits above the oracle wrist.
    oracle = _shoulder_z_traj(list(np.linspace(0.0, -0.6, 21)))
    nominal = _shoulder_z_traj([0.3] * 10)
    q_trigger = oracle[7]

    intent = attribute_correction(oracle, nominal, q_trigger, context)
    assert intent.join_index == 7
    deltas = intent.feature_deltas
    assert deltas["shoulder_elevation"] > 0.0
    assert abs(deltas["shoulder_elevation"]) == max(
        abs(value) for value in deltas.values()
    )
    assert intent.wrist_offset[1] > 0.0
    assert has_feedback_content(intent)


def test_min_join_is_respected(context: MpcCostContext) -> None:
    oracle = _shoulder_z_traj(list(np.linspace(0.0, -0.6, 21)))
    nominal = _shoulder_z_traj([0.3] * 10)
    q_trigger = oracle[3]

    free = attribute_correction(oracle, nominal, q_trigger, context, min_join=0)
    forced = attribute_correction(oracle, nominal, q_trigger, context, min_join=10)
    assert free.join_index == 3
    assert forced.join_index == 10


def test_no_content_below_dead_band() -> None:
    intent = CorrectionIntent(
        join_index=0,
        feature_deltas={
            "elbow_flexion": 0.05,
            "shoulder_flexion_extension": -0.1,
            "shoulder_abduction_adduction": 0.0,
            "shoulder_elevation": 0.14,
        },
        wrist_offset=np.zeros(3),
        elbow_offset=np.zeros(3),
    )
    assert not has_feedback_content(intent)
