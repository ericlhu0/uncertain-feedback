from __future__ import annotations

import numpy as np

from uncertain_feedback.simulated_users.personas import (
    BICEPS_LONG_HEAD_CONTRACTURE,
    BRACHIAL_PLEXUS_MECHANOSENSITIVITY,
    CROSS_BODY_PAIN,
    OUT_OF_SYNERGY_REACH_PREFERENCE,
    PAINFUL_ARC,
    TRICEPS_LONG_HEAD_CONTRACTURE,
    UNRESTRICTED,
)


def test_painful_arc_uses_plane_agnostic_elevation() -> None:
    bound = PAINFUL_ARC.bounds[0]

    assert bound.feature == "shoulder_elevation"
    assert bound.bound_type == "avoid_band"
    assert bound.low == 1.05
    assert bound.high == 2.1

    features = {
        "shoulder_elevation": np.array([0.5, 1.57, 2.5]),
        "shoulder_abduction_adduction": np.array([0.0, 0.0, 0.0]),
    }

    violations = PAINFUL_ARC.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_cross_body_pain_allowance_drops_with_adduction() -> None:
    bound = CROSS_BODY_PAIN.bounds[0]

    assert bound.feature == "shoulder_elevation"
    assert bound.cond_feature == "shoulder_abduction_adduction"

    features = {
        "shoulder_elevation": np.array([1.5, 1.5, 1.5, 0.3]),
        "shoulder_abduction_adduction": np.array([0.4, -0.2, -0.4, -0.4]),
    }

    violations = CROSS_BODY_PAIN.violation_series(features)

    assert violations[0] == 0.0  # elevated but lateral: unrestricted
    assert violations[1] > 0.0  # elevated at moderate adduction: hurts
    assert violations[2] > violations[1]  # deeper across: hurts more
    assert violations[3] == 0.0  # deep across but carried low: comfortable


def test_triceps_contracture_allows_less_elbow_flexion_when_elevated() -> None:
    features = {
        "elbow_flexion": np.array([1.5, 1.5, 0.8]),
        "shoulder_elevation": np.array([0.5, 2.0, 2.0]),
    }

    violations = TRICEPS_LONG_HEAD_CONTRACTURE.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_out_of_synergy_preference_favors_extension_when_elevated() -> None:
    features = {
        "elbow_flexion": np.array([1.5, 1.5, 0.8]),
        "shoulder_elevation": np.array([0.5, 2.0, 2.0]),
    }

    violations = OUT_OF_SYNERGY_REACH_PREFERENCE.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_biceps_contracture_allows_less_extension_behind_the_body() -> None:
    features = {
        "elbow_flexion": np.array([0.3, 0.3, 0.8]),
        "shoulder_flexion_extension": np.array([0.3, -0.5, -0.5]),
    }

    violations = BICEPS_LONG_HEAD_CONTRACTURE.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_neural_mechanosensitivity_requires_flexion_during_abduction() -> None:
    features = {
        "elbow_flexion": np.array([0.4, 0.4, 1.0]),
        "shoulder_abduction_adduction": np.array([0.2, 1.0, 1.0]),
    }

    violations = BRACHIAL_PLEXUS_MECHANOSENSITIVITY.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_limit_cost_penalizes_out_of_box_rollouts() -> None:
    cost = UNRESTRICTED.limit_cost(weight=1.0)
    shoulder = UNRESTRICTED.joint_limits[0]  # left_shoulder box

    inside = np.zeros((1, 2, 3, 3))
    outside = np.zeros((1, 2, 3, 3))
    outside[:, :, 0, 1] = shoulder.high[1] + 0.3  # shoulder axis-1 past its cap

    assert cost(inside)[0] == 0.0
    assert np.isclose(cost(outside)[0], 0.3)
