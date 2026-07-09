from __future__ import annotations

import numpy as np

from uncertain_feedback.simulated_users.personas import PAINFUL_ARC, UNRESTRICTED


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


def test_limit_cost_penalizes_out_of_box_rollouts() -> None:
    cost = UNRESTRICTED.limit_cost(weight=1.0)
    shoulder = UNRESTRICTED.joint_limits[0]  # left_shoulder box

    inside = np.zeros((1, 2, 3, 3))
    outside = np.zeros((1, 2, 3, 3))
    outside[:, :, 0, 1] = shoulder.high[1] + 0.3  # shoulder axis-1 past its cap

    assert cost(inside)[0] == 0.0
    assert np.isclose(cost(outside)[0], 0.3)
