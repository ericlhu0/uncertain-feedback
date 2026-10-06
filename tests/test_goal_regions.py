"""Tests for goal regions: distance math, YAML parsing, and planner semantics."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.planners.mpc import ArmMPC, CartesianConfig
from uncertain_feedback.planners.mpc.action_spaces import RolloutBatch
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_feature_series,
    arm_q_from_features,
)
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.goal_spaces import (
    BoxRegion,
    CartesianGoalSpace,
    FeatureRegion,
    ForearmBoxRegion,
    GoalStallConfig,
    PointRegion,
    SphereRegion,
    as_goal_region,
    goal_point,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import Q_DIM, q_to_arm_aa


def _stage_costs(mpc: ArmMPC, q_trajs: np.ndarray) -> np.ndarray:
    batch = RolloutBatch(
        actions=np.zeros((q_trajs.shape[0], q_trajs.shape[1] - 1, Q_DIM)),
        aa_trajs=q_to_arm_aa(q_trajs, mpc._fk.elbow_hinge_axis),
        q_trajs=q_trajs,
    )
    assert mpc._goal_space is not None
    return mpc._goal_space.stage_cost(mpc._extra_costs)(batch)


def test_box_distance_is_zero_inside_and_euclidean_outside() -> None:
    box = BoxRegion(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0))
    wrist = np.array([[0.5, 0.5, 0.5], [1.3, 0.5, 0.5], [1.3, -0.4, 0.5]])
    dist = box.distance(wrist, np.zeros((3, 3, 3)), Human())
    np.testing.assert_allclose(dist, [0.0, 0.3, 0.5])
    np.testing.assert_allclose(box.marker(), [0.5, 0.5, 0.5])
    assert len(box.outline()) == 12


def test_sphere_with_zero_radius_matches_point() -> None:
    center = (0.1, 0.2, 0.3)
    wrist = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.8]])
    human = Human()
    sphere = SphereRegion(center=center, radius=0.0)
    point = PointRegion(point=center)
    aa = np.zeros((2, 3, 3))
    np.testing.assert_allclose(sphere.distance(wrist, aa, human), [0.0, 0.5])
    np.testing.assert_allclose(
        sphere.distance(wrist, aa, human), point.distance(wrist, aa, human)
    )
    np.testing.assert_allclose(
        SphereRegion(center=center, radius=0.2).distance(wrist, aa, human),
        [0.0, 0.3],
    )


def test_feature_region_hinges_on_arm_features() -> None:
    human = Human()
    features = np.array([[1.2, 0.2, 0.3, 0.9, 0.0]])
    q = arm_q_from_features(features, np.zeros(3), human)
    aa = human.arm_aa_from_q(q)
    measured = arm_feature_series(aa, human)["elbow_flexion"][0]

    inside = FeatureRegion(bounds={"elbow_flexion": (1.0, 1.5)})
    below = FeatureRegion(bounds={"elbow_flexion": (measured + 0.2, None)})
    wrist = np.zeros((1, 3))
    np.testing.assert_allclose(inside.distance(wrist, aa, human), [0.0])
    np.testing.assert_allclose(below.distance(wrist, aa, human), [0.2], atol=1e-9)
    assert inside.marker() is None
    assert not inside.outline()


def test_as_goal_region_wraps_points() -> None:
    region = as_goal_region(np.array([0.1, 0.2, 0.3]))
    assert region == PointRegion((0.1, 0.2, 0.3))
    assert as_goal_region(region) is region
    point = goal_point([0.1, 0.2, 0.3])
    assert point is not None
    np.testing.assert_allclose(point, [0.1, 0.2, 0.3])
    assert goal_point(FeatureRegion(bounds={"elbow_flexion": (0.0, 1.0)})) is None


def test_load_mpc_config_parses_each_region(tmp_path) -> None:
    path = tmp_path / "mpc.yaml"
    path.write_text(
        """
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
cartesian:
  goals:
    - [0.1, 0.2, 0.3]
    - box:
        low: [0.0, 0.0, 0.0]
        high: [0.1, 0.2, 0.3]
    - sphere:
        center: [0.1, 0.2, 0.3]
        radius: 0.05
    - features:
        elbow_flexion: [1.0, 1.5]
        shoulder_abduction_adduction: [null, 0.5]
  threshold: 0.02
""",
        encoding="utf-8",
    )
    cfg = load_mpc_config(path)
    assert cfg.cartesian is not None
    assert cfg.cartesian.goals[0] == [0.1, 0.2, 0.3]
    assert cfg.cartesian.goals[1] == BoxRegion(
        low=(0.0, 0.0, 0.0), high=(0.1, 0.2, 0.3)
    )
    assert cfg.cartesian.goals[2] == SphereRegion(center=(0.1, 0.2, 0.3), radius=0.05)
    assert cfg.cartesian.goals[3] == FeatureRegion(
        bounds={
            "elbow_flexion": (1.0, 1.5),
            "shoulder_abduction_adduction": (None, 0.5),
        }
    )
    assert "elbow_flexion" in FEATURE_NAMES


@pytest.mark.parametrize(
    "goal_yaml, match",
    [
        ("{features: {wrist_twist: [0.0, 1.0]}}", r"cartesian.goals\[1\].features"),
        (
            "{box: {low: [0.0, 0.0, 0.0], high: [0.1, -0.1, 0.3]}}",
            r"box.low must be below",
        ),
        (
            "{sphere: {center: [0.0, 0.0, 0.0], radius: 0.0}}",
            r"radius must be positive",
        ),
        ("{cube: {}}", r"cartesian.goals\[1\] must be"),
    ],
)
def test_load_mpc_config_rejects_malformed_regions(tmp_path, goal_yaml, match) -> None:
    path = tmp_path / "mpc.yaml"
    path.write_text(
        f"""
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
cartesian:
  goals:
    - [0.1, 0.2, 0.3]
    - {goal_yaml}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        load_mpc_config(path)


def test_box_goal_inside_costs_only_extra_terms() -> None:
    human = Human()
    wrist0 = human.wrist_from_q(human.q)
    box = BoxRegion(low=tuple(wrist0 - 0.1), high=tuple(wrist0 + 0.1))

    def constant_cost(q_trajs: np.ndarray) -> np.ndarray:
        return np.full(q_trajs.shape[0], 3.0)

    mpc = ArmMPC(
        human,
        cartesian=CartesianConfig(goals=[box]),
        extra_costs=CompositeTrajectoryCost([constant_cost]),
    )
    q_trajs = np.zeros((2, 2, Q_DIM))
    np.testing.assert_allclose(_stage_costs(mpc, q_trajs), [3.0, 3.0])


def test_box_goal_reached_and_queue_advances() -> None:
    human = Human()
    q0 = human.q
    wrist0 = human.wrist_from_q(q0)
    enclosing = BoxRegion(low=tuple(wrist0 - 0.1), high=tuple(wrist0 + 0.1))
    far = BoxRegion(low=tuple(wrist0 + 0.5), high=tuple(wrist0 + 0.6))

    assert ArmMPC(human, cartesian=CartesianConfig(goals=[enclosing])).goal_reached(q0)
    assert not ArmMPC(human, cartesian=CartesianConfig(goals=[far])).goal_reached(q0)

    mpc = ArmMPC(human, cartesian=CartesianConfig(goals=[enclosing, far]))
    assert not mpc.goal_reached(q0)
    popped: list[bool] = []
    assert mpc._goal_space is not None
    region, dist = mpc._goal_space.progress(q0, on_pop=lambda: popped.append(True))
    assert region == far
    assert popped == [True]
    assert dist > 0.05
    assert mpc.current_cartesian_goal == far


def test_goal_stall_needs_a_flat_distance_and_a_still_arm() -> None:
    human = Human()
    q0 = human.q
    far = PointRegion(point=tuple(human.wrist_from_q(q0) + 0.5))
    stall = GoalStallConfig(window=3, min_gain=0.01, max_travel=0.1)

    held = CartesianGoalSpace([far], 0.05, human, stall)
    stalled = []
    for _ in range(4):
        held.progress(q0, on_pop=lambda: None)
        stalled.append(held.stalled)
    held.reset_stall()
    stalled.append(held.stalled)
    assert stalled == [False, False, False, True, False]

    moving = CartesianGoalSpace([far], 0.05, human, stall)
    for i in range(4):
        q = q0.copy()
        q[6] += 0.1 * i
        moving.progress(q, on_pop=lambda: None)
    assert not moving.stalled

    at_goal = CartesianGoalSpace(
        [PointRegion(point=tuple(human.wrist_from_q(q0)))], 0.05, human, stall
    )
    for _ in range(4):
        at_goal.progress(q0, on_pop=lambda: None)
    assert not at_goal.stalled


def test_forearm_box_rejects_wrist_only_success() -> None:
    human = Human()
    q = np.zeros((1, Q_DIM))
    aa = human.arm_aa_from_q(q)
    elbow, wrist = human.fk_positions_from_q(q)[0, -2:] - human.spine3_pos
    region = ForearmBoxRegion(tuple(wrist - 0.01), tuple(wrist + 0.01))
    expected = np.linalg.norm(np.maximum(np.abs(elbow - wrist) - 0.01, 0.0))
    assert expected > 0.1
    np.testing.assert_allclose(region.distance(wrist[None], aa, human), [expected])
    enclosing = ForearmBoxRegion(
        tuple(np.minimum(elbow, wrist) - 0.01),
        tuple(np.maximum(elbow, wrist) + 0.01),
    )
    np.testing.assert_allclose(enclosing.distance(wrist[None], aa, human), [0.0])
    assert len(enclosing.outline()) == 12


def test_care_demo_configs_use_regions() -> None:
    from pathlib import Path

    configs = (
        Path(__file__).resolve().parents[1]
        / "src/uncertain_feedback/planners/mpc/configs"
    )
    bathing = load_mpc_config(configs / "bathing.yaml")
    transfer = load_mpc_config(configs / "transfer.yaml")
    assert bathing.cartesian is not None
    assert transfer.cartesian is not None
    assert isinstance(bathing.cartesian.goals[0], FeatureRegion)
    assert isinstance(transfer.cartesian.goals[0], ForearmBoxRegion)
    assert bathing.pose is None and transfer.pose is not None
    assert transfer.cartesian.goals[0].wrist is not None
    assert transfer.cartesian.goals[0].elbow is None
    bed = load_mpc_config(configs / "transfer_bed.yaml")
    assert bed.pose is not None and bed.pose.exists()
    assert bed.cartesian is not None
    assert isinstance(bed.cartesian.goals[0], ForearmBoxRegion)


def test_forearm_endpoint_boxes_are_simultaneous() -> None:
    human = Human()
    q = np.zeros((1, Q_DIM))
    aa = human.arm_aa_from_q(q)
    elbow, wrist = human.fk_positions_from_q(q)[0, -2:] - human.spine3_pos
    low = tuple(np.minimum(elbow, wrist) - 0.5)
    high = tuple(np.maximum(elbow, wrist) + 0.5)
    elbow_box = BoxRegion(tuple(elbow - 0.01), tuple(elbow + 0.01))
    wrist_box = BoxRegion(tuple(wrist - 0.01), tuple(wrist + 0.01))
    inside = ForearmBoxRegion(low, high, wrist=wrist_box, elbow=elbow_box)
    np.testing.assert_allclose(inside.distance(wrist[None], aa, human), [0.0])
    shifted = np.array([0.1, 0.0, 0.0])
    outside = ForearmBoxRegion(
        low,
        high,
        wrist=BoxRegion(tuple(wrist - 0.01 + shifted), tuple(wrist + 0.01 + shifted)),
        elbow=BoxRegion(tuple(elbow - 0.01 + shifted), tuple(elbow + 0.01 + shifted)),
    )
    np.testing.assert_allclose(
        outside.distance(wrist[None], aa, human), [np.hypot(0.09, 0.09)]
    )
    np.testing.assert_allclose(inside.marker(), wrist)
    assert len(inside.outline()) == 36
