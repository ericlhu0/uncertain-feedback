from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from uncertain_feedback.planners import run as planner_run
from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC
from uncertain_feedback.planners.mpc.arm_mpc_cartesian import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    ElbowHeightCost,
    MpcCostContext,
    build_extra_costs,
    compute_elbow_heights,
    update_elbow_cost,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.arm_mpc_cartesian_no_mdm import (
    ArmMPCCartesianNoMDM,
)
from uncertain_feedback.uncertainty.base import TrajectoryClusterer


def _write_config(tmp_path, body: str):
    path = tmp_path / "mpc.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _base_yaml(extra: str = "") -> str:
    return f"""
planner: arm_mpc
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
{extra}
"""


def _cost_context(fk: SmplLeftArmFK) -> MpcCostContext:
    return MpcCostContext(
        fk=fk,
        spine3_pos=fk.tpose_spine3_pos,
        spine3_aa=np.zeros(3),
    )


class _FakeMotionGenerator:
    def __init__(self, expected_pose_path: Path) -> None:
        self.expected_pose_path = expected_pose_path
        self.loaded_pose = np.arange(263, dtype=np.float64)
        self.body_pos = np.arange(66, dtype=np.float64).reshape(22, 3)

    def load_hml_pose(self, path: Path) -> np.ndarray:
        assert path == self.expected_pose_path
        return self.loaded_pose

    def decode_pose_with_collar(
        self, pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.testing.assert_allclose(pose, self.loaded_pose)
        return (
            np.ones((3, 3)),
            self.body_pos,
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        )


class _FixedCost:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        assert q_trajs.shape[0] == self._values.shape[0]
        return self._values


class _FakePositionClusterer(TrajectoryClusterer):
    """Cluster all fake position samples into one group."""

    def cluster(self, trajectories: np.ndarray) -> np.ndarray:
        """Unused trajectory clustering path."""
        _ = trajectories
        raise AssertionError("position test should call cluster_positions")

    def cluster_positions(self, positions: np.ndarray) -> np.ndarray:
        """Assign all position samples to cluster 0."""
        assert positions.shape[0] == 2
        return np.zeros(2, dtype=np.intp)


class _FakePositionGenerator:
    """Minimal fake for the UQ position-generation path."""

    def __init__(self, positions: np.ndarray, trajectory: np.ndarray) -> None:
        self.positions = positions
        self.trajectory = trajectory
        self.received_spine3_aa: np.ndarray | None = None

    def generate_left_arm_position_samples(
        self,
        text: str,
        start_pose: np.ndarray | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
    ) -> np.ndarray:
        """Return deterministic fake MDM XYZ samples."""
        assert text
        assert num_samples == self.positions.shape[0]
        _ = start_pose, num_frames, frozen_body
        return self.positions

    def smpl_positions_to_left_arm_trajectory(
        self,
        positions: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Record the base used to convert selected positions."""
        np.testing.assert_allclose(positions, self.positions.mean(axis=0))
        self.received_spine3_aa = spine3_aa
        return self.trajectory


def test_non_mdm_initial_pose_defaults_to_tpose_without_loading_generator() -> None:
    args = Namespace(pose=None, model_path=None)

    def factory(_model_path):
        raise AssertionError("non-MDM without --pose should not load MDM resources")

    gen, state = planner_run._load_initial_pose_state(
        args, uses_mdm=False, motion_generator_factory=factory
    )

    assert gen is None
    np.testing.assert_allclose(state.arm_aa, np.zeros((3, 3)))
    np.testing.assert_allclose(state.fixed_collar_aa, np.zeros(3))
    assert state.body_pos is None
    assert state.spine3_pos is None
    assert state.spine3_aa is None
    assert state.hml_pose is None


def test_non_mdm_initial_pose_uses_pose_when_provided(tmp_path) -> None:
    pose_path = tmp_path / "pose.pt"
    model_path = tmp_path / "model.pt"
    fake_gen = _FakeMotionGenerator(expected_pose_path=pose_path)
    args = Namespace(pose=pose_path, model_path=model_path)

    def factory(received_model_path):
        assert received_model_path == model_path
        return fake_gen

    gen, state = planner_run._load_initial_pose_state(
        args, uses_mdm=False, motion_generator_factory=factory
    )

    assert gen is fake_gen
    np.testing.assert_allclose(state.arm_aa, np.ones((3, 3)))
    np.testing.assert_allclose(state.fixed_collar_aa, [0.4, 0.5, 0.6])
    np.testing.assert_allclose(state.body_pos, fake_gen.body_pos)
    np.testing.assert_allclose(state.spine3_pos, fake_gen.body_pos[9])
    np.testing.assert_allclose(state.spine3_aa, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state.hml_pose, fake_gen.loaded_pose)


def test_initial_pose_uses_config_pose_when_cli_pose_is_omitted(tmp_path) -> None:
    pose_path = tmp_path / "config_pose.pt"
    fake_gen = _FakeMotionGenerator(expected_pose_path=pose_path)
    args = Namespace(pose=None, model_path=None)

    gen, state = planner_run._load_initial_pose_state(
        args,
        uses_mdm=False,
        config_pose=pose_path,
        motion_generator_factory=lambda _model_path: fake_gen,
    )

    assert gen is fake_gen
    np.testing.assert_allclose(state.hml_pose, fake_gen.loaded_pose)


def test_initial_pose_cli_pose_overrides_config_pose(tmp_path) -> None:
    cli_pose_path = tmp_path / "cli_pose.pt"
    config_pose_path = tmp_path / "config_pose.pt"
    fake_gen = _FakeMotionGenerator(expected_pose_path=cli_pose_path)
    args = Namespace(pose=cli_pose_path, model_path=None)

    gen, state = planner_run._load_initial_pose_state(
        args,
        uses_mdm=False,
        config_pose=config_pose_path,
        motion_generator_factory=lambda _model_path: fake_gen,
    )

    assert gen is fake_gen
    np.testing.assert_allclose(state.hml_pose, fake_gen.loaded_pose)


def test_load_mpc_config_with_elbow_height(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  elbow_height:
    min: 0.1
    max: 0.45
    weight: 100
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.planner == "arm_mpc"
    assert cfg.steps == 2
    assert cfg.costs == {"elbow_height": {"min": 0.1, "max": 0.45, "weight": 100}}
    assert cfg.preference_learning is True


def test_load_mpc_config_can_disable_preference_learning(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
preference_learning: false
preference_alpha: 0.25
preference_window: 10
costs:
  elbow_height:
    min: 0.1
    max: 0.45
    weight: 100
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.preference_learning is False
    assert cfg.preference_alpha == 0.25
    assert cfg.preference_window == 10


def test_load_mpc_config_rejects_invalid_preference_learning(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
preference_learning: maybe
"""),
    )

    with pytest.raises(ValueError, match="preference_learning must be a boolean"):
        load_mpc_config(path)


def test_load_mpc_config_rejects_unknown_cost(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  shoulder_spin:
    min: 0
    max: 1
"""),
    )

    with pytest.raises(ValueError, match="Unknown MPC cost"):
        load_mpc_config(path)


def test_build_extra_costs_rejects_invalid_range(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  elbow_height:
    min: 0.5
    max: 0.1
"""),
    )

    cfg = load_mpc_config(path)
    fk = SmplLeftArmFK()

    with pytest.raises(ValueError, match="min must be less than max"):
        build_extra_costs(cfg.costs, _cost_context(fk))


def test_load_mpc_config_rejects_bad_cartesian_goal(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
planner: arm_mpc_cartesian
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
cartesian:
  goals:
    - [0.1, 0.2]
""",
    )

    with pytest.raises(ValueError, match="cartesian.goals"):
        load_mpc_config(path)


def test_load_mpc_config_accepts_no_mdm_cartesian_planner(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
planner: arm_mpc_cartesian_no_mdm
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
pose: src/uncertain_feedback/motion_generators/mdm/demo_pose.pt
cartesian:
  goals:
    - [0.1, 0.2, 0.3]
""",
    )

    cfg = load_mpc_config(path)

    assert cfg.planner == "arm_mpc_cartesian_no_mdm"
    assert cfg.pose == Path("src/uncertain_feedback/motion_generators/mdm/demo_pose.pt")
    assert cfg.cartesian.goals == [[0.1, 0.2, 0.3]]


def test_elbow_height_cost_zero_inside_range() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    context = _cost_context(fk)
    elbow_height = fk.fk(np.zeros((3, 3)))[3, 1] - context.spine3_pos[1]

    cost = ElbowHeightCost(
        min_height=elbow_height - 0.01,
        max_height=elbow_height + 0.01,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_elbow_height_cost_penalizes_outside_range() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    context = _cost_context(fk)
    elbow_height = fk.fk(np.zeros((3, 3)))[3, 1] - context.spine3_pos[1]

    cost = ElbowHeightCost(
        min_height=elbow_height + 0.1,
        max_height=elbow_height + 0.2,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    assert cost(q_trajs)[0] > 0.9


def test_compute_elbow_heights_uses_joint_before_wrist() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    trajectory = np.zeros((1, 3, 3), dtype=np.float64)
    trajectory[0, 0, 2] = 1.0
    positions = fk.fk_batch(
        trajectory,
        context.spine3_pos,
        context.spine3_aa,
    )

    learned_height = compute_elbow_heights(trajectory, context)[0]
    joint_before_wrist_height = positions[0, -2, 1] - context.spine3_pos[1]
    wrist_height = positions[0, -1, 1] - context.spine3_pos[1]

    np.testing.assert_allclose(learned_height, joint_before_wrist_height)
    assert not np.isclose(learned_height, wrist_height)


def test_update_elbow_cost_low_mpc_updates_only_min_to_mdm_5th() -> None:
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=100.0,
        weight=1.0,
        progress_weight=1.0,
        context=_cost_context(SmplLeftArmFK()),
    )
    mdm_heights = np.linspace(50.0, 150.0, 21)
    mpc_heights = np.linspace(0.0, 20.0, 21)

    updated = update_elbow_cost(cost, mdm_heights, mpc_heights, alpha=0.0)

    np.testing.assert_allclose(updated.min_height, 55.0)
    np.testing.assert_allclose(updated.max_height, cost.max_height)


def test_update_elbow_cost_high_mpc_updates_only_max_to_mdm_95th() -> None:
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=100.0,
        weight=1.0,
        progress_weight=1.0,
        context=_cost_context(SmplLeftArmFK()),
    )
    mdm_heights = np.linspace(-50.0, 50.0, 21)
    mpc_heights = np.linspace(100.0, 120.0, 21)

    updated = update_elbow_cost(cost, mdm_heights, mpc_heights, alpha=0.0)

    np.testing.assert_allclose(updated.min_height, cost.min_height)
    np.testing.assert_allclose(updated.max_height, 45.0)


def test_update_elbow_cost_equal_means_leaves_bounds_unchanged() -> None:
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=1.0,
        weight=1.0,
        progress_weight=1.0,
        context=_cost_context(SmplLeftArmFK()),
    )
    mdm_heights = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    mpc_heights = np.array([0.25, 0.5, 0.75], dtype=np.float64)

    updated = update_elbow_cost(cost, mdm_heights, mpc_heights, alpha=1.0)

    assert updated is cost
    np.testing.assert_allclose(updated.min_height, cost.min_height)
    np.testing.assert_allclose(updated.max_height, cost.max_height)


def test_update_elbow_cost_inverted_side_update_falls_back_to_mdm_range() -> None:
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=0.4,
        weight=1.0,
        progress_weight=1.0,
        context=_cost_context(SmplLeftArmFK()),
    )
    mdm_heights = np.linspace(0.5, 1.5, 21)
    mpc_heights = np.linspace(-1.0, 0.0, 21)

    updated = update_elbow_cost(cost, mdm_heights, mpc_heights)

    np.testing.assert_allclose(updated.min_height, 0.55)
    np.testing.assert_allclose(updated.max_height, 1.45)


def test_elbow_height_cost_scores_entire_rollout_not_only_terminal() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    inside = np.zeros((3, 3), dtype=np.float64)
    high = np.zeros((3, 3), dtype=np.float64)
    high[0, 2] = 1.0
    elbow_height = fk.fk(inside)[3, 1] - context.spine3_pos[1]
    q_trajs = np.array(
        [
            [inside, high, inside],
            [inside, inside, inside],
        ],
        dtype=np.float64,
    )
    cost = ElbowHeightCost(
        min_height=elbow_height - 0.01,
        max_height=elbow_height + 0.01,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    costs = cost(q_trajs)

    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1], 0.0)


def test_elbow_height_progress_penalty_only_penalizes_getting_worse_outside() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    low = np.zeros((3, 3), dtype=np.float64)
    low[0, 2] = -1.0
    lower = np.zeros((3, 3), dtype=np.float64)
    lower[0, 2] = -1.5
    less_low = np.zeros((3, 3), dtype=np.float64)
    less_low[0, 2] = -0.5
    q_trajs = np.array(
        [
            [low, lower],
            [low, less_low],
        ],
        dtype=np.float64,
    )
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=0.1,
        weight=0.0,
        progress_weight=100.0,
        context=context,
    )

    costs = cost(q_trajs)

    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1], 0.0)


def test_default_preference_output_path_uses_learned_suffix(tmp_path) -> None:
    config_path = tmp_path / "arm_mpc_cartesian_mdm.yaml"

    output_path = planner_run._default_preference_output_path(config_path)

    assert output_path == tmp_path / "arm_mpc_cartesian_mdm_learned.yaml"


def test_save_learned_preference_yaml_updates_elbow_cost(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        """
planner: arm_mpc_cartesian
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
preference_alpha: 0.25
cartesian:
  goals:
    - [0.1, 0.2, 0.3]
costs:
  elbow_height:
    min: 0.1
    max: 0.4
    weight: 12.0
    progress_weight: 5.0
""",
    )
    output_path = tmp_path / "learned.yaml"
    learned = ElbowHeightCost(
        min_height=0.2,
        max_height=0.6,
        weight=12.0,
        progress_weight=5.0,
        context=_cost_context(SmplLeftArmFK()),
    )

    planner_run._save_learned_preference_yaml(config_path, output_path, learned)

    with open(output_path, encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    assert saved["planner"] == "arm_mpc_cartesian"
    assert saved["preference_alpha"] == 0.25
    assert saved["cartesian"]["goals"] == [[0.1, 0.2, 0.3]]
    assert saved["costs"]["elbow_height"] == {
        "min": 0.2,
        "max": 0.6,
        "weight": 12.0,
        "progress_weight": 5.0,
    }


def test_joint_space_mpc_adds_extra_costs() -> None:
    q_trajs = np.zeros((2, 2, 3, 3), dtype=np.float64)
    target_q = np.zeros((3, 3), dtype=np.float64)
    extra_costs = CompositeTrajectoryCost([_FixedCost([2.0, 3.0])])
    mpc = SmplLeftArmMPC(goals=[target_q], extra_costs=extra_costs)

    np.testing.assert_allclose(mpc._cost(q_trajs, target_q), [2.0, 3.0])


def test_cartesian_mpc_adds_extra_costs() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((2, 2, 3, 3), dtype=np.float64)
    wrist_rel = fk.fk(np.zeros((3, 3)))[-1] - fk.tpose_spine3_pos
    extra_costs = CompositeTrajectoryCost([_FixedCost([4.0, 5.0])])
    mpc = LeftArmMPCCartesian(
        cartesian_goals=[wrist_rel],
        initial_arm_aa=np.zeros((3, 3)),
        fk=fk,
        extra_costs=extra_costs,
    )

    np.testing.assert_allclose(mpc._cartesian_cost(q_trajs), [4.0, 5.0])


def test_cartesian_goal_is_not_relative_to_mdm_endpoint() -> None:
    fk = SmplLeftArmFK()
    spine3_pos = np.array([0.25, 1.0, -0.3], dtype=np.float64)
    spine3_aa = np.zeros(3, dtype=np.float64)
    cartesian_goal = np.array([0.3, 0.5, 0.1], dtype=np.float64)
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    mpc = LeftArmMPCCartesian(
        cartesian_goals=[cartesian_goal],
        initial_arm_aa=np.zeros((3, 3)),
        fk=fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
    )

    target_world_before = mpc._spine3_pos + mpc.current_cartesian_goal
    mdm_endpoint = np.zeros((3, 3), dtype=np.float64)
    mdm_endpoint[0, 1] = 1.0
    mpc.set_mdm_goal(mdm_endpoint)
    mpc.push_trajectory(np.stack([np.zeros((3, 3)), mdm_endpoint]))
    target_world_after = mpc._spine3_pos + mpc.current_cartesian_goal

    np.testing.assert_allclose(target_world_after, target_world_before)
    wrist_rel = fk.fk(np.zeros((3, 3)), spine3_pos, spine3_aa)[-1] - spine3_pos
    expected_cost = ((wrist_rel - cartesian_goal) ** 2).sum()
    np.testing.assert_allclose(mpc._cartesian_cost(q_trajs), [expected_cost])


def test_mdm_push_trajectory_enqueues_every_tenth_frame_and_endpoint() -> None:
    frames = np.arange(23 * 3 * 3, dtype=np.float64).reshape(23, 3, 3)
    mpc = LeftArmMPCMDM()

    mpc.push_trajectory(frames)

    assert len(mpc._goals) == 4
    for queued, expected in zip(mpc._goals, frames[[0, 10, 20, 22]]):
        np.testing.assert_allclose(queued, expected)
    np.testing.assert_allclose(mpc._preview_q, frames[22])


def test_mdm_push_trajectory_does_not_duplicate_stride_endpoint() -> None:
    frames = np.arange(21 * 3 * 3, dtype=np.float64).reshape(21, 3, 3)
    mpc = LeftArmMPCMDM()

    mpc.push_trajectory(frames)

    assert len(mpc._goals) == 3
    for queued, expected in zip(mpc._goals, frames[[0, 10, 20]]):
        np.testing.assert_allclose(queued, expected)
    np.testing.assert_allclose(mpc._preview_q, frames[20])


def test_uq_position_path_converts_selected_mean_with_fixed_mpc_base() -> None:
    """Selected UQ position means are projected into the fixed MPC spine base."""
    fk = SmplLeftArmFK()
    spine3_aa = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    fk.collar_aa = np.array([0.3, 0.1, -0.1], dtype=np.float64)
    positions = np.zeros((2, 3, 22, 3), dtype=np.float64)
    trajectory = np.arange(27, dtype=np.float64).reshape(3, 3, 3) * 0.01
    gen = _FakePositionGenerator(positions, trajectory)
    mpc = LeftArmMPCMDMUQ(
        fk=fk,
        spine3_aa=spine3_aa,
        n_diffusion_samples=2,
        clusterer=_FakePositionClusterer(),
    )

    mpc.query_mdm_with_uncertainty(
        cast(Any, gen),
        "raise my left arm up",
        start_pose=np.zeros(263),
        auto_cluster=0,
    )

    assert gen.received_spine3_aa is not None
    assert mpc.current_goal is not None
    np.testing.assert_allclose(gen.received_spine3_aa, spine3_aa)
    np.testing.assert_allclose(mpc.current_goal, trajectory[0])


def test_no_mdm_cartesian_mpc_adds_extra_costs() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((2, 2, 3, 3), dtype=np.float64)
    wrist_rel = fk.fk(np.zeros((3, 3)))[-1] - fk.tpose_spine3_pos
    extra_costs = CompositeTrajectoryCost([_FixedCost([6.0, 7.0])])
    mpc = ArmMPCCartesianNoMDM(
        cartesian_goals=[wrist_rel],
        initial_arm_aa=np.zeros((3, 3)),
        fk=fk,
        extra_costs=extra_costs,
    )

    np.testing.assert_allclose(mpc._cartesian_cost(q_trajs), [6.0, 7.0])
