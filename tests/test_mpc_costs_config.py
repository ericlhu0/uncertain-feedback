from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from uncertain_feedback.experiments import cluster_comparison
from uncertain_feedback.planners import run as planner_run
from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC
from uncertain_feedback.planners.mpc.arm_mpc_cartesian import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import (
    LeftArmMPCMDMUQ,
    UqClusterResult,
)
from uncertain_feedback.planners.mpc.config import LlmCostConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    ElbowFlexionAngleCost,
    ElbowHeightCost,
    MpcCostContext,
    ShoulderAbductionAngleCost,
    build_extra_costs,
    compute_elbow_flexion_angles,
    compute_elbow_heights,
    compute_shoulder_abduction_angles,
    update_elbow_cost,
    update_preference_cost,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.llm_costs import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
    build_generated_cost_context,
    build_motion_summaries,
    parse_llm_cost_response,
    render_prompt_images,
)
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


class _TwoTrajectoryClusterer(TrajectoryClusterer):
    """Split four fake trajectory samples into two deterministic clusters."""

    def cluster(self, trajectories: np.ndarray) -> np.ndarray:
        assert trajectories.shape[0] == 4
        return np.array([0, 0, 1, 1], dtype=np.intp)


class _FakeLlmModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_images: list[str] | None = None

    def get_full_output(self, text_input: str, image_input=None) -> str:
        self.received_images = image_input
        # image description call — return a plain string
        if "Runtime API" not in text_input:
            return "The arm moves upward in an arc."
        return self.response


class _FakeSequenceLlmModel:
    def __init__(self) -> None:
        self.calls = 0

    def get_full_output(self, text_input: str, image_input=None) -> str:
        if "Runtime API" not in text_input:
            return "The arm moves upward."
        self.calls += 1
        return json.dumps(
            {
                "description": f"fake generated cost {self.calls}",
                "params": {"weight": 1.0},
                "code": (
                    "def cost(q_trajs, context, params):\n"
                    "    future = q_trajs[:, 1:, 0, 0]\n"
                    "    return params['weight'] * np.mean(future ** 2, axis=1)\n"
                ),
            }
        )


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


class _FakeTrajectoryGenerator:
    """Minimal fake for the UQ trajectory-generation path."""

    def __init__(self, trajectories: np.ndarray) -> None:
        self.trajectories = trajectories

    def generate_left_arm_trajectory(
        self,
        text: str,
        start_pose: np.ndarray | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        assert text
        assert num_samples == self.trajectories.shape[0]
        _ = start_pose, num_frames, frozen_body, spine3_aa
        return self.trajectories


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


def test_load_mpc_config_with_elbow_flexion_and_shoulder_abduction(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  elbow_flexion_angle:
    min: 0.4
    max: 1.8
    weight: 50
  shoulder_abduction_angle:
    min: 0.1
    max: 1.2
    weight: 60
    progress_weight: 20
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.costs == {
        "elbow_flexion_angle": {"min": 0.4, "max": 1.8, "weight": 50},
        "shoulder_abduction_angle": {
            "min": 0.1,
            "max": 1.2,
            "weight": 60,
            "progress_weight": 20,
        },
    }


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


def test_elbow_flexion_angle_cost_zero_inside_range() -> None:
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    cost = ElbowFlexionAngleCost(
        min_angle=0.0,
        max_angle=0.1,
        weight=100.0,
        progress_weight=100.0,
        context=_cost_context(SmplLeftArmFK()),
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_elbow_flexion_angle_cost_penalizes_outside_range() -> None:
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    cost = ElbowFlexionAngleCost(
        min_angle=0.4,
        max_angle=0.5,
        weight=100.0,
        progress_weight=100.0,
        context=_cost_context(SmplLeftArmFK()),
    )

    assert cost(q_trajs)[0] > 1.0


def test_shoulder_abduction_angle_cost_zero_inside_range() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    abduction = compute_shoulder_abduction_angles(q_trajs[:, 0], context)[0]
    cost = ShoulderAbductionAngleCost(
        min_angle=abduction - 0.01,
        max_angle=abduction + 0.01,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_shoulder_abduction_angle_cost_penalizes_outside_range() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    abduction = compute_shoulder_abduction_angles(q_trajs[:, 0], context)[0]
    cost = ShoulderAbductionAngleCost(
        min_angle=abduction + 0.1,
        max_angle=abduction + 0.2,
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


def test_compute_elbow_flexion_angles_uses_elbow_joint_row() -> None:
    context = _cost_context(SmplLeftArmFK())
    trajectory = np.zeros((1, 3, 3), dtype=np.float64)
    trajectory[0, 0, 2] = 2.0
    trajectory[0, 1, 0] = 0.3
    trajectory[0, 2, 1] = 4.0

    learned_flexion = compute_elbow_flexion_angles(trajectory, context)[0]

    np.testing.assert_allclose(learned_flexion, 0.3)


def test_compute_shoulder_abduction_angles_changes_with_upper_arm_direction() -> None:
    context = _cost_context(SmplLeftArmFK())
    neutral = np.zeros((1, 3, 3), dtype=np.float64)
    abducted = np.zeros((1, 3, 3), dtype=np.float64)
    abducted[0, 0, 2] = 0.7

    neutral_angle = compute_shoulder_abduction_angles(neutral, context)[0]
    abducted_angle = compute_shoulder_abduction_angles(abducted, context)[0]

    assert not np.isclose(neutral_angle, abducted_angle)


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


def test_update_preference_cost_low_mpc_updates_only_min_to_mdm_5th() -> None:
    cost = ElbowFlexionAngleCost(
        min_angle=0.0,
        max_angle=100.0,
        weight=1.0,
        progress_weight=1.0,
        context=_cost_context(SmplLeftArmFK()),
    )
    mdm_values = np.linspace(50.0, 150.0, 21)
    mpc_values = np.linspace(0.0, 20.0, 21)

    updated = update_preference_cost(cost, mdm_values, mpc_values, alpha=0.0)

    np.testing.assert_allclose(updated.min_value, 55.0)
    np.testing.assert_allclose(updated.max_value, cost.max_value)


def test_update_preference_cost_high_mpc_updates_only_max_to_mdm_95th() -> None:
    cost = ShoulderAbductionAngleCost(
        min_angle=0.0,
        max_angle=100.0,
        weight=1.0,
        progress_weight=1.0,
        context=_cost_context(SmplLeftArmFK()),
    )
    mdm_values = np.linspace(-50.0, 50.0, 21)
    mpc_values = np.linspace(100.0, 120.0, 21)

    updated = update_preference_cost(cost, mdm_values, mpc_values, alpha=0.0)

    np.testing.assert_allclose(updated.min_value, cost.min_value)
    np.testing.assert_allclose(updated.max_value, 45.0)


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


def test_save_learned_preference_yaml_updates_multiple_costs(tmp_path) -> None:
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
  elbow_flexion_angle:
    min: 0.4
    max: 1.8
    weight: 50.0
  shoulder_abduction_angle:
    min: 0.1
    max: 1.2
    weight: 60.0
    progress_weight: 20.0
""",
    )
    output_path = tmp_path / "learned.yaml"
    context = _cost_context(SmplLeftArmFK())
    learned_height = ElbowHeightCost(
        min_height=0.2,
        max_height=0.6,
        weight=12.0,
        progress_weight=5.0,
        context=context,
    )
    learned_flexion = ElbowFlexionAngleCost(
        min_angle=0.5,
        max_angle=1.5,
        weight=50.0,
        progress_weight=50.0,
        context=context,
    )
    learned_abduction = ShoulderAbductionAngleCost(
        min_angle=0.2,
        max_angle=1.0,
        weight=60.0,
        progress_weight=20.0,
        context=context,
    )

    planner_run._save_learned_preference_yaml(
        config_path,
        output_path,
        [learned_height, learned_flexion, learned_abduction],
    )

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
    assert saved["costs"]["elbow_flexion_angle"] == {
        "min": 0.5,
        "max": 1.5,
        "weight": 50.0,
        "progress_weight": 50.0,
    }
    assert saved["costs"]["shoulder_abduction_angle"] == {
        "min": 0.2,
        "max": 1.0,
        "weight": 60.0,
        "progress_weight": 20.0,
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


def test_cartesian_mpc_consumes_final_mdm_goal_then_uses_cartesian_mode() -> None:
    fk = SmplLeftArmFK()
    q0 = np.zeros((3, 3), dtype=np.float64)
    cartesian_goal = fk.fk(q0)[-1] - fk.tpose_spine3_pos
    mpc = LeftArmMPCCartesian(
        cartesian_goals=[cartesian_goal],
        initial_arm_aa=q0,
        fk=fk,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        advance_threshold=0.1,
    )
    mpc.push_trajectory(np.stack([q0]))

    assert not mpc.mdm_tracking_complete
    q1 = mpc.step(q0)

    assert mpc.mdm_tracking_complete
    assert mpc.current_goal is None
    assert len(mpc._goals) == 0

    called = {"cartesian": False}

    def fake_cartesian_solve(current_q):
        called["cartesian"] = True
        plan = np.zeros((mpc._horizon, 3, 3), dtype=np.float64)
        np.testing.assert_allclose(current_q, q1)
        return plan[0], plan

    mpc._cartesian_solve = fake_cartesian_solve  # type: ignore[method-assign]
    mpc.step(q1)

    assert called["cartesian"]


def test_cartesian_mpc_tracking_complete_only_after_playback_exhausts() -> None:
    fk = SmplLeftArmFK()
    q0 = np.zeros((3, 3), dtype=np.float64)
    cartesian_goal = fk.fk(q0)[-1] - fk.tpose_spine3_pos
    mpc = LeftArmMPCCartesian(
        cartesian_goals=[cartesian_goal],
        initial_arm_aa=q0,
        fk=fk,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        max_playback_delta=10.0,  # large cap: each frame reached in one step
    )
    far_goal = np.full((3, 3), 0.5, dtype=np.float64)
    mpc.push_trajectory(np.stack([q0, far_goal]))

    assert not mpc.mdm_tracking_complete
    q1 = mpc.step(q0)
    np.testing.assert_allclose(q1, q0)

    # One frame followed, one remaining: still in playback.
    assert not mpc.mdm_tracking_complete
    assert mpc._playback_idx == 1

    q2 = mpc.step(q1)
    np.testing.assert_allclose(q2, far_goal)

    # Trajectory exhausted: Cartesian mode now engages.
    assert mpc.mdm_tracking_complete
    assert mpc._playback_idx == 2


def test_cartesian_mpc_visualizer_hides_joint_target_and_sets_cartesian_target(
    monkeypatch,
) -> None:
    from uncertain_feedback.utils import plot as plot_module

    class SpyArmVisualizer:
        TARGET_COLOR = "royalblue"
        MDM_COLOR = "darkorange"
        instances = []

        def __init__(self, fk):
            self.fk = fk
            self.open_live_kwargs = None
            self.cartesian_targets = []
            self.step_colors = []
            SpyArmVisualizer.instances.append(self)

        def open_live(self, *args, **kwargs):
            self.open_live_args = args
            self.open_live_kwargs = kwargs

        def start_capture(self):
            pass

        def update_mdm_goal(self, goal_q):
            self.mdm_goal = goal_q

        def update_trajectory_preview(self, preview_q):
            self.preview_q = preview_q

        def update_cartesian_target(self, world_pos):
            self.cartesian_targets.append(np.asarray(world_pos, dtype=np.float64))

        def update_step(self, q, dist, color=TARGET_COLOR):
            self.step_colors.append(color)

    monkeypatch.setattr(plot_module, "ArmVisualizer", SpyArmVisualizer)

    fk = SmplLeftArmFK()
    q0 = np.zeros((3, 3), dtype=np.float64)
    cartesian_goal = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    mpc = LeftArmMPCCartesian(
        cartesian_goals=[cartesian_goal],
        initial_arm_aa=q0,
        fk=fk,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        advance_threshold=0.1,
        visualize=True,
    )
    mpc.push_trajectory(np.stack([np.full((3, 3), 0.5, dtype=np.float64)]))

    mpc.step(q0)

    spy = SpyArmVisualizer.instances[0]
    assert spy.open_live_kwargs["show_target_arm"] is False
    np.testing.assert_allclose(
        spy.cartesian_targets[0],
        fk.tpose_spine3_pos + cartesian_goal,
    )
    assert spy.step_colors == [SpyArmVisualizer.MDM_COLOR]


def test_mdm_push_trajectory_stores_full_trajectory_for_playback() -> None:
    frames = np.arange(23 * 3 * 3, dtype=np.float64).reshape(23, 3, 3)
    final_goal = np.full((3, 3), 0.9, dtype=np.float64)
    mpc = LeftArmMPCMDM(goals=[final_goal])

    mpc.push_trajectory(frames)

    # The full-resolution trajectory is stored for direct playback, not
    # downsampled into the goal queue.
    np.testing.assert_allclose(mpc._playback_frames, frames)
    assert mpc._playback_idx == 0
    assert not mpc.mdm_tracking_complete
    np.testing.assert_allclose(mpc._preview_q, frames[22])
    # The pre-existing final goal is left untouched for the MPC resume phase.
    assert len(mpc._goals) == 1
    np.testing.assert_allclose(mpc._goals[0], final_goal)


def test_mdm_playback_smooth_frames_advance_one_per_step() -> None:
    # Consecutive frames differ by 0.1 rad on the shoulder; with a generous cap
    # each is reached in a single step (smooth motion is not slowed).
    frames = np.array(
        [
            [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    final_goal = np.zeros((3, 3), dtype=np.float64)
    mpc = LeftArmMPCMDM(
        goals=[final_goal],
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        max_playback_delta=1.0,
    )
    mpc.push_trajectory(frames)

    q = np.zeros((3, 3), dtype=np.float64)
    for expected in frames:
        assert not mpc.mdm_tracking_complete
        q = mpc.step(q)
        np.testing.assert_allclose(q, expected, atol=1e-9)

    # Playback exhausted: the MPC resumes sampling toward the final goal.
    assert mpc.mdm_tracking_complete


def _max_joint_rotation(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Largest per-joint geodesic rotation (radians) between two (3, 3) configs."""
    rel = (Rotation.from_rotvec(q_b) * Rotation.from_rotvec(q_a).inv()).as_rotvec()
    return float(np.linalg.norm(rel, axis=1).max())


def test_mdm_playback_caps_large_jump_velocity() -> None:
    # A single far frame: a 1.2 rad shoulder jump must be traversed over many
    # capped steps, never exceeding max_playback_delta per joint per step.
    max_delta = 0.1
    frames = np.array(
        [[[1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    mpc = LeftArmMPCMDM(
        goals=[np.zeros((3, 3))],
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        max_playback_delta=max_delta,
    )
    mpc.push_trajectory(frames)

    q = np.zeros((3, 3), dtype=np.float64)
    n_steps = 0
    while not mpc.mdm_tracking_complete and n_steps < 100:
        prev = q
        q = mpc.step(q)
        assert _max_joint_rotation(prev, q) <= max_delta + 1e-9
        n_steps += 1

    assert n_steps > 1  # not snapped in a single step
    assert mpc.mdm_tracking_complete
    np.testing.assert_allclose(q, frames[0], atol=1e-9)


def test_mdm_playback_eases_in_from_live_pose() -> None:
    # The arm's live pose differs from frames[0]; the first step must ease in
    # (move at most max_playback_delta), not snap straight to frames[0].
    max_delta = 0.1
    current_q = np.zeros((3, 3), dtype=np.float64)
    frames = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    mpc = LeftArmMPCMDM(
        goals=[np.zeros((3, 3))],
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        max_playback_delta=max_delta,
    )
    mpc.push_trajectory(frames)

    q1 = mpc.step(current_q)
    assert _max_joint_rotation(current_q, q1) <= max_delta + 1e-9
    assert not np.allclose(q1, frames[0])  # did not snap to the first frame


def test_mdm_mpc_resumes_toward_final_goal_after_playback() -> None:
    np.random.seed(0)
    frames = np.zeros((2, 3, 3), dtype=np.float64)  # trivial trajectory at origin
    final_goal = np.array(
        [[0.0, 0.6, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    mpc = LeftArmMPCMDM(
        goals=[final_goal],
        horizon=5,
        n_mpc_samples=256,
        max_angle_delta=0.05,
    )
    mpc.push_trajectory(frames)

    q = np.zeros((3, 3), dtype=np.float64)
    for _ in range(len(frames)):  # phase 1: direct playback
        q = mpc.step(q)
    assert mpc.mdm_tracking_complete
    dist_after_playback = float(np.linalg.norm(q - final_goal))

    for _ in range(200):  # phase 2: MPC resumes sampling toward the final goal
        q = mpc.step(q)
    dist_resumed = float(np.linalg.norm(q - final_goal))

    assert dist_resumed < 0.5 * dist_after_playback


def test_mdm_validate_trajectory_warns_on_range_violation() -> None:
    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk,
        spine3_pos=fk.tpose_spine3_pos,
        spine3_aa=np.zeros(3, dtype=np.float64),
    )
    # Constrain elbow flexion to a tight range; a large elbow rotation violates it.
    extra_costs = CompositeTrajectoryCost(
        [
            ElbowFlexionAngleCost(
                min_angle=0.0,
                max_angle=0.1,
                weight=1.0,
                progress_weight=1.0,
                context=context,
            )
        ]
    )
    mpc = LeftArmMPCMDM(goals=[np.zeros((3, 3))], extra_costs=extra_costs)

    safe = np.zeros((4, 3, 3), dtype=np.float64)
    assert mpc.validate_trajectory(safe) == []

    violating = np.zeros((4, 3, 3), dtype=np.float64)
    violating[2, 1, 0] = 1.5  # large elbow axis-angle on frame 2
    warnings = mpc.validate_trajectory(violating)
    assert len(warnings) == 1
    assert "elbow_flexion_angle" in warnings[0]
    assert "frame 2" in warnings[0]


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
    assert mpc._playback_frames is not None
    np.testing.assert_allclose(gen.received_spine3_aa, spine3_aa)
    np.testing.assert_allclose(mpc._playback_frames[0], trajectory[0])


def test_uq_result_contains_all_cluster_mean_trajectories() -> None:
    trajectories = np.zeros((4, 3, 3, 3), dtype=np.float64)
    trajectories[0] = 0.0
    trajectories[1] = 0.2
    trajectories[2] = 1.0
    trajectories[3] = 1.2
    gen = _FakeTrajectoryGenerator(trajectories)
    mpc = LeftArmMPCMDMUQ(
        n_diffusion_samples=4,
        clusterer=_TwoTrajectoryClusterer(),
    )

    chosen = mpc.query_mdm_with_uncertainty(
        cast(Any, gen),
        "move differently",
        start_pose=np.zeros(263),
        auto_cluster=1,
    )

    result = mpc.last_uq_result
    assert result is not None
    assert result.chosen_label == 1
    assert sorted(result.cluster_means) == [0, 1]
    np.testing.assert_allclose(result.cluster_means[0], np.full((3, 3, 3), 0.1))
    np.testing.assert_allclose(result.cluster_means[1], np.full((3, 3, 3), 1.1))
    np.testing.assert_allclose(chosen, result.chosen_mean)


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


def test_load_mpc_config_with_llm_cost(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
llm_cost:
  enabled: true
  model: gpt-test
  strict: true
  artifact_dir: artifacts
  use_images: false
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.llm_cost.enabled is True
    assert cfg.llm_cost.model == "gpt-test"
    assert cfg.llm_cost.strict is True
    assert cfg.llm_cost.artifact_dir == Path("artifacts")
    assert cfg.llm_cost.use_images is False


def test_llm_artifact_run_dir_resolves_relative_to_base_dir(tmp_path) -> None:
    run_dir = planner_run._llm_artifact_run_dir(
        tmp_path,
        Path("llm_cost_artifacts"),
    )

    assert run_dir.parent == tmp_path / "llm_cost_artifacts"


def test_generated_python_cost_executes_with_fk_context() -> None:
    context = build_generated_cost_context(
        _cost_context(SmplLeftArmFK()),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )
    code = """
def cost(q_trajs, context, params):
    positions = context.fk_rollouts(q_trajs)
    elbow = positions[:, 1:, context.joint_index('elbow')]
    target = params['target_elbow_y']
    violation = np.maximum(target - elbow[:, :, 1], 0.0)
    return params['weight'] * np.mean(violation ** 2, axis=1)
"""
    generated = GeneratedPythonCost(
        code=code,
        params={"target_elbow_y": 10.0, "weight": 2.0},
        context=context,
    )
    q_trajs = np.zeros((2, 3, 3, 3), dtype=np.float64)

    costs = generated(q_trajs)

    assert costs.shape == (2,)
    assert np.all(costs > 0.0)


def test_generated_cost_context_named_joint_features_keep_leading_shape() -> None:
    context = build_generated_cost_context(
        _cost_context(SmplLeftArmFK()),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )
    q_trajs = np.zeros((2, 4, 3, 3), dtype=np.float64)

    assert context.elbow_flexion_angles(q_trajs[:, 1:]).shape == (2, 3)
    assert context.shoulder_flexion_extension_angles(q_trajs[:, 1:]).shape == (
        2,
        3,
    )
    assert context.shoulder_abduction_adduction_angles(q_trajs[:, 1:]).shape == (
        2,
        3,
    )
    assert context.shoulder_internal_external_rotation_angles(
        q_trajs[:, 1:]
    ).shape == (2, 3)


def test_generated_cost_context_shoulder_twist_matches_tpose_axis_rotation() -> None:
    fk = SmplLeftArmFK()
    context = build_generated_cost_context(
        _cost_context(fk),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )
    axis = fk.tpose_joints[3] - fk.tpose_joints[2]
    axis = axis / np.linalg.norm(axis)
    trajectory = np.zeros((1, 3, 3), dtype=np.float64)
    trajectory[0, 0] = axis * 0.4

    twist = context.shoulder_internal_external_rotation_angles(trajectory)

    np.testing.assert_allclose(twist, [0.4], atol=1e-10)


def test_generated_cost_context_shoulder_component_angles_are_stable() -> None:
    fk = SmplLeftArmFK()
    context = build_generated_cost_context(
        _cost_context(fk),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )
    neutral = np.zeros((1, 3, 3), dtype=np.float64)
    axis = fk.tpose_joints[3] - fk.tpose_joints[2]
    axis = axis / np.linalg.norm(axis)
    twisted = neutral.copy()
    twisted[0, 0] = axis * 0.4
    adducted = neutral.copy()
    adducted[0, 0, 2] = 0.4

    neutral_flex = context.shoulder_flexion_extension_angles(neutral)[0]
    neutral_abduction = context.shoulder_abduction_adduction_angles(neutral)[0]
    twisted_flex = context.shoulder_flexion_extension_angles(twisted)[0]
    twisted_abduction = context.shoulder_abduction_adduction_angles(twisted)[0]
    adducted_abduction = context.shoulder_abduction_adduction_angles(adducted)[0]

    assert abs(neutral_flex) < 0.2
    assert neutral_abduction > 1.0
    np.testing.assert_allclose(twisted_flex, neutral_flex, atol=1e-10)
    np.testing.assert_allclose(twisted_abduction, neutral_abduction, atol=1e-10)
    assert adducted_abduction < neutral_abduction


def test_motion_summaries_include_named_joint_features() -> None:
    context = build_generated_cost_context(
        _cost_context(SmplLeftArmFK()),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )

    summaries = build_motion_summaries(context)

    assert "joint_features" in summaries["current"]
    assert "joint_features" in summaries["mdm_traj"]
    assert "shoulder_flexion_extension" in summaries["mdm_traj"]["joint_features"]
    assert "shoulder_abduction_adduction" in summaries["mdm_traj"]["joint_features"]
    assert (
        "shoulder_internal_external_rotation"
        in summaries["mdm_traj"]["joint_features"]
    )


def test_generated_python_cost_rejects_bad_shape() -> None:
    context = build_generated_cost_context(
        _cost_context(SmplLeftArmFK()),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )
    generated = GeneratedPythonCost(
        code="def cost(q_trajs, context, params):\n    return np.zeros((q_trajs.shape[0], 1))",
        params={},
        context=context,
    )

    with pytest.raises(GeneratedCostValidationError, match="shape"):
        generated(np.zeros((2, 3, 3, 3), dtype=np.float64))


def test_prompt_images_render_overlay(tmp_path) -> None:
    context = build_generated_cost_context(
        _cost_context(SmplLeftArmFK()),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )

    paths = render_prompt_images(context, tmp_path)

    assert [path.name for path in paths] == ["overlay.png"]
    assert paths[0].exists() and paths[0].stat().st_size > 0


def test_apply_llm_generated_cost_with_fake_model(tmp_path) -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    mdm_traj = np.zeros((3, 3, 3), dtype=np.float64)
    generated_context = build_generated_cost_context(
        context,
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=mdm_traj,
        q_history=[],
        window=5,
    )
    response = {
        "description": "penalize elbow below target height",
        "explanation": "The cost keeps the elbow above the demonstrated target.",
        "recipient_explanation": (
            "I will help keep your elbow lifted while your arm moves upward."
        ),
        "params": {"target_elbow_height": 0.1, "weight": 10.0},
        "code": "def cost(q_trajs, context, params):\n    positions = context.fk_rollouts(q_trajs)\n    elbow = positions[:, 1:, context.joint_index('elbow')]\n    violation = np.maximum(params['target_elbow_height'] - elbow[:, :, 1], 0.0)\n    return params['weight'] * np.mean(violation ** 2, axis=1)\n",
    }
    fake_model = _FakeLlmModel(json.dumps(response))
    mpc = SmplLeftArmMPC(goals=[np.zeros((3, 3))])
    llm_cfg = LlmCostConfig(
        enabled=True,
        artifact_dir=tmp_path / "artifacts",
    )

    generated = planner_run._apply_llm_generated_cost(
        mpc,
        "raise the elbow",
        mdm_traj,
        np.zeros((3, 3), dtype=np.float64),
        [],
        context,
        llm_cfg,
        tmp_path,
        history_window=5,
        llm_model_factory=lambda _model_name: fake_model,
    )

    assert generated is not None
    assert len(mpc._extra_costs.terms()) == 1
    artifact_dirs = list((tmp_path / "artifacts").iterdir())
    assert len(artifact_dirs) == 1
    assert (artifact_dirs[0] / "prompt.txt").exists()
    assert (artifact_dirs[0] / "cost.py").exists()
    assert (artifact_dirs[0] / "recipient_explanation.txt").read_text(
        encoding="utf-8"
    ) == response["recipient_explanation"]
    with open(artifact_dirs[0] / "params.json", encoding="utf-8") as f:
        params = json.load(f)
    assert params["recipient_explanation"] == response["recipient_explanation"]
    assert fake_model.received_images is not None


def test_llm_cluster_experiment_generates_bundle_and_rollout_per_cluster(
    tmp_path,
) -> None:
    cfg = load_mpc_config(
        _write_config(
            tmp_path,
            """
planner: arm_mpc_cartesian
steps: 4
horizon: 2
n_mpc_samples: 3
max_angle_delta: 0.001
goal_threshold: 0.1
advance_threshold: 0.1
trajectory_fraction: 1.0
cartesian:
  goals:
    - [0.0, 0.0, 0.0]
llm_cost:
  enabled: true
  artifact_dir: artifacts
  use_images: false
""",
        )
    )
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    cluster_means = {
        0: np.zeros((3, 3, 3), dtype=np.float64),
        1: np.full((3, 3, 3), 0.05, dtype=np.float64),
    }
    uq_result = UqClusterResult(
        chosen_label=1,
        labels=np.array([0, 0, 1, 1], dtype=np.intp),
        cluster_means=cluster_means,
    )
    mpc = SmplLeftArmMPC(goals=[np.zeros((3, 3), dtype=np.float64)])
    fake_model = _FakeSequenceLlmModel()

    selected = cluster_comparison.run_cluster_comparison(
        mpc,
        cfg,
        "prefer the demonstrated shape",
        uq_result,
        np.zeros((3, 3), dtype=np.float64),
        [],
        context,
        tmp_path,
        history_window=5,
        rollout_steps=2,
        body_pos=None,
        spine3_pos=fk.tpose_spine3_pos,
        spine3_aa=np.zeros(3, dtype=np.float64),
        llm_model_factory=lambda _model_name: fake_model,
    )

    assert selected is not None
    assert fake_model.calls == 2
    assert len(mpc._extra_costs.terms()) == 1
    root_dirs = list((tmp_path / "artifacts").iterdir())
    assert len(root_dirs) == 1
    root = root_dirs[0]
    assert (root / "selected_cluster.txt").read_text(encoding="utf-8").strip() == "1"
    for label in (0, 1):
        cluster_dir = root / f"cluster_{label}"
        assert (cluster_dir / "prompt.txt").exists()
        assert (cluster_dir / "cost.py").exists()
        assert (cluster_dir / "rollout.npy").exists()
        assert (cluster_dir / "metrics.json").exists()
        rollout = np.load(cluster_dir / "rollout.npy")
        assert rollout.shape == (3, 3, 3)
    with open(root / "comparison_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["selected_cluster"] == 1
    assert summary["cluster_ids"] == [0, 1]
    assert set(summary["clusters"]) == {"0", "1"}
    assert summary["clusters"]["0"]["validation"]["ok"] is True
    assert summary["clusters"]["1"]["rollout_metrics"]["steps_completed"] == 2


def test_parse_llm_cost_response_accepts_markdown_json() -> None:
    response = parse_llm_cost_response(
        '```json\n{"description":"d","code":"def cost(q_trajs, context, params):\\n    return np.zeros(q_trajs.shape[0])","params":{},"recipient_explanation":"plain"}\n```'
    )

    assert response.description == "d"
    assert response.recipient_explanation == "plain"
    assert "def cost" in response.code
