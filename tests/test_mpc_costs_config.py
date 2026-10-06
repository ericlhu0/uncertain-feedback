"""Tests for MPC cost terms, planner behavior, and YAML config loading."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from uncertain_feedback.consts import MDM_ROOT, MDM_START_POSE_PATH
from uncertain_feedback.cost_generation import (
    artifact_run_dir,
    build_motion_summaries,
    create_cost_generator,
    render_prompt_images,
)
from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import HML_STATS_DIR
from uncertain_feedback.planners import run as planner_run
from uncertain_feedback.planners.mpc import ArmMPC, CartesianConfig, FeedbackConfig
from uncertain_feedback.planners.mpc.action_spaces import RolloutBatch
from uncertain_feedback.planners.mpc.config import LlmCostConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    ElbowFlexionAngleCost,
    ElbowHeightCost,
    GeneratedCostValidationError,
    GeneratedPythonCost,
    ShoulderAbductionAngleCost,
    build_extra_costs,
    build_generated_cost_context,
    compute_elbow_flexion_angles,
    compute_elbow_heights,
    compute_shoulder_abduction_angles,
    parse_llm_cost_response,
    update_elbow_cost,
    update_preference_cost,
)
from uncertain_feedback.planners.mpc.feedback import MdmFeedback
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    Q_DIM,
    Q_ELBOW,
    SmplLeftArmFK,
)
from uncertain_feedback.planners.mpc.rollout import run_planning_loop
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    JointBoxLimit,
    SimulatedUser,
    compute_violations,
)
from uncertain_feedback.uncertainty import UqConfig
from uncertain_feedback.uncertainty.cluster_picker import ClusterPickResult
from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer


def _write_config(tmp_path, body: str):
    path = tmp_path / "mpc.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _base_yaml(extra: str = "") -> str:
    return f"""
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
{extra}
"""


_requires_hml_stats = pytest.mark.skipif(
    not (HML_STATS_DIR / "Mean.npy").exists(),
    reason="HumanML3D normalization statistics not available",
)


def _build_run(
    tmp_path,
    *,
    pose: Path | None = None,
    arm: Path | None = None,
    config_pose: Path | None = None,
) -> planner_run.RunSetup:
    """Build a non-MDM run whose generator factory must never be called."""
    extra = f"pose: {'null' if config_pose is None else config_pose}\n"
    cfg = load_mpc_config(_write_config(tmp_path, _base_yaml(extra)))
    args = Namespace(live=False, save=None, pose=pose, arm=arm, model_path=None)

    def factory(_model_path):
        raise AssertionError("non-MDM runs should not load MDM resources")

    return planner_run.build_run(args, cfg, motion_generator_factory=factory)


def _arm_positions(human: Human, q_trajs: np.ndarray) -> np.ndarray:
    """``(..., 7)`` arm states as ``(..., 22, 3)`` samples with the arm chain set."""
    positions = np.zeros((*q_trajs.shape[:-1], 22, 3), dtype=np.float64)
    positions[..., LEFT_ARM_CHAIN_INDICES, :] = human.fk_positions_from_q(q_trajs)
    return positions


def _elbow_trajectories(elbow_angles: list[float]) -> np.ndarray:
    """``(N, 3, 7)`` constant trajectories bending only the elbow."""
    q_trajs = np.zeros((len(elbow_angles), 3, Q_DIM), dtype=np.float64)
    q_trajs[..., Q_ELBOW] = np.asarray(elbow_angles)[:, None]
    return q_trajs


def _stage_costs(mpc: ArmMPC, q_trajs: np.ndarray) -> np.ndarray:
    """Evaluate the goal space's stage cost on raw ``(N, H+1, 7)`` rollouts."""
    batch = RolloutBatch(
        actions=np.zeros((q_trajs.shape[0], q_trajs.shape[1] - 1, Q_DIM)),
        aa_trajs=mpc.human.arm_aa_from_q(q_trajs),
        q_trajs=q_trajs,
    )
    assert mpc._goal_space is not None
    return mpc._goal_space.stage_cost(mpc._extra_costs)(batch)


def _goal_marker(mpc: ArmMPC) -> np.ndarray:
    goal = mpc.current_cartesian_goal
    assert goal is not None
    marker = goal.marker()
    assert marker is not None
    return marker


def _playback(mpc: ArmMPC) -> MdmFeedback:
    assert mpc._feedback is not None
    return mpc._feedback


def _joint_limit_user() -> SimulatedUser:
    return SimulatedUser(
        name="test_user",
        description="test",
        feedback_text="keep that joint comfortable",
        bounds=(),
        joint_limits=(
            JointBoxLimit(
                joint="left_elbow",
                low=(-1.0, -1.0, -1.0),
                high=(0.15, 1.0, 1.0),
            ),
        ),
    )


class _FixedCost:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        assert q_trajs.shape[0] == self._values.shape[0]
        return self._values


class _FakePositionClusterer(TrajectoryClusterer):
    """Cluster all fake position samples into one group."""

    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        raise AssertionError("position test should cluster positions")

    def _positions_to_features(self, positions: np.ndarray) -> np.ndarray:
        return positions.reshape(positions.shape[0], -1)

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[0] == 2
        return np.zeros(2, dtype=np.intp)


# _positions_to_features is an optional hook (see supports_positions) that
# pylint >= 4.1 reports as unimplemented abstract.
class _TwoTrajectoryClusterer(TrajectoryClusterer):  # pylint: disable=abstract-method
    """Split four fake trajectory samples into two deterministic clusters."""

    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        return trajectories.reshape(trajectories.shape[0], -1)

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[0] == 4
        return np.array([0, 0, 1, 1], dtype=np.intp)


class _FakeLlmModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_images: list[str] | None = None

    def get_full_output(self, text_input: str, image_input=None) -> str:
        if image_input is not None:
            self.received_images = image_input
        # image description call — return a plain string
        if "Runtime API" not in text_input:
            return "The arm moves upward in an arc."
        return self.response


class _FakePositionGenerator:
    """Minimal fake for the UQ generation path."""

    def __init__(self, positions: np.ndarray) -> None:
        self.positions = positions

    def generate_positions(
        self,
        text: str,
        human: Human,
        *,
        prefix: bool,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
    ) -> np.ndarray:
        """Return deterministic fake MDM XYZ samples."""
        assert text
        assert num_samples == self.positions.shape[0]
        _ = human, prefix, num_frames, frozen_body
        return self.positions


def test_non_mdm_initial_pose_defaults_to_tpose_without_loading_generator(
    tmp_path,
) -> None:
    setup = _build_run(tmp_path)

    human, tpose = setup.human, Human()
    assert setup.gen is None
    np.testing.assert_allclose(human.q, np.zeros(Q_DIM))
    np.testing.assert_allclose(human.fk.collar_aa, np.zeros(3))
    np.testing.assert_allclose(human.posture, tpose.posture)
    np.testing.assert_allclose(human.spine3_pos, tpose.spine3_pos)
    np.testing.assert_allclose(human.spine3_aa, np.zeros(3))
    assert human.hml_pose is None


@_requires_hml_stats
def test_non_mdm_initial_pose_uses_pose_when_provided(tmp_path) -> None:
    setup = _build_run(tmp_path, pose=MDM_START_POSE_PATH)

    human, expected = setup.human, Human(pose=MDM_START_POSE_PATH)
    assert setup.gen is None
    np.testing.assert_allclose(human.q, expected.q)
    np.testing.assert_allclose(human.fk.collar_aa, expected.fk.collar_aa)
    np.testing.assert_allclose(human.posture, expected.posture)
    np.testing.assert_allclose(human.spine3_pos, expected.posture[9])
    np.testing.assert_allclose(human.spine3_aa, expected.spine3_aa)
    assert human.hml_pose is not None and expected.hml_pose is not None
    np.testing.assert_allclose(human.hml_pose, expected.hml_pose)


@_requires_hml_stats
def test_initial_pose_uses_config_pose_when_cli_pose_is_omitted(tmp_path) -> None:
    config_pose = MDM_ROOT / "demo_pose.pt"
    setup = _build_run(tmp_path, config_pose=config_pose)

    np.testing.assert_allclose(setup.human.posture, Human(pose=config_pose).posture)


@_requires_hml_stats
def test_initial_pose_cli_pose_overrides_config_pose(tmp_path) -> None:
    setup = _build_run(
        tmp_path, pose=MDM_START_POSE_PATH, config_pose=MDM_ROOT / "demo_pose.pt"
    )

    np.testing.assert_allclose(
        setup.human.posture, Human(pose=MDM_START_POSE_PATH).posture
    )


def test_arm_override_rejects_unexpected_shape(tmp_path) -> None:
    arm_path = tmp_path / "arm.npy"
    np.save(arm_path, np.zeros((5, 3), dtype=np.float64))

    with pytest.raises(ValueError, match=r"arm must have shape \(3, 3\)"):
        _build_run(tmp_path, arm=arm_path)


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

    assert cfg.cartesian is None and cfg.feedback is None
    assert cfg.steps == 2
    assert cfg.costs == {"elbow_height": {"min": 0.1, "max": 0.45, "weight": 100}}
    assert cfg.preference_learning is True
    assert cfg.seed == 0


def test_seeded_mpc_sampling_is_reproducible() -> None:
    human = Human()
    wrist_rel = human.wrist_from_q(human.q)
    first = ArmMPC(
        human,
        horizon=2,
        n_mpc_samples=8,
        seed=17,
        cartesian=CartesianConfig(goals=[wrist_rel]),
    )
    second = ArmMPC(
        human,
        horizon=2,
        n_mpc_samples=8,
        seed=17,
        cartesian=CartesianConfig(goals=[wrist_rel]),
    )

    _, first_plan = first.solve(human.q)
    _, second_plan = second.solve(human.q)

    np.testing.assert_array_equal(first_plan, second_plan)


def test_mpc_step_returns_env_achieved_state() -> None:
    class FixedResultEnv(ExecutionEnv):
        """Env stub that records commands and returns a fixed achieved state."""

        def __init__(self) -> None:
            super().__init__()
            self.commands: list[np.ndarray] = []

        def execute(self, q_cmd: np.ndarray) -> np.ndarray:
            self.commands.append(q_cmd)
            return np.full(Q_DIM, 0.5)

        def visualize(self, path: Path | None = None) -> np.ndarray:
            raise NotImplementedError

        def save_video(self, path: str | Path, fps: int = 20) -> None:
            raise NotImplementedError

    env = FixedResultEnv()
    human = Human()
    planner = ArmMPC(
        human,
        horizon=2,
        n_mpc_samples=8,
        seed=17,
        env=env,
        cartesian=CartesianConfig(goals=[human.wrist_from_q(human.q)]),
    )

    achieved = planner.step()

    assert len(env.commands) == 1
    np.testing.assert_array_equal(achieved.q, np.full(Q_DIM, 0.5))


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


def test_load_mpc_config_parses_arm_override(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml(
            "arm:\n"
            "  - [0.1, 0.2, 0.3]\n"
            "  - [0.4, 0.5, 0.6]\n"
            "  - [0.7, 0.8, 0.9]\n"
        ),
    )

    cfg = load_mpc_config(path)

    assert cfg.arm == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


def test_load_mpc_config_rejects_wrong_shape_arm(tmp_path) -> None:
    path = _write_config(
        tmp_path, _base_yaml("arm:\n  - [0.1, 0.2, 0.3]\n  - [0.4, 0.5, 0.6]\n")
    )

    with pytest.raises(ValueError, match="arm must be a 3x3"):
        load_mpc_config(path)


def test_load_mpc_config_env_defaults_to_kinematic(tmp_path) -> None:
    path = _write_config(tmp_path, _base_yaml())

    cfg = load_mpc_config(path)

    assert cfg.env == "kinematic"


def test_load_mpc_config_rejects_unknown_env(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
env: holodeck
"""),
    )

    with pytest.raises(ValueError, match="env must be one of"):
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

    with pytest.raises(ValueError, match="min must be less than max"):
        build_extra_costs(cfg.costs, Human())


def test_load_mpc_config_rejects_bad_cartesian_goal(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
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

    assert cfg.cartesian is not None and cfg.feedback is None
    assert cfg.pose == Path("src/uncertain_feedback/motion_generators/mdm/demo_pose.pt")
    assert cfg.cartesian.goals == [[0.1, 0.2, 0.3]]


def test_load_mpc_config_rejects_retired_planner_key(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
planner: arm_mpc
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
""",
    )

    with pytest.raises(ValueError, match="'planner' is retired"):
        load_mpc_config(path)


def test_load_mpc_config_rejects_retired_top_level_uq(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
uq:
  diffusion_samples: 4
"""),
    )

    with pytest.raises(ValueError, match="'uq' is retired"):
        load_mpc_config(path)


def test_load_mpc_config_rejects_constraints_with_robot_actions(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
constraints:
  robot_ik:
    max_residual: 0.001
robot_actions:
  max_joint_delta: 0.005
"""),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_mpc_config(path)


def test_load_mpc_config_rejects_empty_cartesian_goals(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
cartesian:
  threshold: 0.05
"""),
    )

    with pytest.raises(ValueError, match="cartesian.goals must be non-empty"):
        load_mpc_config(path)


def test_correction_threshold_defaults_to_legacy_transfer_value(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
transfer:
  trigger_threshold: 0.07
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.corrections.trigger_threshold == 0.07
    assert cfg.transfer.trigger_threshold == 0.07


def test_correction_threshold_overrides_legacy_transfer_value(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
corrections:
  trigger_threshold: 0.03
transfer:
  trigger_threshold: 0.07
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.corrections.trigger_threshold == 0.03
    assert cfg.transfer.trigger_threshold == 0.03


def test_correction_threshold_rejects_negative_value(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
corrections:
  trigger_threshold: -0.01
"""),
    )

    with pytest.raises(ValueError, match="must be nonnegative"):
        load_mpc_config(path)


def test_elbow_height_cost_zero_inside_range() -> None:
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    human = Human()
    elbow_height = human.fk.fk(np.zeros((3, 3)))[3, 1] - human.spine3_pos[1]

    cost = ElbowHeightCost(
        min_height=elbow_height - 0.01,
        max_height=elbow_height + 0.01,
        weight=100.0,
        progress_weight=100.0,
        human=human,
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_elbow_height_cost_penalizes_outside_range() -> None:
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    human = Human()
    elbow_height = human.fk.fk(np.zeros((3, 3)))[3, 1] - human.spine3_pos[1]

    cost = ElbowHeightCost(
        min_height=elbow_height + 0.1,
        max_height=elbow_height + 0.2,
        weight=100.0,
        progress_weight=100.0,
        human=human,
    )

    assert cost(q_trajs)[0] > 0.9


def test_elbow_flexion_angle_cost_zero_inside_range() -> None:
    human = Human()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    flexion = compute_elbow_flexion_angles(q_trajs[:, 0], human)[0]
    cost = ElbowFlexionAngleCost(
        min_angle=flexion - 0.01,
        max_angle=flexion + 0.01,
        weight=100.0,
        progress_weight=100.0,
        human=human,
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_elbow_flexion_angle_cost_penalizes_outside_range() -> None:
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    cost = ElbowFlexionAngleCost(
        min_angle=0.4,
        max_angle=0.5,
        weight=100.0,
        progress_weight=100.0,
        human=Human(),
    )

    assert cost(q_trajs)[0] > 1.0


def test_shoulder_abduction_angle_cost_zero_inside_range() -> None:
    human = Human()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    abduction = compute_shoulder_abduction_angles(q_trajs[:, 0], human)[0]
    cost = ShoulderAbductionAngleCost(
        min_angle=abduction - 0.01,
        max_angle=abduction + 0.01,
        weight=100.0,
        progress_weight=100.0,
        human=human,
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_shoulder_abduction_angle_cost_penalizes_outside_range() -> None:
    human = Human()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    abduction = compute_shoulder_abduction_angles(q_trajs[:, 0], human)[0]
    cost = ShoulderAbductionAngleCost(
        min_angle=abduction + 0.1,
        max_angle=abduction + 0.2,
        weight=100.0,
        progress_weight=100.0,
        human=human,
    )

    assert cost(q_trajs)[0] > 0.9


def test_compute_elbow_heights_uses_joint_before_wrist() -> None:
    human = Human()
    trajectory = np.zeros((1, 3, 3), dtype=np.float64)
    trajectory[0, 0, 2] = 1.0
    positions = human.fk.fk_batch(
        trajectory,
        human.spine3_pos,
        human.spine3_aa,
    )

    learned_height = compute_elbow_heights(trajectory, human)[0]
    joint_before_wrist_height = positions[0, -2, 1] - human.spine3_pos[1]
    wrist_height = positions[0, -1, 1] - human.spine3_pos[1]

    np.testing.assert_allclose(learned_height, joint_before_wrist_height)
    assert not np.isclose(learned_height, wrist_height)


def test_compute_elbow_flexion_angles_measures_arm_bend() -> None:
    human = Human()
    neutral = np.zeros((1, 3, 3), dtype=np.float64)
    bent = np.zeros((1, 3, 3), dtype=np.float64)
    bent[0, 2, 1] = 1.5  # wrist slot bends the forearm relative to the upper arm
    reoriented = np.zeros((1, 3, 3), dtype=np.float64)
    reoriented[0, 0, 2] = 2.0  # shoulder slot: moves the whole arm, no bend
    reoriented[0, 1, 0] = 0.3  # elbow slot: reorients the upper arm, no bend

    neutral_angle = compute_elbow_flexion_angles(neutral, human)[0]
    bent_angle = compute_elbow_flexion_angles(bent, human)[0]
    reoriented_angle = compute_elbow_flexion_angles(reoriented, human)[0]

    assert bent_angle > neutral_angle + 1.0
    np.testing.assert_allclose(reoriented_angle, neutral_angle)


def test_compute_shoulder_abduction_angles_changes_with_upper_arm_direction() -> None:
    human = Human()
    neutral = np.zeros((1, 3, 3), dtype=np.float64)
    abducted = np.zeros((1, 3, 3), dtype=np.float64)
    abducted[0, 0, 2] = 0.7

    neutral_angle = compute_shoulder_abduction_angles(neutral, human)[0]
    abducted_angle = compute_shoulder_abduction_angles(abducted, human)[0]

    assert not np.isclose(neutral_angle, abducted_angle)


def test_update_elbow_cost_low_mpc_updates_only_min_to_mdm_5th() -> None:
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=100.0,
        weight=1.0,
        progress_weight=1.0,
        human=Human(),
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
        human=Human(),
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
        human=Human(),
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
        human=Human(),
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
        human=Human(),
    )
    mdm_values = np.linspace(50.0, 150.0, 21)
    mpc_values = np.linspace(0.0, 20.0, 21)

    updated = update_preference_cost(cost, mdm_values, mpc_values, alpha=0.0)  # type: ignore[arg-type]

    np.testing.assert_allclose(updated.min_value, 55.0)
    np.testing.assert_allclose(updated.max_value, cost.max_value)


def test_update_preference_cost_high_mpc_updates_only_max_to_mdm_95th() -> None:
    cost = ShoulderAbductionAngleCost(
        min_angle=0.0,
        max_angle=100.0,
        weight=1.0,
        progress_weight=1.0,
        human=Human(),
    )
    mdm_values = np.linspace(-50.0, 50.0, 21)
    mpc_values = np.linspace(100.0, 120.0, 21)

    updated = update_preference_cost(cost, mdm_values, mpc_values, alpha=0.0)  # type: ignore[arg-type]

    np.testing.assert_allclose(updated.min_value, cost.min_value)
    np.testing.assert_allclose(updated.max_value, 45.0)


def test_elbow_height_cost_scores_entire_rollout_not_only_terminal() -> None:
    human = Human()
    inside = np.zeros((3, 3), dtype=np.float64)
    high = np.zeros((3, 3), dtype=np.float64)
    high[0, 2] = 1.0
    elbow_height = human.fk.fk(inside)[3, 1] - human.spine3_pos[1]
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
        human=human,
    )

    costs = cost(q_trajs)

    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1], 0.0)


def test_elbow_height_progress_penalty_only_penalizes_getting_worse_outside() -> None:
    human = Human()
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
        human=human,
    )

    costs = cost(q_trajs)

    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1], 0.0)


def test_default_preference_output_path_uses_learned_suffix(tmp_path) -> None:
    config_path = tmp_path / "mdm.yaml"

    output_path = planner_run._default_preference_output_path(config_path)

    assert output_path == tmp_path / "mdm_learned.yaml"


def test_save_learned_preference_yaml_updates_multiple_costs(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        """
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
    human = Human()
    learned_height = ElbowHeightCost(
        min_height=0.2,
        max_height=0.6,
        weight=12.0,
        progress_weight=5.0,
        human=human,
    )
    learned_flexion = ElbowFlexionAngleCost(
        min_angle=0.5,
        max_angle=1.5,
        weight=50.0,
        progress_weight=50.0,
        human=human,
    )
    learned_abduction = ShoulderAbductionAngleCost(
        min_angle=0.2,
        max_angle=1.0,
        weight=60.0,
        progress_weight=20.0,
        human=human,
    )

    planner_run._save_learned_preference_yaml(
        config_path,
        output_path,
        [learned_height, learned_flexion, learned_abduction],  # type: ignore[list-item]
    )

    with open(output_path, encoding="utf-8") as f:
        saved = yaml.safe_load(f)
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


def test_cartesian_mpc_adds_extra_costs() -> None:
    human = Human()
    q_trajs = np.zeros((2, 2, Q_DIM), dtype=np.float64)
    extra_costs = CompositeTrajectoryCost([_FixedCost([4.0, 5.0])])
    mpc = ArmMPC(
        human,
        extra_costs=extra_costs,
        cartesian=CartesianConfig(goals=[human.wrist_from_q(human.q)]),
    )

    np.testing.assert_allclose(_stage_costs(mpc, q_trajs), [4.0, 5.0])


def test_cartesian_goal_is_not_relative_to_mdm_endpoint() -> None:
    fk = SmplLeftArmFK()
    spine3_pos = np.array([0.25, 1.0, -0.3], dtype=np.float64)
    spine3_aa = np.zeros(3, dtype=np.float64)
    cartesian_goal = np.array([0.3, 0.5, 0.1], dtype=np.float64)
    q_trajs = np.zeros((1, 2, Q_DIM), dtype=np.float64)
    human = Human().measured(
        fk, spine3_pos, spine3_aa, fk.tpose_all_joints, np.zeros(Q_DIM)
    )
    mpc = ArmMPC(
        human,
        cartesian=CartesianConfig(goals=[cartesian_goal]),
        feedback=FeedbackConfig(),
    )

    marker_before = _goal_marker(mpc)
    mdm_endpoint = np.zeros((3, 3), dtype=np.float64)
    mdm_endpoint[0, 1] = 1.0
    mpc.set_mdm_goal(fk.arm_aa_to_q(mdm_endpoint, spine3_aa))
    mpc.push_trajectory(np.stack([np.zeros((3, 3)), mdm_endpoint]))

    np.testing.assert_allclose(_goal_marker(mpc), marker_before)
    wrist_rel = fk.fk(np.zeros((3, 3)), spine3_pos, spine3_aa)[-1] - spine3_pos
    expected_cost = ((wrist_rel - cartesian_goal) ** 2).sum()
    np.testing.assert_allclose(_stage_costs(mpc, q_trajs), [expected_cost])


def test_cartesian_mpc_consumes_final_mdm_goal_then_uses_cartesian_mode() -> None:
    human = Human()
    arm0 = np.zeros((3, 3), dtype=np.float64)
    mpc = ArmMPC(
        human,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        cartesian=CartesianConfig(goals=[human.wrist_from_q(human.q)]),
        feedback=FeedbackConfig(),
    )
    mpc.push_trajectory(np.stack([arm0]))

    assert not mpc.mdm_tracking_complete
    q1 = mpc.step().q

    assert not _playback(mpc).in_playback()

    called = {"cartesian": False}
    real_solve_sampling = mpc._solve_sampling

    def spy_solve_sampling(current_q, stage_cost, actions):
        called["cartesian"] = True
        np.testing.assert_allclose(current_q, q1)
        return real_solve_sampling(current_q, stage_cost, actions)

    mpc._solve_sampling = spy_solve_sampling  # type: ignore[method-assign]
    mpc.step()

    assert called["cartesian"]


def test_cartesian_mpc_tracking_complete_only_after_playback_exhausts() -> None:
    human = Human()
    arm0 = np.zeros((3, 3), dtype=np.float64)
    q0 = human.q
    mpc = ArmMPC(
        human,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        cartesian=CartesianConfig(goals=[human.wrist_from_q(q0)]),
        # large cap: each frame reached in one step
        feedback=FeedbackConfig(max_playback_delta=10.0),
    )
    far_goal = np.full((3, 3), 0.5, dtype=np.float64)
    mpc.push_trajectory(np.stack([arm0, far_goal]))

    assert not mpc.mdm_tracking_complete
    q1 = mpc.step().q
    np.testing.assert_allclose(q1, q0)

    # One frame followed, one remaining: still in playback.
    assert not mpc.mdm_tracking_complete
    assert _playback(mpc)._idx == 1

    q2 = mpc.step().q
    np.testing.assert_allclose(q2, human.q_from_arm_aa(far_goal))

    # Trajectory exhausted: Cartesian mode now engages.
    assert not _playback(mpc).in_playback()
    assert _playback(mpc)._idx == 2


def test_cartesian_mpc_visualizer_hides_joint_target_and_sets_cartesian_target(
    monkeypatch,
) -> None:
    from uncertain_feedback.utils import plot as plot_module

    class SpyArmVisualizer:
        """Visualizer spy recording every live-view call it receives."""

        TARGET_COLOR = "royalblue"
        MDM_COLOR = "darkorange"
        instances: list["SpyArmVisualizer"] = []

        def __init__(self, fk):
            self.fk = fk
            self.open_live_kwargs = None
            self.cartesian_targets = []
            self.step_colors = []
            self.open_live_args = ()
            self.open_live_kwargs = {}
            self.mdm_goal = None
            self.preview_q = None
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

        def update_goal_region(self, marker_world, outlines_world):
            del outlines_world
            self.cartesian_targets.append(np.asarray(marker_world, dtype=np.float64))

        def update_step(self, q, dist, color=TARGET_COLOR):
            del q, dist
            self.step_colors.append(color)

    monkeypatch.setattr(plot_module, "ArmVisualizer", SpyArmVisualizer)

    human = Human()
    cartesian_goal = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    mpc = ArmMPC(
        human,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        visualize=True,
        cartesian=CartesianConfig(goals=[cartesian_goal]),
        feedback=FeedbackConfig(),
    )
    mpc.push_trajectory(np.stack([np.full((3, 3), 0.5, dtype=np.float64)]))

    mpc.step()

    spy = SpyArmVisualizer.instances[0]
    assert spy.open_live_kwargs["show_target_arm"] is False
    np.testing.assert_allclose(
        spy.cartesian_targets[0],
        human.spine3_pos + cartesian_goal,
    )
    assert spy.step_colors == [SpyArmVisualizer.MDM_COLOR]


def test_mdm_push_trajectory_stores_full_trajectory_for_playback() -> None:
    frames = np.arange(23 * 3 * 3, dtype=np.float64).reshape(23, 3, 3)
    mpc = ArmMPC(Human(), feedback=FeedbackConfig())

    mpc.push_trajectory(frames)

    # The full-resolution trajectory is stored for direct playback.
    expected = mpc.human.q_from_arm_aa(frames)
    frames_now = _playback(mpc)._frames
    assert frames_now is not None
    np.testing.assert_allclose(frames_now, expected)
    assert _playback(mpc)._idx == 0
    assert not mpc.mdm_tracking_complete
    preview_now = _playback(mpc).preview_q
    assert preview_now is not None
    np.testing.assert_allclose(preview_now, expected[22])


def test_mdm_push_trajectory_accepts_canonical_arm_q() -> None:
    human = Human()
    frames = human.q_from_arm_aa(np.zeros((3, 3, 3), dtype=np.float64))
    mpc = ArmMPC(human, feedback=FeedbackConfig())

    mpc.push_trajectory(frames)

    frames_pushed = _playback(mpc)._frames
    assert frames_pushed is not None
    np.testing.assert_allclose(frames_pushed, frames)


def test_mdm_push_trajectory_rejects_collar_row() -> None:
    mpc = ArmMPC(Human(), feedback=FeedbackConfig())
    frames = np.zeros((2, 4, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="arm_aa must end in shape"):
        mpc.push_trajectory(frames)


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
    mpc = ArmMPC(
        Human(),
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        feedback=FeedbackConfig(max_playback_delta=1.0),
    )
    mpc.push_trajectory(frames)

    for expected in mpc.human.q_from_arm_aa(frames):
        assert not mpc.mdm_tracking_complete
        np.testing.assert_allclose(mpc.step().q, expected, atol=1e-9)

    # Playback exhausted: the planner holds (no goal space configured).
    assert mpc.mdm_tracking_complete


def _max_joint_rotation(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Largest clavicle, shoulder, or elbow angular change."""
    rel = (
        Rotation.from_rotvec(q_b[:6].reshape(2, 3))
        * Rotation.from_rotvec(q_a[:6].reshape(2, 3)).inv()
    ).as_rotvec()
    return max(float(np.linalg.norm(rel, axis=1).max()), abs(q_b[6] - q_a[6]))


def test_mdm_playback_caps_large_jump_velocity() -> None:
    # A single far frame: a 1.2 rad shoulder jump must be traversed over many
    # capped steps, never exceeding max_playback_delta per joint per step.
    max_delta = 0.1
    frames = np.array(
        [[[1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    mpc = ArmMPC(
        Human(),
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        feedback=FeedbackConfig(max_playback_delta=max_delta),
    )
    mpc.push_trajectory(frames)

    q = mpc.human.q
    n_steps = 0
    while not mpc.mdm_tracking_complete and n_steps < 100:
        prev = q
        q = mpc.step().q
        assert _max_joint_rotation(prev, q) <= max_delta + 1e-9
        n_steps += 1

    assert n_steps > 1  # not snapped in a single step
    assert mpc.mdm_tracking_complete
    np.testing.assert_allclose(q, mpc.human.q_from_arm_aa(frames[0]), atol=1e-9)


def test_mdm_playback_eases_in_from_live_pose() -> None:
    # The arm's live pose differs from frames[0]; the first step must ease in
    # (move at most max_playback_delta), not snap straight to frames[0].
    max_delta = 0.1
    human = Human()
    frames = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    mpc = ArmMPC(
        human,
        horizon=1,
        n_mpc_samples=1,
        max_angle_delta=0.0,
        feedback=FeedbackConfig(max_playback_delta=max_delta),
    )
    mpc.push_trajectory(frames)

    q1 = mpc.step().q
    assert _max_joint_rotation(human.q, q1) <= max_delta + 1e-9
    assert not np.allclose(q1, human.q_from_arm_aa(frames[0]))


def test_mdm_mpc_resumes_toward_cartesian_goal_after_playback() -> None:
    human = Human()
    frames = np.zeros((2, 3, 3), dtype=np.float64)  # trivial trajectory at origin
    goal = human.wrist_from_q(human.q) + np.array([0.0, 0.08, -0.05])
    mpc = ArmMPC(
        human,
        horizon=5,
        n_mpc_samples=128,
        max_angle_delta=0.02,
        seed=0,
        cartesian=CartesianConfig(goals=[goal]),
        feedback=FeedbackConfig(),
    )
    mpc.push_trajectory(frames)

    def wrist_dist(q: np.ndarray) -> float:
        return float(np.linalg.norm(human.wrist_from_q(q) - goal))

    for _ in range(len(frames)):  # phase 1: direct playback
        mpc.step()
    assert mpc.mdm_tracking_complete
    dist_after_playback = wrist_dist(mpc.human.q)

    for _ in range(100):  # phase 2: the goal-space phase resumes sampling
        if mpc.goal_reached(mpc.step().q):
            break

    assert wrist_dist(mpc.human.q) < 0.05 < dist_after_playback


def test_mdm_validate_trajectory_warns_on_range_violation() -> None:
    human = Human()
    # Constrain elbow flexion to a tight range around the neutral bend; a large
    # forearm bend (wrist slot) violates it.
    neutral = compute_elbow_flexion_angles(np.zeros((1, 3, 3)), human)[0]
    extra_costs = CompositeTrajectoryCost(
        [
            ElbowFlexionAngleCost(
                min_angle=neutral - 0.05,
                max_angle=neutral + 0.05,
                weight=1.0,
                progress_weight=1.0,
                human=human,
            )
        ]
    )
    mpc = ArmMPC(human, extra_costs=extra_costs, feedback=FeedbackConfig())

    safe = np.zeros((4, 3, 3), dtype=np.float64)
    assert (  # pylint: disable=use-implicit-booleaness-not-comparison
        mpc.validate_trajectory(human.q_from_arm_aa(safe)) == []
    )

    violating = np.zeros((4, 3, 3), dtype=np.float64)
    violating[2, 2, 1] = 1.5  # large forearm bend on frame 2
    warnings = mpc.validate_trajectory(human.q_from_arm_aa(violating))
    assert len(warnings) == 1
    assert "elbow_flexion_angle" in warnings[0]
    assert "frame 2" in warnings[0]


def test_uq_position_path_converts_selected_mean_with_fixed_mpc_base() -> None:
    """Selected UQ position means are projected into the fixed MPC spine base."""
    fk = SmplLeftArmFK(collar_aa=np.array([0.3, 0.1, -0.1], dtype=np.float64))
    spine3_aa = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    human = Human().measured(
        fk, fk.tpose_spine3_pos, spine3_aa, fk.tpose_all_joints, np.zeros(Q_DIM)
    )
    trajectory = np.zeros((3, Q_DIM), dtype=np.float64)
    trajectory[:, 3:6] = [[0.3, 0.1, -0.2], [0.5, 0.0, 0.1], [0.7, -0.1, 0.3]]
    trajectory[:, Q_ELBOW] = [0.4, 0.6, 0.8]
    gen = _FakePositionGenerator(_arm_positions(human, np.stack([trajectory] * 2)))
    mpc = ArmMPC(
        human,
        feedback=FeedbackConfig(
            anchor_correction=False, uq=UqConfig(diffusion_samples=2)
        ),
        clusterer=_FakePositionClusterer(n_clusters=1),
    )

    mpc.query_mdm_with_uncertainty(
        cast(Any, gen), "raise my left arm up", prefix=False, auto_cluster=0
    )

    uq_frames = _playback(mpc)._frames
    assert uq_frames is not None
    np.testing.assert_allclose(
        human.fk_positions_from_q(uq_frames),
        human.fk_positions_from_q(trajectory),
        atol=1e-9,
    )


def test_uq_result_contains_all_cluster_medoid_trajectories() -> None:
    human = Human()
    trajectories = _elbow_trajectories([0.0, 0.2, 1.0, 1.2])
    gen = _FakePositionGenerator(_arm_positions(human, trajectories))
    mpc = ArmMPC(
        human,
        feedback=FeedbackConfig(
            anchor_correction=False, uq=UqConfig(diffusion_samples=4)
        ),
        clusterer=_TwoTrajectoryClusterer(n_clusters=2),
    )

    chosen = mpc.query_mdm_with_uncertainty(
        cast(Any, gen), "move differently", prefix=False, auto_cluster=1
    )

    result = mpc.last_uq_result
    assert result is not None
    assert result.chosen_label == 1
    assert sorted(result.cluster_means) == [0, 1]
    np.testing.assert_allclose(
        result.cluster_means[0], human.arm_aa_from_q(trajectories[0]), atol=1e-9
    )
    np.testing.assert_allclose(
        result.cluster_means[1], human.arm_aa_from_q(trajectories[2]), atol=1e-9
    )
    np.testing.assert_allclose(chosen, human.q_from_arm_aa(result.chosen_mean))


def test_uq_axis_angle_picker_uses_refined_subset_medoid(monkeypatch) -> None:
    human = Human()
    trajectories = _elbow_trajectories([0.0, 0.2, 1.0, 1.2])
    gen = _FakePositionGenerator(_arm_positions(human, trajectories))
    mpc = ArmMPC(
        human,
        feedback=FeedbackConfig(
            anchor_correction=False, uq=UqConfig(diffusion_samples=4)
        ),
        clusterer=_TwoTrajectoryClusterer(n_clusters=2),
    )
    monkeypatch.setattr(
        "uncertain_feedback.uncertainty.uq_selector.pick_cluster",
        lambda *_args, **_kwargs: ClusterPickResult(
            root_label=0,
            sample_indices=np.array([1], dtype=np.intp),
            scale=1.0,
        ),
    )

    chosen = mpc.query_mdm_with_uncertainty(
        cast(Any, gen), "move differently", prefix=False
    )

    result = mpc.last_uq_result
    assert result is not None
    assert result.chosen_label == 0
    np.testing.assert_allclose(chosen, trajectories[1], atol=1e-9)
    np.testing.assert_allclose(result.cluster_means[0], human.arm_aa_from_q(chosen))
    np.testing.assert_allclose(
        result.cluster_means[1], human.arm_aa_from_q(trajectories[2]), atol=1e-9
    )


def test_uq_position_picker_uses_refined_subset_medoid(monkeypatch) -> None:
    class TwoPositionClusterer(TrajectoryClusterer):
        """Clusterer splitting samples into two fixed position groups."""

        def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
            raise AssertionError("position path does not use axis-angle features")

        def _positions_to_features(self, positions: np.ndarray) -> np.ndarray:
            return positions.reshape(positions.shape[0], -1)

        def _fit_predict(self, features: np.ndarray) -> np.ndarray:
            assert features.shape[0] == 4
            return np.array([0, 0, 1, 1], dtype=np.intp)

    human = Human()
    trajectories = _elbow_trajectories([0.0, 0.2, 1.0, 1.2])
    gen = _FakePositionGenerator(_arm_positions(human, trajectories))
    mpc = ArmMPC(
        human,
        feedback=FeedbackConfig(
            anchor_correction=False, uq=UqConfig(diffusion_samples=4)
        ),
        clusterer=TwoPositionClusterer(n_clusters=2),
    )
    monkeypatch.setattr(
        "uncertain_feedback.uncertainty.uq_selector.pick_cluster_positions",
        lambda *_args, **_kwargs: ClusterPickResult(
            root_label=0,
            sample_indices=np.array([1], dtype=np.intp),
            scale=1.0,
        ),
    )

    chosen = mpc.query_mdm_with_uncertainty(
        cast(Any, gen), "move differently", prefix=False
    )

    result = mpc.last_uq_result
    assert result is not None
    np.testing.assert_allclose(chosen, trajectories[1], atol=1e-9)
    np.testing.assert_allclose(result.cluster_means[0], human.arm_aa_from_q(chosen))
    np.testing.assert_allclose(
        result.cluster_means[1], human.arm_aa_from_q(trajectories[2]), atol=1e-9
    )


def test_hidden_joint_limits_accept_canonical_arm_q() -> None:
    human = Human()
    arm_aa = np.zeros((2, 3, 3), dtype=np.float64)
    arm_aa[1, 1, 0] = 0.4
    q = human.q_from_arm_aa(arm_aa)
    user = _joint_limit_user()

    expected = compute_violations(user, human, arm_aa)

    np.testing.assert_allclose(compute_violations(user, human, q), expected)
    rollouts = np.stack([q, q])
    costs = HiddenCostTerm(user=user, human=human)(rollouts)
    assert costs.shape == (2,)
    assert np.all(np.isfinite(costs))


def test_no_mdm_cartesian_mpc_adds_extra_costs() -> None:
    human = Human()
    q_trajs = np.zeros((2, 2, Q_DIM), dtype=np.float64)
    extra_costs = CompositeTrajectoryCost([_FixedCost([6.0, 7.0])])
    mpc = ArmMPC(
        human,
        extra_costs=extra_costs,
        cartesian=CartesianConfig(goals=[human.wrist_from_q(human.q)]),
    )

    np.testing.assert_allclose(_stage_costs(mpc, q_trajs), [6.0, 7.0])


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
    run_dir = artifact_run_dir(
        tmp_path,
        Path("llm_cost_artifacts"),
    )

    assert run_dir.parent == tmp_path / "llm_cost_artifacts"


def test_generated_python_cost_executes_with_fk_context() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
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
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
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
    assert context.shoulder_internal_external_rotation_angles(q_trajs[:, 1:]).shape == (
        2,
        3,
    )


def test_generated_cost_context_shoulder_twist_matches_tpose_axis_rotation() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        window=5,
    )
    axis = context.fk.tpose_joints[3] - context.fk.tpose_joints[2]
    axis = axis / np.linalg.norm(axis)
    trajectory = np.zeros((2, Q_DIM), dtype=np.float64)
    trajectory[:, 3:6] = axis * 0.4
    trajectory[1, :3] = axis * -0.7

    twist = context.shoulder_internal_external_rotation_angles(trajectory)

    np.testing.assert_allclose(twist, [0.4, 0.4], atol=1e-10)


def test_generated_python_cost_uses_canonical_shoulder_twist_feature() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((2, Q_DIM)),
        window=5,
    )
    axis = context.fk.tpose_joints[3] - context.fk.tpose_joints[2]
    axis = axis / np.linalg.norm(axis)
    q_trajs = np.zeros((2, 2, Q_DIM), dtype=np.float64)
    q_trajs[:, 1, 3:6] = axis * 0.4
    q_trajs[1, 1, :3] = axis * -0.7
    generated = GeneratedPythonCost(
        code="""def cost(q_trajs, context, params):
    twist = context.shoulder_internal_external_rotation_angles(q_trajs[:, 1:])
    return np.mean(twist, axis=1)
""",
        params={},
        context=context,
    )

    np.testing.assert_allclose(generated(q_trajs), [0.4, 0.4], atol=1e-10)


def test_generated_cost_context_shoulder_component_angles_are_stable() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        window=5,
    )
    neutral = np.zeros((1, Q_DIM), dtype=np.float64)
    axis = context.fk.tpose_joints[3] - context.fk.tpose_joints[2]
    axis = axis / np.linalg.norm(axis)
    twisted = neutral.copy()
    twisted[0, 3:6] = axis * 0.4
    adducted = neutral.copy()
    adducted[0, 5] = 0.4

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
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        window=5,
    )

    summaries = build_motion_summaries(context)

    assert "joint_features" in summaries["current"]
    assert "joint_features" in summaries["mdm_traj"]
    assert "shoulder_flexion_extension" in summaries["mdm_traj"]["joint_features"]
    assert "shoulder_abduction_adduction" in summaries["mdm_traj"]["joint_features"]
    assert (
        "shoulder_internal_external_rotation" in summaries["mdm_traj"]["joint_features"]
    )


def test_motion_summaries_include_reference_and_goal_when_present() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        window=5,
        reference_traj=np.zeros((4, 3, 3), dtype=np.float64),
    )

    summaries = build_motion_summaries(
        context, cartesian_goal=np.array([0.1, 0.2, 0.3])
    )

    assert "reference" in summaries
    assert "joint_features" in summaries["reference"]
    assert summaries["cartesian_goal"] == [0.1, 0.2, 0.3]


def test_motion_summaries_compare_chosen_to_named_rejected_rollouts() -> None:
    chosen = np.zeros((3, 3, 3), dtype=np.float64)
    original = chosen.copy()
    original[-1, 0, 2] = -0.3
    rejected_0 = chosen.copy()
    rejected_0[-1, 0, 2] = 0.2
    rejected_1 = chosen.copy()
    rejected_1[-1, 0, 2] = 0.5
    context = build_generated_cost_context(
        Human(),
        mdm_traj=chosen,
        window=5,
        reference_traj=original,
        rejected_trajs=(rejected_0, rejected_1),
    )

    summaries = build_motion_summaries(context)

    comparison = summaries["candidate_comparison"]
    abduction = comparison["shoulder_abduction_adduction"]
    rejected_ends = np.array(
        [
            context.shoulder_abduction_adduction_angles(rejected_0)[-1],
            context.shoulder_abduction_adduction_angles(rejected_1)[-1],
        ]
    )
    assert abduction["chosen_rollout"] == "mdm_traj"
    assert abduction["current_rollout"] == "current"
    assert abduction["current_value"] == pytest.approx(
        context.shoulder_abduction_adduction_angles(context.current_q)
    )
    assert abduction["chosen_minus_current"] == pytest.approx(
        abduction["chosen_end"] - abduction["current_value"]
    )
    assert abduction["original_plan_rollout"] == "reference"
    assert abduction["original_plan_end"] == pytest.approx(
        context.shoulder_abduction_adduction_angles(original)[-1]
    )
    assert abduction["chosen_minus_original_plan"] == pytest.approx(
        abduction["chosen_end"] - abduction["original_plan_end"]
    )
    assert list(abduction["rejected_ends"]) == [
        "rejected_cluster_0",
        "rejected_cluster_1",
    ]
    np.testing.assert_allclose(list(abduction["rejected_ends"].values()), rejected_ends)
    assert abduction["rejected_median"] == pytest.approx(np.median(rejected_ends))
    assert abduction["rejected_std"] == pytest.approx(np.std(rejected_ends))
    assert abduction["standardized_separation"] == pytest.approx(
        (abduction["chosen_end"] - np.median(rejected_ends)) / np.std(rejected_ends)
    )
    assert comparison["elbow_flexion"]["standardized_separation"] is None


def test_motion_summaries_omit_reference_without_reference_traj() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
        window=5,
    )

    summaries = build_motion_summaries(context)

    assert "reference" not in summaries
    assert "cartesian_goal" not in summaries


def test_prompt_images_render_overlay_with_reference(tmp_path) -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        window=5,
        reference_traj=np.zeros((5, 3, 3), dtype=np.float64),
    )

    images = render_prompt_images(
        context,
        tmp_path,
        reference_traj=np.zeros((5, 3, 3), dtype=np.float64),
        goal_pos=np.array([0.1, 0.2, 0.3]),
    )

    # No candidate clusters → no "others" image; reference present → reference image.
    assert set(images) == {"current_cluster_traj_img", "reference_traj_img"}
    for path in images.values():
        assert path.exists() and path.stat().st_size > 0


def test_generated_python_cost_rejects_bad_shape() -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((3, 3, 3), dtype=np.float64),
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
        Human(),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        window=5,
    )

    images = render_prompt_images(context, tmp_path)

    # No candidates and no reference → only the current-cluster image.
    assert set(images) == {"current_cluster_traj_img"}
    assert images["current_cluster_traj_img"].name == "current.png"
    path = images["current_cluster_traj_img"]
    assert path.exists() and path.stat().st_size > 0


def test_prompt_images_render_only_other_clusters_terminal_poses(
    tmp_path, monkeypatch
) -> None:
    context = build_generated_cost_context(
        Human(),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        window=5,
    )
    rejected = np.stack(
        [np.full((3, 3), value, dtype=np.float64) for value in (0.1, 0.2, 0.3, 0.4)]
    )

    candidate_trajs = {
        0: np.zeros((4, 3, 3), dtype=np.float64),
        1: rejected,
    }
    rendered_poses: list[np.ndarray] = []
    original_full_body_positions = context.fk.full_body_positions

    def record_full_body_positions(q, spine3_pos, spine3_aa):
        rendered_poses.append(np.asarray(q, dtype=np.float64).copy())
        return original_full_body_positions(q, spine3_pos, spine3_aa)

    monkeypatch.setattr(context.fk, "full_body_positions", record_full_body_positions)
    images = render_prompt_images(
        context,
        tmp_path,
        candidate_trajs=candidate_trajs,
        highlight_label=0,
    )

    # More than one cluster → the "others" image is rendered; no reference.
    assert set(images) == {"current_cluster_traj_img", "other_clusters_traj_img"}
    assert images["other_clusters_traj_img"].name == "others.png"
    for path in images.values():
        assert path.exists() and path.stat().st_size > 0
    canonical_rejected = context.arm_aa(rejected)
    for intermediate_pose in canonical_rejected[:-1]:
        assert not any(
            np.array_equal(rendered, intermediate_pose) for rendered in rendered_poses
        )
    assert (
        sum(
            np.array_equal(rendered, canonical_rejected[-1])
            for rendered in rendered_poses
        )
        == 3
    )


def test_apply_llm_generated_cost_with_fake_model(tmp_path) -> None:
    mdm_traj = np.zeros((3, 3, 3), dtype=np.float64)
    generated_context = build_generated_cost_context(
        Human(),
        mdm_traj=mdm_traj,
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
    mpc = ArmMPC(Human())
    llm_cfg = LlmCostConfig(
        enabled=True,
        artifact_dir=tmp_path / "artifacts",
    )

    run_dir = artifact_run_dir(tmp_path, llm_cfg.artifact_dir)
    summaries = build_motion_summaries(generated_context)
    images = render_prompt_images(generated_context, run_dir / "images")
    generator = create_cost_generator(
        llm_cfg,
        generated_context,
        "raise the elbow",
        summaries=summaries,
        run_dir=run_dir,
        images=images,
        mpc=mpc,
        llm_model_factory=lambda _model_name: fake_model,
    )
    generated = generator.generate(install=True)

    assert generated is not None
    assert len(mpc._extra_costs.terms()) == 1
    artifact_dirs = list((tmp_path / "artifacts").iterdir())
    assert len(artifact_dirs) == 1
    assert (artifact_dirs[0] / "interpret_prompt.txt").exists()
    assert (artifact_dirs[0] / "ground_prompt.txt").exists()
    assert (artifact_dirs[0] / "author_prompt.txt").exists()
    assert (artifact_dirs[0] / "cost.py").exists()
    assert (artifact_dirs[0] / "recipient_explanation.txt").read_text(
        encoding="utf-8"
    ) == response["recipient_explanation"]
    with open(artifact_dirs[0] / "params.json", encoding="utf-8") as f:
        params = json.load(f)
    assert params["recipient_explanation"] == response["recipient_explanation"]
    assert fake_model.received_images is not None


def test_planning_loop_stops_when_cartesian_goal_reached() -> None:
    human = Human()
    goal = human.wrist_from_q(human.q) + np.array([0.03, 0.06, 0.0])
    planner = ArmMPC(
        human,
        horizon=10,
        n_mpc_samples=256,
        max_angle_delta=0.1,
        visualize=False,
        cartesian=CartesianConfig(goals=[goal], threshold=0.12),
    )

    result = run_planning_loop(planner, 300, stop_on_runtime_error=True)

    # Stops well before the 300-step budget once the wrist is within threshold.
    assert result.reached_goal is True
    assert len(result.human.history) - 1 < 300
    assert planner.goal_reached(result.human.q) is True


def test_planning_loop_waits_for_mdm_correction_before_stopping() -> None:
    human = Human()
    arm0 = np.zeros((3, 3), dtype=np.float64)
    # Cartesian goal == start wrist, so goal_reached(q0) is True from the very
    # first step; the loop must still play the pushed correction out before it
    # may stop.
    planner = ArmMPC(
        human,
        horizon=10,
        n_mpc_samples=64,
        max_angle_delta=0.1,
        visualize=False,
        cartesian=CartesianConfig(goals=[human.wrist_from_q(human.q)], threshold=0.12),
        feedback=FeedbackConfig(max_playback_delta=0.2),
    )
    planner.push_trajectory(np.stack([arm0] * 8))

    assert (
        planner.goal_reached(human.q) is True
    )  # goal trivially satisfied at the start
    assert planner.mdm_ready_to_terminate is False  # but a correction is still queued

    result = run_planning_loop(planner, 100, stop_on_runtime_error=True)

    # The loop ran the 8 playback frames before stopping, not stopping at step 1.
    assert result.reached_goal is True
    assert len(result.human.history) - 1 == 8
    assert planner.mdm_ready_to_terminate is True


def test_planning_loop_runs_cartesian_phase_after_correction_then_stops() -> None:
    # The real active-planner scenario: play a correction that ends AWAY from the
    # goal, drive the cartesian phase to the goal, then stop — all in one loop.
    human = Human()
    goal = human.wrist_from_q(human.q) + np.array([0.03, 0.06, 0.0])
    planner = ArmMPC(
        human,
        horizon=10,
        n_mpc_samples=256,
        max_angle_delta=0.1,
        visualize=False,
        cartesian=CartesianConfig(goals=[goal], threshold=0.12),
        feedback=FeedbackConfig(max_playback_delta=0.3),
    )
    # 4 playback frames whose endpoint wrist is well outside cartesian_threshold.
    playback = np.stack(
        [
            k * np.array([[0.0, 0.0, 0.4], [0, 0, 0], [0, 0, 0]])
            for k in (0.25, 0.5, 0.75, 1.0)
        ]
    )
    end_q = human.q_from_arm_aa(playback[-1])
    assert (
        float(np.linalg.norm(human.wrist_from_q(end_q) - goal)) > 0.12
    )  # playback ends away from goal
    planner.set_mdm_goal(end_q)
    planner.push_trajectory(playback)

    result = run_planning_loop(planner, 300, stop_on_runtime_error=True)

    assert result.reached_goal is True
    # Cartesian phase ran AFTER the 4 playback frames, then stopped before the budget.
    assert len(result.human.history) - 1 > playback.shape[0]
    assert len(result.human.history) - 1 < 300
    assert planner.goal_reached(result.human.q) is True


def test_parse_llm_cost_response_accepts_markdown_json() -> None:
    response = parse_llm_cost_response(
        '```json\n{"description":"d","code":"def cost(q_trajs, context, params):\\n    return np.zeros(q_trajs.shape[0])","params":{},"recipient_explanation":"plain"}\n```'
    )

    assert response.description == "d"
    assert response.recipient_explanation == "plain"
    assert "def cost" in response.code
