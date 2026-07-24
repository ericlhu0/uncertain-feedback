"""Tests for execution environments: registry, grasp FK, and visualization."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs import ENV_BUILDERS, make_env
from uncertain_feedback.envs.grasp import GRASP_FRACTION, grasp_pose_fk
from uncertain_feedback.envs.kinematic import KinematicEnv
from uncertain_feedback.envs.sim_mannequin import SimMannequinEnv, _mannequin_joints
from uncertain_feedback.envs.sim_robot_visual import _SMPL_TO_PB, SimRobotVisualEnv
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    q_to_arm_aa,
)


def test_sim_robot_visual_registered() -> None:
    assert "sim_robot_visual" in ENV_BUILDERS
    assert isinstance(make_env("sim_robot_visual"), SimRobotVisualEnv)


def test_grasp_pose_fk_geometry() -> None:
    fk = SmplLeftArmFK()
    q = np.zeros(Q_DIM)
    positions = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), None, None)
    elbow, wrist = positions[3], positions[4]

    pos, quat = grasp_pose_fk(fk, q, None, None, GRASP_FRACTION)

    np.testing.assert_allclose(pos, elbow + GRASP_FRACTION * (wrist - elbow))
    assert np.isclose(float(np.linalg.norm(quat)), 1.0)
    rot = Rotation.from_quat(quat).as_matrix()
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-10)
    assert np.isclose(float(np.linalg.det(rot)), 1.0)
    forearm = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    np.testing.assert_allclose(np.abs(float(np.dot(rot[:, 0], forearm))), 1.0)
    assert np.isclose(float(np.dot(rot[:, 2], forearm)), 0.0, atol=1e-10)


def test_sim_env_execute_passthrough_and_reaches_grasp() -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_robot_visual")
    assert isinstance(env, SimRobotVisualEnv)
    env.set_pose_context(fk, None, None)

    q = np.zeros(Q_DIM)
    achieved = env.execute(q)
    np.testing.assert_array_equal(achieved, q)

    grasp_pos, _ = grasp_pose_fk(fk, q, None, None, GRASP_FRACTION)
    target = _SMPL_TO_PB @ grasp_pos
    ee_pos = np.array(
        p.getLinkState(
            env._robot,
            env._ee_index,
            computeForwardKinematics=True,
            physicsClientId=env._cid,
        )[4]
    )
    assert float(np.linalg.norm(ee_pos - target)) < 5e-2


def test_sim_env_visualize_and_save_video(tmp_path) -> None:
    env = make_env("sim_robot_visual")
    env.set_pose_context(SmplLeftArmFK(), None, None)
    q = np.zeros(Q_DIM)
    for _ in range(3):
        q = env.execute(q + 0.05)

    frame = env.visualize()
    assert frame.shape == (960, 640, 3)
    assert frame.dtype == np.uint8

    out = tmp_path / "sim.gif"
    env.save_video(out, fps=5)
    assert out.stat().st_size > 0


_BENT_ARM_AA = np.array(
    [
        [-0.2578684731175734, 0.11215073986226905, -0.44187915175365783],
        [0.6868865380518193, -0.18, -0.11006289820501858],
        [0.035651850950680734, -1.23, 0.7515284911778386],
    ]
)


def test_sim_mannequin_registered() -> None:
    assert "sim_mannequin" in ENV_BUILDERS
    assert isinstance(make_env("sim_mannequin"), SimMannequinEnv)


def test_sim_mannequin_env_params() -> None:
    env = make_env(
        "sim_mannequin",
        robot_base_offset=[0.5, -0.2, -0.3],
        robot_max_joint_delta=0.05,
    )
    assert isinstance(env, SimMannequinEnv)
    np.testing.assert_allclose(env._robot_base_offset, [0.5, -0.2, -0.3])
    assert env._robot_max_joint_delta == 0.05


def test_mannequin_joint_mapping_round_trip() -> None:
    from uncertain_feedback.envs.sim_mannequin import _MANNEQUIN_BASE_ROT

    shoulder = np.array([0.1, -0.2, 0.9])
    elbow = shoulder + 0.24 * np.array([0.8, -0.3, -0.5]) / np.linalg.norm(
        [0.8, -0.3, -0.5]
    )
    wrist = elbow + 0.29 * np.array([0.2, -0.9, -0.4]) / np.linalg.norm(
        [0.2, -0.9, -0.4]
    )
    q0, q1, q2, elbow_angle = _mannequin_joints(shoulder, elbow, wrist)
    assert elbow_angle >= 0.0

    shoulder_world = (
        _MANNEQUIN_BASE_ROT @ Rotation.from_euler("XYZ", [-q0, -q1, q2]).as_matrix()
    )
    u = shoulder_world @ np.array([0.0, 0.0, -1.0])
    f = shoulder_world @ np.array([0.0, -np.sin(elbow_angle), -np.cos(elbow_angle)])
    np.testing.assert_allclose(
        u, (elbow - shoulder) / np.linalg.norm(elbow - shoulder), atol=1e-8
    )
    np.testing.assert_allclose(
        f, (wrist - elbow) / np.linalg.norm(wrist - elbow), atol=1e-8
    )


def test_sim_mannequin_execute_tracks_command() -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_mannequin")
    assert isinstance(env, SimMannequinEnv)
    env.set_pose_context(fk, None, None)

    q_cmd = fk.arm_aa_to_q(_BENT_ARM_AA)
    achieved = env.execute(q_cmd)
    assert not np.allclose(achieved, q_cmd)
    assert float(np.linalg.norm(achieved[3:] - q_cmd[3:])) < 0.25

    first_elbow = float(achieved[6])
    for _ in range(5):
        q_cmd = q_cmd.copy()
        q_cmd[6] += 0.02
        achieved = env.execute(q_cmd)
    assert float(achieved[6]) > first_elbow
    assert abs(float(achieved[6]) - float(q_cmd[6])) < 0.15


def test_sim_mannequin_readback_roundtrip_is_stable() -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_mannequin")
    assert isinstance(env, SimMannequinEnv)
    env.set_pose_context(fk, None, None)

    q = env.execute(fk.arm_aa_to_q(_BENT_ARM_AA))
    start = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis))[-1]
    for _ in range(30):
        q = env.execute(q)
    end = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis))[-1]
    assert float(np.linalg.norm(end - start)) < 0.01


def test_sim_mannequin_hold_stable_with_vertical_forearm() -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_mannequin")
    assert isinstance(env, SimMannequinEnv)
    env.set_pose_context(fk, None, None)

    q_cmd = np.zeros(Q_DIM)
    q_cmd[6] = 1.5708
    pos = fk.fk(q_to_arm_aa(q_cmd, fk.elbow_hinge_axis), None, None)
    forearm = (pos[4] - pos[3]) / np.linalg.norm(pos[4] - pos[3])
    rot, _ = Rotation.align_vectors([[0.0, 1.0, 0.0]], [forearm])
    q_cmd[3:6] = rot.as_rotvec()
    pos = fk.fk(q_to_arm_aa(q_cmd, fk.elbow_hinge_axis), None, None)
    forearm = (pos[4] - pos[3]) / np.linalg.norm(pos[4] - pos[3])
    assert forearm[1] > 0.99  # inside grasp_pose_fk's up-reference fallback cone

    q = env.execute(q_cmd)
    q_hold = q.copy()
    start = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis))[-1]
    for _ in range(20):
        q = env.execute(q_hold)
    end = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis))[-1]
    assert float(np.linalg.norm(end - start)) < 0.01


def test_sim_mannequin_visualize_and_save_video(tmp_path) -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_mannequin")
    env.set_pose_context(fk, None, None)
    q = fk.arm_aa_to_q(_BENT_ARM_AA)
    for _ in range(3):
        q = env.execute(q)

    frame = env.visualize()
    assert frame.shape == (960, 640, 3)
    assert frame.dtype == np.uint8

    out = tmp_path / "mannequin.gif"
    env.save_video(out, fps=5)
    assert out.stat().st_size > 0


def test_kinematic_env_visualize_and_save_video(tmp_path) -> None:
    env = make_env("kinematic")
    assert isinstance(env, KinematicEnv)
    env.set_pose_context(SmplLeftArmFK(), None, None)
    q = np.zeros(Q_DIM)
    for _ in range(3):
        q = env.execute(q + 0.05)

    frame = env.visualize()
    assert frame.ndim == 3
    assert frame.dtype == np.uint8

    out = tmp_path / "kin.gif"
    env.save_video(out, fps=5)
    assert out.stat().st_size > 0
