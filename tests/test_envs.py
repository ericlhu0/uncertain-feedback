"""Tests for execution environments: registry, grasp FK, and visualization."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs import ENV_BUILDERS, make_env
from uncertain_feedback.envs.grasp import GRASP_FRACTION, grasp_pose_fk
from uncertain_feedback.envs.kinematic import KinematicEnv
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
