"""Tests for execution environments: registry, grasp FK, and visualization."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs import ENV_BUILDERS, make_env
from uncertain_feedback.envs.grasp import (
    GRASP_FRACTION,
    MeasuredGrasp,
    forearm_frame_fk,
    grasp_pose_fk,
)
from uncertain_feedback.envs.kinematic import KinematicEnv
from uncertain_feedback.envs.sim_mannequin import SimMannequinEnv, _mannequin_joints
from uncertain_feedback.envs.sim_robot_visual import _SMPL_TO_PB, SimRobotVisualEnv
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    q_to_arm_aa,
)
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache


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


def _vertical_forearm_q(fk: SmplLeftArmFK) -> np.ndarray:
    """A configuration whose forearm points straight up (SMPL +y)."""
    q = np.zeros(Q_DIM)
    q[6] = 1.5708
    pos = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), None, None)
    forearm = (pos[4] - pos[3]) / np.linalg.norm(pos[4] - pos[3])
    rot, _ = Rotation.align_vectors([[0.0, 1.0, 0.0]], [forearm])
    q[3:6] = rot.as_rotvec()
    return q


def test_measured_grasp_stays_rigid_on_the_forearm() -> None:
    """A grasp measured once must ride the forearm as the arm moves."""
    fk = SmplLeftArmFK()
    q0 = np.zeros(Q_DIM)
    elbow, wrist = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[3:5]
    # An arbitrary gripper pose near the forearm — not the nominal grasp.
    ee_pos = elbow + 0.3 * (wrist - elbow) + np.array([0.01, 0.04, -0.02])
    ee_rot = Rotation.from_euler("xyz", [0.2, -0.5, 1.1])

    grasp = MeasuredGrasp.measure(*forearm_frame_fk(fk, q0, None, None), ee_pos, ee_rot)

    q1 = np.array([0.1, -0.2, 0.15, 0.6, -0.4, 0.3, -0.9])
    elbow1, wrist1 = fk.fk(q_to_arm_aa(q1, fk.elbow_hinge_axis), None, None)[3:5]
    pos1, rot1 = grasp.gripper_pose(*forearm_frame_fk(fk, q1, None, None))

    # Rigid attachment: distances to both ends of the bone are preserved, and so
    # is the angle between the gripper's approach axis and the bone.
    for before, after in ((elbow, elbow1), (wrist, wrist1)):
        assert np.isclose(
            float(np.linalg.norm(ee_pos - before)),
            float(np.linalg.norm(pos1 - after)),
            atol=1e-9,
        )
    bone = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    bone1 = (wrist1 - elbow1) / np.linalg.norm(wrist1 - elbow1)
    assert np.isclose(
        float(np.dot(ee_rot.apply([0.0, 0.0, 1.0]), bone)),
        float(np.dot(rot1.apply([0.0, 0.0, 1.0]), bone1)),
        atol=1e-9,
    )


def test_remeasured_grasp_reproduces_the_pose_it_was_measured_from() -> None:
    """A freshly measured grasp implies the ee pose it was read from, exactly.

    What lets `RealEnv._drive` command an absolute gripper pose: the relative
    target it used to compute subtracts the pose implied by the measured
    configuration, and after re-measuring that *is* the current ee pose, so the
    two forms are identical. Holds however far FK's forearm sits from the real
    one — the bias goes into the transform instead of into the command.
    """
    fk = SmplLeftArmFK()
    q_meas = np.array([0.1, -0.2, 0.15, 0.6, -0.4, 0.3, -0.9])
    # Deliberately far off the forearm: the FK-vs-real disagreement this stands
    # in for must not leak into the implied pose.
    ee_pos = fk.fk(q_to_arm_aa(q_meas, fk.elbow_hinge_axis), None, None)[3] + np.array(
        [0.07, -0.11, 0.05]
    )
    ee_rot = Rotation.from_euler("xyz", [-0.8, 0.3, 2.0])

    forearm = forearm_frame_fk(fk, q_meas, None, None)
    implied_pos, implied_rot = MeasuredGrasp.measure(
        *forearm, ee_pos, ee_rot
    ).gripper_pose(*forearm)

    np.testing.assert_allclose(implied_pos, ee_pos, atol=1e-12)
    np.testing.assert_allclose(implied_rot.as_quat(), ee_rot.as_quat(), atol=1e-12)


def test_measured_grasp_does_not_flip_through_vertical_forearm() -> None:
    """The FK forearm frame has no up-reference, unlike ``grasp_pose_fk``'s.

    A grasp expressed in a frame built from the forearm direction plus world up
    would swing the gripper half a turn as the forearm crosses vertical.
    """
    fk = SmplLeftArmFK()
    q_vertical = _vertical_forearm_q(fk)
    # Sweep the arm about an axis perpendicular to the vertical forearm, so the
    # forearm passes exactly through vertical mid-sweep.
    base = Rotation.from_rotvec(q_vertical[3:6])
    sweep = []
    for s in np.linspace(-0.2, 0.2, 81):
        q = q_vertical.copy()
        q[3:6] = (Rotation.from_rotvec([s, 0.0, 0.0]) * base).as_rotvec()
        sweep.append(q)

    grasp = MeasuredGrasp.measure(
        *forearm_frame_fk(fk, sweep[0], None, None),
        *_gripper_probe(fk, sweep[0]),
    )
    poses = [grasp.gripper_pose(*forearm_frame_fk(fk, q, None, None)) for q in sweep]
    for (pos_a, rot_a), (pos_b, rot_b) in zip(poses, poses[1:]):
        assert float(np.linalg.norm(pos_b - pos_a)) < 0.01
        assert float((rot_b * rot_a.inv()).magnitude()) < np.deg2rad(2.0)


def _gripper_probe(fk: SmplLeftArmFK, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """A plausible gripper pose: the nominal grasp, held off the forearm axis.

    The offset is what makes a frame flip visible as a position jump, not only
    as a rotation.
    """
    pos, quat = grasp_pose_fk(fk, q, None, None, GRASP_FRACTION)
    rot = Rotation.from_quat(quat)
    return pos - 0.04 * rot.apply([0.0, 0.0, 1.0]), rot


def test_left_arm_faces_cover_what_the_arm_pose_moves() -> None:
    """The goal ghost draws the arm alone, so the mask must track the arm."""
    fk = SmplLeftArmFK()
    cache = SmplMeshCache(fk.tpose_all_joints)
    arm_faces = cache.left_arm_faces
    assert 0 < len(arm_faces) < len(cache.faces)

    q1 = np.array([0.0, 0.0, 0.0, 0.5, -0.6, 0.4, -1.2])
    verts = [
        cache.preview(fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), None, None))
        for q in (np.zeros(Q_DIM), q1)
    ]
    moved = np.linalg.norm(verts[1] - verts[0], axis=1)
    on_arm = np.zeros(len(moved), dtype=bool)
    on_arm[np.unique(arm_faces)] = True

    # The masked vertices are the ones the arm pose carries; the rest shift only
    # by the skinning's reach into the shoulder (measured: 134 mm vs 8 mm at p95).
    assert float(np.median(moved[on_arm])) > 0.05
    assert float(np.percentile(moved[~on_arm], 95)) < 0.02


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


def test_initial_q_defaults_to_nominal() -> None:
    """Envs that cannot measure the person plan from the config's start pose."""
    q_nominal = np.arange(Q_DIM, dtype=np.float64)
    for name in ENV_BUILDERS:
        if name == "real":
            continue  # needs a live mocap stream; measured start covered in test_mocap
        np.testing.assert_array_equal(make_env(name).initial_q(q_nominal), q_nominal)


def test_pose_context_round_trips_for_unmeasured_envs() -> None:
    """Only envs that measure the person revise the anchor the run plans against.

    `run.py` reads this back after `initial_q` for every env, so the default has
    to hand back exactly what the config supplied.
    """
    spine3_pos = np.array([0.1, -0.2, 0.3])
    spine3_aa = np.array([0.0, 0.4, 0.0])
    body_pos = np.zeros((22, 3))
    for name in ENV_BUILDERS:
        if name == "real":
            continue  # measures the person; covered in test_mocap
        env = make_env(name)
        env.set_pose_context(SmplLeftArmFK(), spine3_pos, spine3_aa, body_pos)
        assert env.pose_context() == (spine3_pos, spine3_aa, body_pos)


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


def test_sim_mannequin_joint_limit_padding() -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_mannequin", robot_joint_limit_padding=0.3)
    assert isinstance(env, SimMannequinEnv)
    env.set_pose_context(fk, None, None)

    for j, (low, high) in zip(
        env._movable_joints, zip(env._joint_lower, env._joint_upper)
    ):
        info = p.getJointInfo(env._robot, j, physicsClientId=env._cid)
        if info[8] > info[9]:
            continue
        assert np.isclose(low, info[8] + 0.3)
        assert np.isclose(high, info[9] - 0.3)

    env.execute(fk.arm_aa_to_q(_BENT_ARM_AA))
    # Loose tolerance: commands are clamped, achieved q may overshoot slightly.
    assert np.all(env._robot_q() >= env._joint_lower - 0.02)
    assert np.all(env._robot_q() <= env._joint_upper + 0.02)


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


def test_sim_mannequin_robot_action_interface() -> None:
    fk = SmplLeftArmFK()
    env = make_env("sim_mannequin")
    assert isinstance(env, SimMannequinEnv)
    env.set_pose_context(fk, None, None)
    q0 = fk.arm_aa_to_q(_BENT_ARM_AA)

    with pytest.raises(RuntimeError):
        env.execute_robot(np.zeros(7))

    grasp = env.current_grasp(q0)
    q_robot = env.current_robot_q()
    pos, _rot = env.robot_fk().ee_pose(q_robot)
    ee_pos, _ee_rot = env._ee_pose_pb()
    np.testing.assert_allclose(pos, ee_pos, atol=1e-6)
    g_pos, _g_rot = grasp.gripper_pose(*env._forearm_frame_pb(q0))
    np.testing.assert_allclose(g_pos, ee_pos, atol=1e-9)

    lower, upper = env.robot_joint_limits()
    assert lower.shape == upper.shape == q_robot.shape

    wrist_before = fk.fk(q_to_arm_aa(env._read_back_q(), fk.elbow_hinge_axis))[-1]
    target = q_robot.copy()
    target[1] += 0.15
    for _ in range(5):
        q_meas = env.execute_robot(target)
    assert q_meas.shape == (Q_DIM,)
    wrist_after = fk.fk(q_to_arm_aa(q_meas, fk.elbow_hinge_axis))[-1]
    assert float(np.linalg.norm(wrist_after - wrist_before)) > 0.01


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
