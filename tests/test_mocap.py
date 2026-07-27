"""Tests for the NatNet decoder and the mocap→planner registration."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import struct
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.sim_mannequin import _SMPL_TO_PB
from uncertain_feedback.mocap.natnet import (
    NAT_FRAMEOFDATA,
    MocapStaleError,
    RigidBodyPose,
    decode_frame,
    require_fresh,
)
from uncertain_feedback.mocap.registration import (
    _LEFT_COLLAR_22,
    _RIGHT_COLLAR_22,
    ArmRegistration,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa

_TRUE_YAW = 0.83


def _frame_bytes(
    bodies: list[tuple[int, tuple[float, ...], tuple[float, ...], bool]],
    marker_sets: tuple[tuple[str, int], ...] = (("kinova_base", 3), ("all", 16)),
    n_unlabeled: int = 6,
) -> bytes:
    """Synthesize a NatNet FrameOfData datagram."""
    payload = struct.pack("<i", 4242) + struct.pack("<i", len(marker_sets))
    for name, n_markers in marker_sets:
        payload += name.encode() + b"\0" + struct.pack("<i", n_markers)
        payload += struct.pack(f"<{3 * n_markers}f", *range(3 * n_markers))
    payload += struct.pack("<i", n_unlabeled)
    payload += struct.pack(f"<{3 * n_unlabeled}f", *range(3 * n_unlabeled))
    payload += struct.pack("<i", len(bodies))
    for body_id, position, orientation, valid in bodies:
        payload += struct.pack("<i", body_id)
        payload += struct.pack("<7f", *position, *orientation)
        payload += struct.pack("<f", 0.0007)
        payload += struct.pack("<h", 1 if valid else 0)
    return struct.pack("<HH", NAT_FRAMEOFDATA, len(payload)) + payload


def test_decode_frame_skips_marker_sections() -> None:
    bodies = decode_frame(
        _frame_bytes(
            [
                (4, (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), True),
                (2, (-1.0, 0.5, 1.5), (0.5, 0.5, 0.5, 0.5), True),
                (40, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), False),
            ]
        )
    )

    assert sorted(bodies) == [2, 4, 40]
    np.testing.assert_allclose(bodies[4].position, [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(bodies[2].orientation, [0.5, 0.5, 0.5, 0.5], atol=1e-6)
    assert bodies[4].valid
    assert not bodies[40].valid


def test_decode_frame_without_marker_sections() -> None:
    bodies = decode_frame(
        _frame_bytes(
            [(7, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), True)],
            marker_sets=(),
            n_unlabeled=0,
        )
    )

    np.testing.assert_allclose(bodies[7].position, [1.0, 2.0, 3.0], atol=1e-6)


def test_require_fresh_boundary() -> None:
    require_fresh(0.4, 0.5)
    require_fresh(0.5, 0.5)
    with pytest.raises(MocapStaleError):
        require_fresh(0.51, 0.5)


def _canonical_q(fk: SmplLeftArmFK, arm_aa: np.ndarray) -> np.ndarray:
    """Round ``arm_aa`` through positions so its clavicle block is canonical.

    ``arm_aa_from_positions`` reconstructs the clavicle slot as the minimal
    rotation onto the measured clavicle bone, so only such states can be
    recovered from positions at all.
    """
    return fk.arm_aa_to_q(fk.arm_aa_from_positions(fk.fk(arm_aa), None), None)


def test_registration_solves_yaw_and_recovers_q() -> None:
    """The collar-to-collar axis pins the yaw, and q then round-trips through mocap.

    Synthesizes a mocap frame under a random base orientation, an arbitrary
    mocap origin, an unknown yaw, and deliberately non-SMPL limb lengths — only
    bone directions may survive, and the yaw must be recovered from the
    measured left->right collar axis rather than supplied. No slot of a nominal
    startup q enters at all: the whole configuration, clavicle included, is
    measured.
    """
    fk = SmplLeftArmFK()
    q_true = _canonical_q(
        fk,
        np.array(
            [
                [-0.2578684731175734, 0.11215073986226905, -0.44187915175365783],
                [0.6868865380518193, -0.18, -0.11006289820501858],
                [0.035651850950680734, -1.23, 0.7515284911778386],
            ]
        ),
    )
    rng = np.random.default_rng(0)

    # The base body's orientation affects only the robot's scene facing, never
    # the measured bone directions, so the mapping is a pure yaw.
    plate_yaw = -0.41
    base_orientation = Rotation.from_euler("z", plate_yaw).as_quat()
    rotation = Rotation.from_euler("z", _TRUE_YAW)
    positions = fk.fk(q_to_arm_aa(q_true, fk.elbow_hinge_axis))
    collar_pb, shoulder_pb, elbow_pb, wrist_pb = (
        _SMPL_TO_PB @ pos for pos in positions[1:5]
    )
    tpose = fk.tpose_all_joints
    across_pb = _SMPL_TO_PB @ (tpose[_RIGHT_COLLAR_22] - tpose[_LEFT_COLLAR_22])

    collar_mocap = rng.normal(size=3)
    to_mocap = rotation.inv().apply
    collar_right_mocap = collar_mocap + to_mocap(across_pb) * 0.9
    shoulder_mocap = collar_mocap + to_mocap(shoulder_pb - collar_pb) * 1.15
    elbow_mocap = shoulder_mocap + to_mocap(elbow_pb - shoulder_pb) * 1.3
    wrist_mocap = elbow_mocap + to_mocap(wrist_pb - elbow_pb) * 0.7
    base_offset_pb = np.array([0.6, -0.35, -0.35])
    base_position = collar_mocap + to_mocap(base_offset_pb)

    registration = ArmRegistration.calibrate(
        fk=fk,
        spine3_pos=None,
        spine3_aa=None,
        base_position=base_position,
        base_orientation=base_orientation,
        collar_mocap=collar_mocap,
        collar_right_mocap=collar_right_mocap,
    )

    np.testing.assert_allclose(
        registration.rotation.as_matrix(), rotation.as_matrix(), atol=1e-9
    )
    assert np.isclose(registration.robot_base_yaw, _TRUE_YAW + plate_yaw, atol=1e-9)
    # Both bodies are placed where they were measured, so what mocap fixes is
    # the human->robot offset — not the person's position in a pinned scene.
    np.testing.assert_allclose(
        registration.collar_pb, rotation.apply(collar_mocap), atol=1e-12
    )
    np.testing.assert_allclose(
        registration.base_pb - registration.collar_pb, base_offset_pb, atol=1e-9
    )
    q_start = registration.q_from_keypoints(
        collar_mocap, shoulder_mocap, elbow_mocap, wrist_mocap
    )
    np.testing.assert_allclose(q_start, q_true, atol=1e-6)
    # The torso anchor moved with the person: FK from it reproduces the measured
    # collar, so the run's spine3-relative goals hang off the real body.
    np.testing.assert_allclose(
        fk.fk(q_to_arm_aa(q_start, fk.elbow_hinge_axis), registration.spine3_smpl)[1],
        _SMPL_TO_PB.T @ registration.collar_pb,
        atol=1e-9,
    )


def test_vertical_collar_axis_is_rejected() -> None:
    """A vertical collar-to-collar axis leaves the yaw undetermined, so it must not be guessed."""
    fk = SmplLeftArmFK()
    collar = np.array([0.0, 0.0, 1.2])
    with pytest.raises(ValueError, match="undetermined"):
        ArmRegistration.calibrate(
            fk=fk,
            spine3_pos=None,
            spine3_aa=None,
            base_position=np.array([1.0, 0.5, 0.0]),
            base_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            collar_mocap=collar,
            collar_right_mocap=collar + np.array([0.0, 0.0, 0.1]),
        )


def _calibrate_at(
    fk: SmplLeftArmFK, collar: np.ndarray, collar_right: np.ndarray
) -> ArmRegistration:
    return ArmRegistration.calibrate(
        fk=fk,
        spine3_pos=None,
        spine3_aa=None,
        base_position=np.array([1.0, 0.5, 0.0]),
        base_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        collar_mocap=collar,
        collar_right_mocap=collar_right,
    )


def test_torso_anchor_follows_the_person_between_runs() -> None:
    """A person who sits elsewhere next run moves; the bolted-down robot does not.

    The whole point of measuring the collar: the torso anchor is not a constant
    of the config, so two runs with the person in different places give anchors
    that differ by exactly where they moved.
    """
    fk = SmplLeftArmFK()
    collar = np.array([0.0, 0.0, 1.2])
    collar_right = collar + np.array([-0.10, 0.24, 0.01])
    shift = np.array([0.4, -0.9, 0.15])

    first = _calibrate_at(fk, collar, collar_right)
    second = _calibrate_at(fk, collar + shift, collar_right + shift)

    # Same collar axis both times, so the same yaw — the person only moved.
    np.testing.assert_allclose(
        second.rotation.as_matrix(), first.rotation.as_matrix(), atol=1e-12
    )
    shift_pb = first.rotation.apply(shift)
    np.testing.assert_allclose(second.collar_pb - first.collar_pb, shift_pb, atol=1e-12)
    np.testing.assert_allclose(
        second.spine3_smpl - first.spine3_smpl, _SMPL_TO_PB.T @ shift_pb, atol=1e-12
    )
    np.testing.assert_allclose(second.base_pb, first.base_pb, atol=1e-12)


def test_registration_ignores_mocap_translation() -> None:
    """Within a run only bone directions are used, so a shifted person's q is the same.

    The anchor is fixed at calibration, so mid-run torso translation is
    deliberately discarded — unlike between runs
    (:func:`test_torso_anchor_follows_the_person_between_runs`).
    """
    fk = SmplLeftArmFK()
    collar = np.array([0.0, 0.0, 1.2])
    collar_right = collar + np.array([-0.10, 0.24, 0.01])
    shoulder = collar + np.array([0.10, 0.02, 0.01])
    elbow = shoulder + np.array([0.30, 0.05, -0.10])
    wrist = elbow + np.array([0.22, -0.08, -0.12])
    registration = _calibrate_at(fk, collar, collar_right)

    shift = np.array([0.4, -0.9, 0.15])
    np.testing.assert_allclose(
        registration.q_from_keypoints(collar, shoulder, elbow, wrist),
        registration.q_from_keypoints(
            collar + shift, shoulder + shift, elbow + shift, wrist + shift
        ),
        atol=1e-12,
    )


class _StubReceiver:
    """One frozen mocap frame, standing in for a live NatNet stream."""

    def __init__(self, bodies: dict[int, RigidBodyPose]) -> None:
        self._bodies = bodies

    def latest(self) -> tuple[float, dict[int, RigidBodyPose]]:
        return time.monotonic(), self._bodies

    def wait_for(
        self, body_ids, timeout: float  # pylint: disable=unused-argument
    ) -> dict[int, RigidBodyPose]:
        return self._bodies


def _mocap_frame(
    fk: SmplLeftArmFK, arm_aa: np.ndarray, base_offset_pb: np.ndarray
) -> dict[int, RigidBodyPose]:
    """Synthesize the six rigid bodies `RealEnv` registers against.

    The person is built from `arm_aa` so the measured configuration is known,
    and the robot base is placed at a chosen offset from their collar — the one
    thing a real setup varies run to run, and what the IK branch turns on.
    """
    rotation = Rotation.from_euler("z", _TRUE_YAW)
    to_mocap = rotation.inv().apply
    collar_pb, shoulder_pb, elbow_pb, wrist_pb = (
        _SMPL_TO_PB @ pos for pos in fk.fk(arm_aa)[1:5]
    )
    tpose = fk.tpose_all_joints
    across_pb = _SMPL_TO_PB @ (tpose[_RIGHT_COLLAR_22] - tpose[_LEFT_COLLAR_22])
    collar = np.array([0.3, -1.1, 0.0])
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    points = {
        1: collar + to_mocap(base_offset_pb),
        2: collar,
        6: collar + to_mocap(across_pb),
        3: collar + to_mocap(shoulder_pb - collar_pb),
        4: collar + to_mocap(elbow_pb - collar_pb),
        5: collar + to_mocap(wrist_pb - collar_pb),
    }
    return {i: RigidBodyPose(pos, identity, True) for i, pos in points.items()}


def _gripper_offset_from_forearm(env, q: np.ndarray) -> float:
    """Distance from the IK robot's gripper to the forearm segment, in metres."""
    elbow, wrist = (
        _SMPL_TO_PB @ pos
        for pos in env._fk.fk(  # pylint: disable=protected-access
            q_to_arm_aa(
                q, env._fk.elbow_hinge_axis
            ),  # pylint: disable=protected-access
            env._spine3_pos,  # pylint: disable=protected-access
            env._spine3_aa,  # pylint: disable=protected-access
        )[3:5]
    )
    gripper, _ = env._ee_pose_pb()  # pylint: disable=protected-access
    bone = wrist - elbow
    along = float(np.clip(np.dot(gripper - elbow, bone) / np.dot(bone, bone), 0.0, 1.0))
    return float(np.linalg.norm(gripper - (elbow + along * bone)))


def test_batched_gen3_seed_tracking_reaches_exact_targets() -> None:
    from ssik.prebuilt import gen3_ik

    from uncertain_feedback.envs.real import _gen3_seeded_track_batch

    rng = np.random.default_rng(4)
    seeds = rng.normal(0.0, 0.3, (64, 7))
    targets = np.stack(
        [gen3_ik.fk(q) for q in seeds + rng.normal(0.0, 0.001, seeds.shape)]
    )
    solutions, feasible = _gen3_seeded_track_batch(
        targets,
        seeds,
        np.full(7, -10.0),
        np.full(7, 10.0),
        np.zeros(7, dtype=bool),
    )

    assert np.all(feasible)
    residuals = np.array(
        [
            np.linalg.norm(gen3_ik.fk(q) - target)
            for q, target in zip(solutions, targets)
        ]
    )
    assert np.max(residuals) < 1e-10


@pytest.mark.parametrize(
    "base_offset_pb", [(0.6, -0.35, -0.35), (0.7, -0.2, -0.4), (0.8, 0.0, -0.45)]
)
def test_ik_keeps_the_gripper_on_the_forearm(monkeypatch, base_offset_pb) -> None:
    """The plan preview's robot must hold the forearm, not float away from it.

    `_solve_ik` only accepts a configuration inside the controller's joint
    limits. Solving without them and clipping afterwards instead moves the end
    effector by however far the clip travelled with nothing re-solving, which
    for these ordinary base placements put the gripper up to 0.9 m off the arm —
    visible in the preview as a robot reaching somewhere the person is not.
    """
    pytest.importorskip("pybullet")
    from uncertain_feedback.envs import real as real_module

    if not real_module._ROBOT_SPECS["kinova_gen3"].urdf.exists():
        pytest.skip("kortex_description URDF not available")

    fk = SmplLeftArmFK()
    arm_aa = np.array(
        [
            [-0.2578684731175734, 0.11215073986226905, -0.44187915175365783],
            [0.6868865380518193, -0.18, -0.11006289820501858],
            [0.035651850950680734, -1.23, 0.7515284911778386],
        ]
    )
    bodies = _mocap_frame(fk, arm_aa, np.asarray(base_offset_pb, dtype=np.float64))
    monkeypatch.setattr(
        real_module.NatNetReceiver,
        "connect",
        staticmethod(lambda *a, **k: _StubReceiver(bodies)),
    )

    env = real_module.RealEnv(
        mocap_host="stub",
        mocap_rigid_bodies={
            "robot_base": 1,
            "collar": 2,
            "collar_right": 6,
            "shoulder": 3,
            "elbow": 4,
            "wrist": 5,
        },
        real_mirror_host=None,
        live_view=False,
    )
    env.set_pose_context(fk, None, None, fk.tpose_all_joints.copy())
    q = env.initial_q(fk.arm_aa_to_q(arm_aa, None))
    env._capture_grasp(q)  # pylint: disable=protected-access

    assert _gripper_offset_from_forearm(env, q) < 0.05

    # And it stays there as the plan sweeps the arm, which is what the preview
    # animates: a solution pinned against a limit walks the gripper off instead.
    for _ in range(400):
        q = q.copy()
        q[6] += 0.001
        env._drive(q, mirror=False)  # pylint: disable=protected-access
    assert _gripper_offset_from_forearm(env, q) < 0.1


def test_ik_holds_position_when_the_pose_is_infeasible(monkeypatch) -> None:
    """A reach that rotates the shoulder must not walk the gripper off the arm.

    The grasp is rigid, so a reach that rotates the forearm demands a gripper
    attitude along with the position — and pybullet's null-space solver gave up
    *position* to keep chasing orientation, drawing a preview whose robot was
    right at the start configuration and drifted off the forearm as the plan
    played (15 cm by the end of this sweep). Continuing the analytical branch
    tracks the whole rigid pose while it is reachable — which, with the
    controller's URDF-derived joint limits, is this entire sweep — and the
    weighted fallback keeps the gripper on the forearm, attitude carrying the
    miss, when it is not.
    """
    pytest.importorskip("pybullet")
    from uncertain_feedback.envs import real as real_module

    if not real_module._ROBOT_SPECS["kinova_gen3"].urdf.exists():
        pytest.skip("kortex_description URDF not available")

    fk = SmplLeftArmFK()
    arm_aa = np.array(
        [
            [-0.2578684731175734, 0.11215073986226905, -0.44187915175365783],
            [0.6868865380518193, -0.18, -0.11006289820501858],
            [0.035651850950680734, -1.23, 0.7515284911778386],
        ]
    )
    bodies = _mocap_frame(fk, arm_aa, np.array([0.7, -0.2, -0.4]))
    monkeypatch.setattr(
        real_module.NatNetReceiver,
        "connect",
        staticmethod(lambda *a, **k: _StubReceiver(bodies)),
    )

    env = real_module.RealEnv(
        mocap_host="stub",
        mocap_rigid_bodies={
            "robot_base": 1,
            "collar": 2,
            "collar_right": 6,
            "shoulder": 3,
            "elbow": 4,
            "wrist": 5,
        },
        real_mirror_host=None,
        live_view=False,
    )
    env.set_pose_context(fk, None, None, fk.tpose_all_joints.copy())
    q0 = env.initial_q(fk.arm_aa_to_q(arm_aa, None))
    env._capture_grasp(q0)  # pylint: disable=protected-access

    q_goal = q0.copy()
    q_goal[3:6] += np.array([0.35, -0.45, 0.3])
    q_goal[6] -= 0.5
    worst_offset = 0.0
    worst_attitude = 0.0
    for i in range(1, 601):
        q = q0 + (q_goal - q0) * (i / 600)
        env._drive(q, mirror=False)  # pylint: disable=protected-access
        worst_offset = max(worst_offset, _gripper_offset_from_forearm(env, q))
        _, target_rot = env._grasp_pose_pb(q)  # pylint: disable=protected-access
        _, ee_rot = env._ee_pose_pb()  # pylint: disable=protected-access
        worst_attitude = max(
            worst_attitude, float((ee_rot * target_rot.inv()).magnitude())
        )
    assert worst_offset < 0.05
    assert worst_attitude < np.deg2rad(5.0)


def test_closed_live_view_window_continues_headless(monkeypatch, capsys) -> None:
    """Closing the viz window must drop to headless, not end the process.

    The GUI window is the pybullet client, and that client also holds the IK
    robot — a close makes every later pybullet call raise. The env must
    reconnect DIRECT, restore the robot's joint state, and keep stepping.
    Simulated by disconnecting the client, which is what the close does to
    the connection.
    """
    pytest.importorskip("pybullet")
    import pybullet as p

    from uncertain_feedback.envs import real as real_module

    if not real_module._ROBOT_SPECS["kinova_gen3"].urdf.exists():
        pytest.skip("kortex_description URDF not available")

    fk = SmplLeftArmFK()
    arm_aa = np.array(
        [
            [-0.2578684731175734, 0.11215073986226905, -0.44187915175365783],
            [0.6868865380518193, -0.18, -0.11006289820501858],
            [0.035651850950680734, -1.23, 0.7515284911778386],
        ]
    )
    bodies = _mocap_frame(fk, arm_aa, np.array([0.7, -0.2, -0.4]))
    monkeypatch.setattr(
        real_module.NatNetReceiver,
        "connect",
        staticmethod(lambda *a, **k: _StubReceiver(bodies)),
    )

    env = real_module.RealEnv(
        mocap_host="stub",
        mocap_rigid_bodies={
            "robot_base": 1,
            "collar": 2,
            "collar_right": 6,
            "shoulder": 3,
            "elbow": 4,
            "wrist": 5,
        },
        real_mirror_host=None,
        live_view=False,
    )
    env.set_pose_context(fk, None, None, fk.tpose_all_joints.copy())
    q0 = env.initial_q(fk.arm_aa_to_q(arm_aa, None))
    env._capture_grasp(q0)  # pylint: disable=protected-access
    robot_q = env._current_q()  # pylint: disable=protected-access

    p.disconnect(env._cid)  # pylint: disable=protected-access
    achieved = env.execute_robot(robot_q)

    assert "continuing headless" in capsys.readouterr().out
    cid = env._cid  # pylint: disable=protected-access
    assert p.getConnectionInfo(physicsClientId=cid)["isConnected"]
    np.testing.assert_allclose(
        env._current_q(), robot_q, atol=1e-9  # pylint: disable=protected-access
    )
    np.testing.assert_allclose(achieved, q0, atol=1e-9)


def test_preview_reports_the_grasp_error(monkeypatch, capsys) -> None:
    """The preview must report how far the gripper strays, not just animate.

    Driven past what `robot_max_joint_delta` can keep up with, so the reported
    error is a real miss rather than the zero an exactly-tracked plan gives —
    otherwise the line could print zeros forever and nobody would notice.
    """
    pytest.importorskip("pybullet")
    from uncertain_feedback.envs import real as real_module

    if not real_module._ROBOT_SPECS["kinova_gen3"].urdf.exists():
        pytest.skip("kortex_description URDF not available")

    fk = SmplLeftArmFK()
    arm_aa = np.array(
        [
            [-0.2578684731175734, 0.11215073986226905, -0.44187915175365783],
            [0.6868865380518193, -0.18, -0.11006289820501858],
            [0.035651850950680734, -1.23, 0.7515284911778386],
        ]
    )
    bodies = _mocap_frame(fk, arm_aa, np.array([0.7, -0.2, -0.4]))
    monkeypatch.setattr(
        real_module.NatNetReceiver,
        "connect",
        staticmethod(lambda *a, **k: _StubReceiver(bodies)),
    )

    env = real_module.RealEnv(
        mocap_host="stub",
        mocap_rigid_bodies={
            "robot_base": 1,
            "collar": 2,
            "collar_right": 6,
            "shoulder": 3,
            "elbow": 4,
            "wrist": 5,
        },
        real_mirror_host=None,
        live_view=False,
    )
    env.set_pose_context(fk, None, None, fk.tpose_all_joints.copy())
    q0 = env.initial_q(fk.arm_aa_to_q(arm_aa, None))
    env._capture_grasp(q0)  # pylint: disable=protected-access

    # One stride of the sweep the infeasibility test walks in 600 steps, so the
    # delta clip cannot absorb it and the gripper is left behind the grasp.
    q_far = q0.copy()
    q_far[3:6] += np.array([0.35, -0.45, 0.3])
    env._drive(q_far, mirror=False)  # pylint: disable=protected-access
    position, attitude = env._grasp_error(q_far)  # pylint: disable=protected-access
    assert position > 0.01

    env._print_grasp_error(  # pylint: disable=protected-access
        np.array([[0.0, 0.0], [position, attitude]])
    )
    line = capsys.readouterr().out
    assert "grasp error over the plan" in line
    assert f"max {1e3 * position:.1f} mm at step 1" in line
