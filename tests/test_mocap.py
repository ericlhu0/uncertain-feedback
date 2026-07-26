"""Tests for the NatNet decoder and the mocap→planner registration."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.sim_mannequin import _SMPL_TO_PB
from uncertain_feedback.mocap.natnet import (
    NAT_FRAMEOFDATA,
    MocapStaleError,
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
