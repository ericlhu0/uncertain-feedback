"""Tests for the anatomical elbow-hinge / shoulder-rotation reparameterization.

Covers ``anatomical_elbow_wrist_slots`` (kinematics.py): the left-arm slot
reallocation that makes the elbow carry the recovered shoulder internal/external
rotation and the wrist a pure forearm hinge (zero pronation), while preserving
joint positions exactly.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc import ArmMPC, CartesianConfig
from uncertain_feedback.planners.mpc.arm_features import arm_feature_series
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import (
    Q_CLAVICLE,
    Q_DIM,
    SmplLeftArmFK,
    _rate_limited_step_q,
    anatomical_elbow_wrist_slots,
    q_to_arm_aa,
)

# Synthetic near-SMPL T-pose axes: upper arm along +x with a small forearm bend
# (the real SMPL left arm carries a ~7.5° bend, which defines the hinge plane).
_TPOSE_UPPER = np.array([1.0, 0.0, 0.0])
_TPOSE_FOREARM = np.array([np.cos(np.deg2rad(7.5)), 0.0, np.sin(np.deg2rad(7.5))])


def _slot_worlds(
    shoulder, elbow, wrist, parent
) -> tuple[Rotation, Rotation, np.ndarray, np.ndarray]:
    """Return (elbow_world, wrist_world, elbow_aa, wrist_aa) for a config."""
    elbow_aa, wrist_aa = anatomical_elbow_wrist_slots(
        shoulder, elbow, wrist, parent, _TPOSE_UPPER, _TPOSE_FOREARM
    )
    elbow_world = parent * Rotation.from_rotvec(elbow_aa)
    wrist_world = elbow_world * Rotation.from_rotvec(wrist_aa)
    return elbow_world, wrist_world, elbow_aa, wrist_aa


def test_positions_preserved_by_construction() -> None:
    """FK of the recovered slots reproduces both arm bone directions exactly."""
    rng = np.random.default_rng(0)
    upper_len, forearm_len = 0.30, 0.25
    for _ in range(200):
        parent = Rotation.from_rotvec(rng.uniform(-0.5, 0.5, 3))
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        # forearm at a realistic flexion angle (20-160 deg) in a random plane
        ang = np.deg2rad(rng.uniform(20, 160))
        perp = np.cross(u, rng.normal(size=3))
        perp /= np.linalg.norm(perp)
        f = np.cos(ang) * u + np.sin(ang) * perp
        shoulder = rng.normal(size=3)
        elbow = shoulder + upper_len * u
        wrist = elbow + forearm_len * f

        elbow_world, wrist_world, _, _ = _slot_worlds(shoulder, elbow, wrist, parent)
        got_u = elbow_world.apply(_TPOSE_UPPER / np.linalg.norm(_TPOSE_UPPER))
        got_f = wrist_world.apply(_TPOSE_FOREARM / np.linalg.norm(_TPOSE_FOREARM))
        np.testing.assert_allclose(got_u, u, atol=1e-9)
        np.testing.assert_allclose(got_f, f, atol=1e-9)


def test_wrist_slot_is_pure_hinge_zero_pronation() -> None:
    """The wrist slot never rolls about the forearm axis (no pronation)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        parent = Rotation.from_rotvec(rng.uniform(-0.5, 0.5, 3))
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        ang = np.deg2rad(rng.uniform(20, 160))
        perp = np.cross(u, rng.normal(size=3))
        perp /= np.linalg.norm(perp)
        f = np.cos(ang) * u + np.sin(ang) * perp
        shoulder = rng.normal(size=3)
        elbow = shoulder + 0.3 * u
        wrist = elbow + 0.25 * f

        _, _, _, wrist_aa = _slot_worlds(shoulder, elbow, wrist, parent)
        # The wrist-slot rotation axis must be orthogonal to the forearm bone it
        # acts on: zero twist component about the forearm direction.
        forearm_dir = _TPOSE_FOREARM / np.linalg.norm(_TPOSE_FOREARM)
        assert abs(np.dot(wrist_aa, forearm_dir)) < 1e-9


def test_continuity_through_near_antiparallel_sweep() -> None:
    """Sweeping the forearm smoothly gives no discontinuous hand-frame roll.

    The old shortest-arc gauge sat near the T-pose antiparallel singularity and
    produced ~12°/frame fabricated roll; the elbow-relative hinge tracks the
    forearm direction step (~forearm angular speed) instead.
    """
    parent = Rotation.identity()
    shoulder = np.zeros(3)
    elbow = np.array([0.3, 0.0, 0.0])
    prev_world = None
    prev_f = None
    max_frame_step = 0.0
    max_f_step = 0.0
    for t in np.linspace(0.0, 1.0, 400):
        ang = np.deg2rad(20 + 150 * t)  # elbow flexion 20 -> 170 deg
        f = np.array([np.cos(ang), 0.0, np.sin(ang)])
        wrist = elbow + 0.25 * f
        _, wrist_world, _, _ = _slot_worlds(shoulder, elbow, wrist, parent)
        if prev_world is not None:
            step = (prev_world.inv() * wrist_world).magnitude()
            f_step = np.arccos(np.clip(prev_f @ f, -1.0, 1.0))
            max_frame_step = max(max_frame_step, step)
            max_f_step = max(max_f_step, f_step)
        prev_world = wrist_world
        prev_f = f
    # Hand frame moves no faster than the forearm direction itself: no
    # fabricated roll (old gauge: ~12° per frame here).
    assert np.degrees(max_frame_step) < 1.5
    np.testing.assert_allclose(max_frame_step, max_f_step, atol=1e-9)


def test_recovers_shoulder_rotation_into_elbow_slot() -> None:
    """Rotating the flexion plane about the upper arm moves only the elbow-slot
    twist; the wrist-slot flexion angle is invariant."""
    parent = Rotation.identity()
    shoulder = np.zeros(3)
    u = np.array([1.0, 0.0, 0.0])
    elbow = shoulder + 0.3 * u
    flex = np.deg2rad(80.0)

    twists: list[float] = []
    wrist_angles: list[float] = []
    for plane_deg in (0.0, 40.0, 80.0, 120.0):
        # forearm at fixed flexion angle, in a plane rotated about u
        base_perp = np.array([0.0, 0.0, 1.0])
        perp = Rotation.from_rotvec(np.deg2rad(plane_deg) * u).apply(base_perp)
        f = np.cos(flex) * u + np.sin(flex) * perp
        wrist = elbow + 0.25 * f
        elbow_world, _, _, wrist_aa = _slot_worlds(shoulder, elbow, wrist, parent)
        # elbow-slot twist about u (recovered shoulder internal/external rotation)
        hinge = elbow_world.apply([0.0, 1.0, 0.0])  # canonical hinge carried to world
        twists.append(np.arctan2(np.dot(np.cross([0, 1, 0], hinge), u), hinge[1]))
        wrist_angles.append(float(np.linalg.norm(wrist_aa)))

    # Wrist flexion magnitude is invariant to the plane orientation ...
    np.testing.assert_allclose(wrist_angles, wrist_angles[0], atol=1e-9)
    # ... while the elbow-slot twist tracks the plane rotation (spans a range).
    assert np.degrees(np.ptp(twists)) > 90.0


def test_q_roundtrip_is_exact_for_hinge_constrained_states() -> None:
    fk = SmplLeftArmFK()
    rng = np.random.default_rng(2)
    states = rng.uniform(-0.8, 0.8, size=(100, Q_DIM))

    recovered = fk.arm_aa_to_q_batch(q_to_arm_aa(states, fk.elbow_hinge_axis))

    np.testing.assert_allclose(recovered, states, atol=1e-9)


def test_all_arm_features_match_for_q_and_decoded_boundary_states() -> None:
    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    q = np.random.default_rng(5).uniform(-0.6, 0.6, size=(20, Q_DIM))

    q_features = arm_feature_series(q, context)
    boundary_features = arm_feature_series(q_to_arm_aa(q, fk.elbow_hinge_axis), context)

    assert q_features.keys() == boundary_features.keys()
    for name in q_features:
        np.testing.assert_allclose(q_features[name], boundary_features[name], atol=1e-9)


def test_raw_arm_conversion_preserves_fk_positions() -> None:
    fk = SmplLeftArmFK()
    rng = np.random.default_rng(3)
    raw = rng.uniform(-0.8, 0.8, size=(100, 3, 3))

    converted = fk.arm_aa_to_q_batch(raw)
    reconstructed = q_to_arm_aa(converted, fk.elbow_hinge_axis)

    for original, constrained in zip(raw, reconstructed):
        np.testing.assert_allclose(fk.fk(constrained), fk.fk(original), atol=1e-9)


def test_rate_limited_q_step_reaches_target() -> None:
    current = np.zeros(Q_DIM)
    target = np.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.3, 0.55])
    reached = False

    for _ in range(20):
        current, reached = _rate_limited_step_q(current, target, 0.1)
        if reached:
            break

    assert reached
    np.testing.assert_allclose(current, target, atol=1e-9)


def test_scale_arm_lengths_sets_measured_segments() -> None:
    fk = SmplLeftArmFK()
    reference = SmplLeftArmFK()
    lengths = np.array([0.19, 0.33, 0.27])

    fk.scale_arm_lengths(*lengths)

    q = np.random.default_rng(6).uniform(-0.6, 0.6, size=Q_DIM)
    segments = np.diff(fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis))[1:], axis=0)
    np.testing.assert_allclose(np.linalg.norm(segments, axis=1), lengths, atol=1e-12)
    # Directions are untouched: the same q gives parallel bones on both
    # skeletons, and the hinge axis keeps its meaning.
    ref_segments = np.diff(
        reference.fk(q_to_arm_aa(q, reference.elbow_hinge_axis))[1:], axis=0
    )
    unit = segments / np.linalg.norm(segments, axis=1, keepdims=True)
    ref_unit = ref_segments / np.linalg.norm(ref_segments, axis=1, keepdims=True)
    np.testing.assert_allclose(unit, ref_unit, atol=1e-12)
    np.testing.assert_allclose(
        fk.elbow_hinge_axis, reference.elbow_hinge_axis, atol=1e-12
    )
    # Lengths are absolute, so re-applying is a no-op.
    tpose = fk.tpose_joints
    fk.scale_arm_lengths(*lengths)
    np.testing.assert_allclose(fk.tpose_joints, tpose, atol=1e-15)
    np.testing.assert_allclose(fk.tpose_all_joints[[13, 16, 18, 20]], tpose[1:])


def test_mpc_actions_never_move_the_clavicle() -> None:
    """The robot holds the forearm, so plans may not use the clavicle DOFs."""
    fk = SmplLeftArmFK()
    target = np.array([0.1, -0.2, 0.1, 0.2, 0.1, -0.1, 0.4])
    q0 = np.array([0.05, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
    wrist_rel = (
        fk.fk(q_to_arm_aa(target, fk.elbow_hinge_axis))[-1] - fk.tpose_spine3_pos
    )
    mpc = ArmMPC(
        horizon=3,
        n_mpc_samples=32,
        fk=fk,
        seed=7,
        initial_q=q0,
        cartesian=CartesianConfig(goals=[wrist_rel]),
    )

    _, plan = mpc.solve(q0)
    next_q = mpc.step(q0)

    assert np.all(plan[:, Q_CLAVICLE] == 0.0)
    np.testing.assert_allclose(next_q[Q_CLAVICLE], q0[Q_CLAVICLE], atol=1e-12)


def test_seeded_mpc_step_keeps_scalar_elbow_representation() -> None:
    fk = SmplLeftArmFK()
    target = np.array([0.1, -0.2, 0.1, 0.2, 0.1, -0.1, 0.4])
    wrist_rel = (
        fk.fk(q_to_arm_aa(target, fk.elbow_hinge_axis))[-1] - fk.tpose_spine3_pos
    )
    mpc = ArmMPC(
        horizon=2,
        n_mpc_samples=16,
        fk=fk,
        seed=4,
        initial_q=np.zeros(Q_DIM),
        cartesian=CartesianConfig(goals=[wrist_rel]),
    )

    next_q = mpc.step(np.zeros(Q_DIM))
    arm_aa = q_to_arm_aa(next_q, fk.elbow_hinge_axis)

    assert next_q.shape == (Q_DIM,)
    np.testing.assert_allclose(arm_aa[2], next_q[6] * fk.elbow_hinge_axis, atol=1e-12)


def _canonical_q(fk: SmplLeftArmFK, q, spine3_pos, spine3_aa) -> np.ndarray:
    """The decode-branch representative of ``q`` — what measuring envs report."""
    positions = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), spine3_pos, spine3_aa)
    aa = fk.arm_aa_from_positions(positions, spine3_aa)
    q_canon = fk.arm_aa_to_q(aa, spine3_aa)
    q_canon[Q_CLAVICLE] = np.asarray(q, dtype=np.float64)[Q_CLAVICLE]
    return q_canon


# Synthetic gripper origin in the forearm frame for the projection tests.
_GRASP_V = np.array([0.06, -0.02, 0.05])


def test_project_forearm_frames_roundtrips_on_manifold() -> None:
    """Frames FK'd from measured-style configurations project back exactly."""
    from uncertain_feedback.envs.grasp import forearm_frame_fk
    from uncertain_feedback.planners.mpc.kinematics import project_forearm_frames

    fk = SmplLeftArmFK()
    rng = np.random.default_rng(3)
    clavicle = rng.uniform(-0.2, 0.2, 3)
    spine3_aa = rng.uniform(-0.3, 0.3, 3)
    spine3_pos = rng.normal(size=3)
    q_ref = np.concatenate([clavicle, rng.uniform(-0.8, 0.8, 3), [-0.9]])
    for _ in range(30):
        q = _canonical_q(
            fk,
            np.concatenate(
                [clavicle, rng.uniform(-1.0, 1.0, 3), [rng.uniform(-2.0, -0.1)]]
            ),
            spine3_pos,
            spine3_aa,
        )
        elbow_pos, forearm_rot = forearm_frame_fk(fk, q, spine3_pos, spine3_aa)
        ee = elbow_pos + forearm_rot.apply(_GRASP_V)
        arm_aa, wrist_pos, residual = project_forearm_frames(
            fk, ee, forearm_rot.as_matrix(), _GRASP_V, q_ref, spine3_pos, spine3_aa
        )
        expected_aa = q_to_arm_aa(q, fk.elbow_hinge_axis)
        expected_pos = fk.fk(expected_aa, spine3_pos, spine3_aa)
        np.testing.assert_allclose(arm_aa, expected_aa, atol=1e-8)
        np.testing.assert_allclose(wrist_pos, expected_pos[4], atol=1e-8)
        assert residual < 1e-8


def test_project_forearm_frames_batched_matches_single() -> None:
    from uncertain_feedback.envs.grasp import forearm_frame_fk
    from uncertain_feedback.planners.mpc.kinematics import project_forearm_frames

    fk = SmplLeftArmFK()
    rng = np.random.default_rng(4)
    q_ref = np.concatenate([rng.uniform(-0.2, 0.2, 3), np.zeros(3), [-0.5]])
    ees, rots = [], []
    for _ in range(6):
        q = q_ref.copy()
        q[3:6] = rng.uniform(-1.0, 1.0, 3)
        q[6] = rng.uniform(-2.0, -0.1)
        elbow_pos, forearm_rot = forearm_frame_fk(fk, q, None, None)
        ees.append(
            elbow_pos + forearm_rot.apply(_GRASP_V) + rng.normal(scale=0.02, size=3)
        )
        rots.append(
            (
                Rotation.from_rotvec(rng.normal(scale=0.05, size=3)) * forearm_rot
            ).as_matrix()
        )
    ee_batch = np.stack(ees).reshape(2, 3, 3)
    rot_batch = np.stack(rots).reshape(2, 3, 3, 3)
    aa_batch, wrist_batch, res_batch = project_forearm_frames(
        fk, ee_batch, rot_batch, _GRASP_V, q_ref
    )
    assert aa_batch.shape == (2, 3, 3, 3)
    assert wrist_batch.shape == (2, 3, 3)
    assert res_batch.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            aa, wrist, res = project_forearm_frames(
                fk, ee_batch[i, j], rot_batch[i, j], _GRASP_V, q_ref
            )
            np.testing.assert_allclose(aa_batch[i, j], aa, atol=1e-12)
            np.testing.assert_allclose(wrist_batch[i, j], wrist, atol=1e-12)
            np.testing.assert_allclose(res_batch[i, j], res, atol=1e-12)


def test_project_forearm_frames_off_manifold() -> None:
    """Roll is reported untransmitted; a dragged ee keeps the grasp distance."""
    from uncertain_feedback.envs.grasp import forearm_frame_fk
    from uncertain_feedback.planners.mpc.kinematics import project_forearm_frames

    fk = SmplLeftArmFK()
    q_seed = np.zeros(Q_DIM)
    q_seed[6] = -0.8
    q_ref = _canonical_q(fk, q_seed, None, None)
    elbow_pos, forearm_rot = forearm_frame_fk(fk, q_ref, None, None)
    shoulder = fk.fk(q_to_arm_aa(q_ref, fk.elbow_hinge_axis), None, None)[2]
    upper_len = float(np.linalg.norm(elbow_pos - shoulder))

    # Pure roll about the forearm axis: position feasible, rotation is not.
    roll_axis = forearm_rot.apply(
        fk._bone_offsets[3] / np.linalg.norm(fk._bone_offsets[3])
    )
    bad_rot = Rotation.from_rotvec(0.4 * roll_axis) * forearm_rot
    ee_roll = elbow_pos + bad_rot.apply(_GRASP_V)
    arm_aa, wrist_pos, residual = project_forearm_frames(
        fk, ee_roll, bad_rot.as_matrix(), _GRASP_V, q_ref
    )
    # Same configuration recovered (rotvec rows may take the antipodal
    # representation near the +-pi boundary), only the roll is reported.
    expected_aa = q_to_arm_aa(q_ref, fk.elbow_hinge_axis)
    for row in range(3):
        relative = (
            Rotation.from_rotvec(arm_aa[row])
            * Rotation.from_rotvec(expected_aa[row]).inv()
        )
        assert float(np.linalg.norm(relative.as_rotvec())) < 1e-8
    assert residual == pytest.approx(0.4, abs=1e-6)

    # Gripper dragged radially outward: the projected arm keeps both the
    # upper-arm length and the rigid elbow-to-gripper distance (the position
    # coupling physics enforces), and reports the elbow displacement.
    ee_true = elbow_pos + forearm_rot.apply(_GRASP_V)
    pull = (ee_true - shoulder) / np.linalg.norm(ee_true - shoulder)
    ee_far = ee_true + 0.02 * pull
    arm_aa, wrist_pos, residual = project_forearm_frames(
        fk, ee_far, forearm_rot.as_matrix(), _GRASP_V, q_ref
    )
    positions = fk.fk(arm_aa, None, None)
    np.testing.assert_allclose(
        np.linalg.norm(positions[3] - shoulder), upper_len, atol=1e-8
    )
    np.testing.assert_allclose(
        np.linalg.norm(ee_far - positions[3]),
        np.linalg.norm(_GRASP_V),
        atol=1e-8,
    )
    np.testing.assert_allclose(positions[4], wrist_pos, atol=1e-8)
    assert residual > 0.01
