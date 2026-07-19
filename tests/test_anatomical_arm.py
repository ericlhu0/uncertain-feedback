"""Tests for the anatomical elbow-hinge / shoulder-rotation reparameterization.

Covers ``anatomical_elbow_wrist_slots`` (kinematics.py): the left-arm slot
reallocation that makes the elbow carry the recovered shoulder internal/external
rotation and the wrist a pure forearm hinge (zero pronation), while preserving
joint positions exactly.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.kinematics import anatomical_elbow_wrist_slots

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
