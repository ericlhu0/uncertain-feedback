"""Canonical anatomical features for the 7-DOF left-arm state."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.kinematics import (
    ELBOW_CHAIN_IDX,
    Q_CLAVICLE,
    Q_DIM,
    Q_ELBOW,
    Q_SHOULDER,
    SHOULDER_CHAIN_IDX,
    WRIST_CHAIN_IDX,
    SmplLeftArmFK,
    _shortest_arc_rotvecs,
    q_to_arm_aa,
)

FEATURE_NAMES = (
    "elbow_flexion",
    "shoulder_flexion_extension",
    "shoulder_abduction_adduction",
    "shoulder_elevation",
    "shoulder_internal_external_rotation",
)


class ArmFeatureContext(Protocol):
    """FK state required to interpret an arm trajectory."""

    @property
    def fk(self) -> SmplLeftArmFK:
        """Forward-kinematics model for the controlled arm."""
        raise NotImplementedError

    @property
    def spine3_pos(self) -> np.ndarray:
        """World-space spine3 anchor position."""
        raise NotImplementedError

    @property
    def spine3_aa(self) -> np.ndarray:
        """World-space spine3 anchor rotation vector."""
        raise NotImplementedError


def canonical_arm_q(
    trajectory: np.ndarray,
    context: ArmFeatureContext,
) -> np.ndarray:
    """Return ``trajectory`` in canonical ``(..., 7)`` planner coordinates."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.shape[-1:] == (Q_DIM,):
        return trajectory
    if trajectory.shape[-2:] != (3, 3):
        raise ValueError(
            f"arm trajectory must end in ({Q_DIM},) or (3, 3), got {trajectory.shape}"
        )
    if trajectory.size == 0:
        return np.empty((*trajectory.shape[:-2], Q_DIM), dtype=np.float64)
    return context.fk.arm_aa_to_q_batch(trajectory, context.spine3_aa)


def arm_aa_from_state(
    trajectory: np.ndarray,
    context: ArmFeatureContext,
) -> np.ndarray:
    """Return ``trajectory`` in the FK boundary representation ``(..., 3, 3)``."""
    q = canonical_arm_q(trajectory, context)
    return q_to_arm_aa(q, context.fk.elbow_hinge_axis)


def arm_feature_series(
    trajectory: np.ndarray,
    context: ArmFeatureContext,
) -> dict[str, np.ndarray]:
    """Return every canonical anatomical feature for an arm trajectory.

    Shoulder internal/external rotation is the twist stored in the anatomical
    shoulder block ``q[..., 3:6]``. Clavicle rotation does not contribute to it.
    """
    q = canonical_arm_q(trajectory, context)
    leading = q.shape[:-1]
    flat = q.reshape((-1, Q_DIM))
    if flat.shape[0] == 0:
        return {name: np.empty(leading, dtype=np.float64) for name in FEATURE_NAMES}

    upper_axis = _tpose_bone_axis(context.fk, SHOULDER_CHAIN_IDX, ELBOW_CHAIN_IDX)
    forearm_axis = _tpose_bone_axis(context.fk, ELBOW_CHAIN_IDX, WRIST_CHAIN_IDX)
    arm_aa = q_to_arm_aa(flat, context.fk.elbow_hinge_axis)

    forearm = Rotation.from_rotvec(arm_aa[:, 2]).apply(forearm_axis)
    elbow_flexion = np.arccos(np.clip(forearm @ upper_axis, -1.0, 1.0))

    upper_arm_rotation = (
        Rotation.from_rotvec(context.fk.collar_aa[None])
        * Rotation.from_rotvec(flat[:, Q_CLAVICLE])
        * Rotation.from_rotvec(flat[:, Q_SHOULDER])
    )
    upper_arm = upper_arm_rotation.apply(upper_axis)
    shoulder_flexion = np.arcsin(np.clip(upper_arm[:, 2], -1.0, 1.0))
    shoulder_abduction = np.arcsin(np.clip(upper_arm[:, 0], -1.0, 1.0))
    shoulder_elevation = np.arccos(np.clip(-upper_arm[:, 1], -1.0, 1.0))
    shoulder_rotation = _twist_angles_about_axis(flat[:, Q_SHOULDER], upper_axis)

    values = (
        elbow_flexion,
        shoulder_flexion,
        shoulder_abduction,
        shoulder_elevation,
        shoulder_rotation,
    )
    return {name: value.reshape(leading) for name, value in zip(FEATURE_NAMES, values)}


def arm_q_from_features(
    features: np.ndarray,
    clavicle: np.ndarray,
    context: ArmFeatureContext,
) -> np.ndarray:
    """Invert :func:`arm_feature_series`: ``(T, 5)`` features -> ``(T, 7)`` states.

    ``features`` holds radians in :data:`FEATURE_NAMES` order; ``clavicle`` is the
    ``(3,)`` clavicle rotation held fixed on every frame (the planner never
    actuates it). Flexion, abduction and elevation over-determine the two-DOF
    upper-arm direction, so an inconsistent triple — what an LLM writing
    anatomical trajectories produces — is resolved by taking the lateral and depth
    components from abduction/flexion, only the sign of the vertical component
    from elevation, and normalizing.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"features must have shape (T, {len(FEATURE_NAMES)}), got {features.shape}"
        )
    clavicle = np.asarray(clavicle, dtype=np.float64)
    upper_axis = _tpose_bone_axis(context.fk, SHOULDER_CHAIN_IDX, ELBOW_CHAIN_IDX)

    elbow_flexion = features[:, 0]
    flexion, abduction, elevation, rotation = (features[:, i] for i in range(1, 5))
    lateral, depth = np.sin(abduction), np.sin(flexion)
    vertical = -np.sign(np.cos(elevation)) * np.sqrt(
        np.maximum(0.0, 1.0 - lateral**2 - depth**2)
    )
    upper_arm = np.stack([lateral, vertical, depth], axis=-1)
    upper_arm /= np.maximum(np.linalg.norm(upper_arm, axis=-1, keepdims=True), 1e-12)

    parent = Rotation.from_rotvec(context.fk.collar_aa) * Rotation.from_rotvec(clavicle)
    swing = Rotation.from_rotvec(
        _shortest_arc_rotvecs(upper_axis, parent.inv().apply(upper_arm))
    )
    twist = Rotation.from_rotvec(rotation[:, np.newaxis] * upper_axis)

    q = np.empty((features.shape[0], Q_DIM), dtype=np.float64)
    q[:, Q_CLAVICLE] = clavicle
    q[:, Q_SHOULDER] = (swing * twist).as_rotvec()
    q[:, Q_ELBOW] = _elbow_scalar_from_flexion(elbow_flexion, context.fk)
    return q


def elbow_heights(
    trajectory: np.ndarray,
    context: ArmFeatureContext,
) -> np.ndarray:
    """Return spine3-relative elbow Y heights for any arm-state representation."""
    arm_aa = arm_aa_from_state(trajectory, context)
    leading = arm_aa.shape[:-2]
    flat = arm_aa.reshape((-1, 3, 3))
    positions = context.fk.fk_batch(flat, context.spine3_pos, context.spine3_aa)
    return (positions[:, ELBOW_CHAIN_IDX, 1] - context.spine3_pos[1]).reshape(leading)


def shoulder_abduction_angles(
    trajectory: np.ndarray,
    context: ArmFeatureContext,
) -> np.ndarray:
    """Return the unsigned upper-arm angle from the torso-down direction."""
    return arm_feature_series(trajectory, context)["shoulder_elevation"]


def resample_equidistant(traj: np.ndarray, n: int) -> np.ndarray:
    """Resample a ``(T, ...)`` trajectory to ``n`` points equidistant in joint-space
    arclength.

    Removes timing entirely — speed differences and dwell frames carry no signal
    here (MDM frames and MPC steps are on different clocks, and MDM output is
    systematically slower than a fresh rollout), so trajectories compare purely by
    path. A stationary trajectory becomes its pose repeated ``n`` times.
    """
    traj = np.asarray(traj, dtype=np.float64)
    flat = traj.reshape(traj.shape[0], -1)
    segments = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segments)])
    if arclength[-1] <= 0.0:
        return np.repeat(traj[:1], n, axis=0)
    targets = np.linspace(0.0, arclength[-1], n)
    out = np.stack(
        [np.interp(targets, arclength, flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape((n, *traj.shape[1:]))


def _tpose_bone_axis(fk: SmplLeftArmFK, start_idx: int, end_idx: int) -> np.ndarray:
    tpose = fk.tpose_joints
    axis = tpose[end_idx] - tpose[start_idx]
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return axis / norm


def _elbow_scalar_from_flexion(flexion: np.ndarray, fk: SmplLeftArmFK) -> np.ndarray:
    """Invert the elbow-flexion feature on its monotone branch.

    The hinge axis is the T-pose flexion-plane normal, so rotating the forearm
    about it keeps the forearm in that plane: the feature is exactly the T-pose
    bend plus the scalar elbow angle, folded through ``arccos``. Only the fold's
    orientation depends on the skeleton, so it is probed rather than restated
    here. Flexion under the T-pose bend inverts to a hyperextended elbow.
    """
    upper_axis = _tpose_bone_axis(fk, SHOULDER_CHAIN_IDX, ELBOW_CHAIN_IDX)
    forearm_axis = _tpose_bone_axis(fk, ELBOW_CHAIN_IDX, WRIST_CHAIN_IDX)

    def measure(angle: float) -> float:
        forearm = Rotation.from_rotvec(angle * fk.elbow_hinge_axis).apply(forearm_axis)
        return float(np.arccos(np.clip(forearm @ upper_axis, -1.0, 1.0)))

    rest = measure(0.0)
    sign = 1.0 if measure(0.1) > rest else -1.0
    return sign * (np.asarray(flexion, dtype=np.float64) - rest)


def _twist_angles_about_axis(rotvecs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Return signed swing-twist decomposition angles about a unit axis."""
    rotvecs = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / axis_norm

    quats = Rotation.from_rotvec(rotvecs).as_quat()
    vec = quats[:, :3]
    w = quats[:, 3]
    projected_vec = axis[np.newaxis, :] * (vec @ axis)[:, np.newaxis]
    twist_norm = np.sqrt(np.sum(projected_vec**2, axis=1) + w**2)
    safe_vec = np.divide(
        projected_vec,
        twist_norm[:, np.newaxis],
        out=np.zeros_like(projected_vec),
        where=twist_norm[:, np.newaxis] > 1e-12,
    )
    safe_w = np.divide(w, twist_norm, out=np.ones_like(w), where=twist_norm > 1e-12)
    signed_vec = safe_vec @ axis
    angles = 2.0 * np.arctan2(signed_vec, safe_w)
    return (angles + np.pi) % (2.0 * np.pi) - np.pi
