"""Convert SMPL body_pose parameters to 263-dim HumanML3D (HML263) features.

Pipeline
--------
1. Apply ``global_orient`` + ``transl`` to SMPL FK positions (``smpl_body_pose_to_positions``).
2. Compute the per-frame heading direction from the hip joint pair (joints 1 & 2).
3. Express skeleton in the heading-local frame (Y-axis rotation only, no
   pitch/roll).
4. Build all seven HML263 feature blocks.
5. Normalize with the HumanML3D ``Mean.npy`` / ``Std.npy`` statistics.

HML263 layout (per frame)
--------------------------
- ``[0:1]``     Root angular velocity (Y axis)
- ``[1:3]``     Root linear velocity (XZ, in local heading frame)
- ``[3:4]``     Root height above ground (Y)
- ``[4:67]``    RIC joint positions: 21 joints × 3, relative to root, in local frame
- ``[67:193]``  6D joint rotations: 21 joints × 6 (first two cols of rotation matrix)
- ``[193:259]`` Joint velocities: 22 joints × 3, world frame
- ``[259:263]`` Foot contacts: 4 binary flags (l_ankle, r_ankle, l_foot, r_foot)

Units
-----
All positions and translations are expected in the same unit as the SMPL model
(typically **metres** for SMPL-neutral from the smplx package).  The foot
contact thresholds should be tuned accordingly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    positions_to_smpl_body_pose,
    smpl_body_pose_to_positions,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

# SMPL 22-joint indices for foot contacts (l_ankle, r_ankle, l_foot, r_foot)
_FOOT_JOINT_INDICES = [7, 8, 10, 11]

# Minimum frames for a valid HumanML3D sequence at 20 fps
_MIN_FRAMES_AT_20FPS = 40


def smpl_params_to_positions(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """Convert SMPL parameters to world-space 22-joint positions.

    Applies ``global_orient`` as a root rotation and ``transl`` as the pelvis
    world position.

    Args:
        body_pose:     ``(N, 69)`` SMPL body_pose (23 joints × 3 axis-angle).
                       Only the first 21 joints (63 values) are used.
        global_orient: ``(N, 3)`` root orientation as axis-angle.
        transl:        ``(N, 3)`` root translation (pelvis world position).
        tpose_22:      ``(22, 3)`` T-pose joint positions.

    Returns:
        ``(N, 22, 3)`` world-space joint positions.
    """
    body_pose = np.asarray(body_pose, dtype=np.float64)
    global_orient = np.asarray(global_orient, dtype=np.float64)
    transl = np.asarray(transl, dtype=np.float64)
    tpose_22 = np.asarray(tpose_22, dtype=np.float64)

    n_frames = body_pose.shape[0]
    bp_21 = body_pose[:, :63].reshape(n_frames, 21, 3)  # drop hand joints 22-23

    positions_seq = np.empty((n_frames, 22, 3), dtype=np.float64)
    for t in range(n_frames):
        # FK in T-pose space (root at tpose_22[0])
        pos_local = smpl_body_pose_to_positions(bp_21[t], tpose_22)  # (22, 3)
        # Apply global_orient (rotate around pelvis) and translate
        root_rot = Rotation.from_rotvec(global_orient[t])
        positions_seq[t] = root_rot.apply(pos_local - tpose_22[0]) + transl[t]

    return positions_seq


def smpl_params_to_hml263(  # pylint: disable=too-many-locals,too-many-statements
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    tpose_22: np.ndarray | None = None,
    foot_height_thresh: float = 0.05,
    foot_vel_thresh: float = 0.05,
) -> np.ndarray:
    """Convert SMPL parameters to normalized HML263 feature vectors.

    Args:
        body_pose:          ``(N, 69)`` SMPL body_pose.
        global_orient:      ``(N, 3)``  root orientation axis-angle.
        transl:             ``(N, 3)``  root world translation.
        mean:               ``(263,)``  HumanML3D mean (from ``Mean.npy``).
        std:                ``(263,)``  HumanML3D std  (from ``Std.npy``).
        tpose_22:           ``(22, 3)`` T-pose joint positions.  Defaults to
                            :attr:`~uncertain_feedback.planners.mpc.kinematics\
.SmplLeftArmFK.tpose_all_joints`.
        foot_height_thresh: Height (in SMPL units) below which a foot joint is
                            considered in contact with the ground.
        foot_vel_thresh:    Per-frame displacement (in SMPL units) below which a
                            foot joint is considered stationary.

    Returns:
        ``(N, 263)`` normalized HML263 feature array, ``float32``.
    """
    body_pose = np.asarray(body_pose, dtype=np.float64)
    global_orient = np.asarray(global_orient, dtype=np.float64)
    transl = np.asarray(transl, dtype=np.float64)
    n_frames = body_pose.shape[0]

    if tpose_22 is None:
        tpose_22 = SmplLeftArmFK().tpose_all_joints  # (22, 3)
    tpose_22 = np.asarray(tpose_22, dtype=np.float64)

    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames, got {n_frames}.")
    if n_frames < _MIN_FRAMES_AT_20FPS:
        warnings.warn(
            f"Sequence has only {n_frames} frames (< {_MIN_FRAMES_AT_20FPS} recommended "
            "at 20 fps for HumanML3D)."
        )

    # ── Step 1: World-space 22-joint positions ────────────────────────────────
    positions = smpl_params_to_positions(body_pose, global_orient, transl, tpose_22)
    # positions: (N, 22, 3)

    # ── Step 2: Heading direction (Y-axis rotation per frame) ─────────────────
    # Facing direction: perpendicular to the left_hip→right_hip vector in XZ.
    # Joints: 1 = left_hip, 2 = right_hip.
    hip_line = positions[:, 1] - positions[:, 2]  # (N, 3)  l_hip - r_hip
    hip_line[:, 1] = 0.0  # project to XZ

    # 90° CCW in XZ: (x, z) → (-z, x) so that the facing direction is +Z when
    # the hips are aligned with +X.
    facing_x = -hip_line[:, 2]
    facing_z = hip_line[:, 0]
    norm = np.sqrt(facing_x**2 + facing_z**2) + 1e-8
    facing_x, facing_z = facing_x / norm, facing_z / norm

    # Heading angle: theta = atan2(facing_x, facing_z)
    # Convention: Ry(theta) @ [0,0,1] = [sin(theta), 0, cos(theta)] ≈ facing
    theta = np.arctan2(facing_x, facing_z)  # (N,)

    # ── Step 3: Angular velocity and accumulated heading ──────────────────────
    rot_vel = np.zeros(n_frames, dtype=np.float64)
    rot_vel[1:] = theta[1:] - theta[:-1]
    # Wrap angular differences to [-pi, pi]
    rot_vel[1:] = (rot_vel[1:] + np.pi) % (2.0 * np.pi) - np.pi
    # Accumulated heading relative to frame 0 (= 0 at t=0)
    heading = np.cumsum(rot_vel)  # (N,)

    # ── Step 4: Local-frame positions (vectorised) ────────────────────────────
    # Rotate world positions into the heading-local frame:
    #   Ry(-heading[t]) @ (pos[t] - root[t])
    # Ry(-a) @ [x,y,z] = [x*cos(a) - z*sin(a), y, x*sin(a) + z*cos(a)]
    cos_h = np.cos(heading)  # (N,)
    sin_h = np.sin(heading)  # (N,)

    rel = positions - positions[:, 0:1]  # (N, 22, 3)  root-relative
    local_x = cos_h[:, None] * rel[:, :, 0] - sin_h[:, None] * rel[:, :, 2]
    local_y = rel[:, :, 1]
    local_z = sin_h[:, None] * rel[:, :, 0] + cos_h[:, None] * rel[:, :, 2]
    local_positions = np.stack([local_x, local_y, local_z], axis=-1)  # (N, 22, 3)

    # ── Step 5: Ground height ─────────────────────────────────────────────────
    ground_y = float(np.min(positions[:, :, 1]))

    # ── Step 6: Root linear velocity in local frame ───────────────────────────
    root_vel_world = np.zeros((n_frames, 3), dtype=np.float64)
    root_vel_world[1:] = positions[1:, 0] - positions[:-1, 0]
    root_vel_world[:, 1] = 0.0  # XZ only
    # Rotate into local frame
    root_vel_local_x = cos_h * root_vel_world[:, 0] - sin_h * root_vel_world[:, 2]
    root_vel_local_z = sin_h * root_vel_world[:, 0] + cos_h * root_vel_world[:, 2]

    # ── Step 7: Joint velocities (world frame) ────────────────────────────────
    joint_vel = np.zeros((n_frames, 22, 3), dtype=np.float64)
    joint_vel[1:] = positions[1:] - positions[:-1]

    # ── Step 8: Foot contacts ─────────────────────────────────────────────────
    foot_contacts = np.zeros((n_frames, 4), dtype=np.float32)
    for fi, ji in enumerate(_FOOT_JOINT_INDICES):
        height = positions[:, ji, 1] - ground_y  # (N,)
        speed = np.linalg.norm(joint_vel[:, ji], axis=-1)  # (N,)
        foot_contacts[:, fi] = (
            (height < foot_height_thresh) & (speed < foot_vel_thresh)
        ).astype(np.float32)

    # ── Step 9: 6D rotations via IK on local positions ───────────────────────
    rotations_6d = np.zeros((n_frames, 21, 6), dtype=np.float64)
    bp_21 = body_pose[:, :63].reshape(n_frames, 21, 3)  # (N, 21, 3)
    del bp_21  # unused — we re-derive from IK on local positions
    for t in range(n_frames):
        bp_local = positions_to_smpl_body_pose(local_positions[t], tpose_22)  # (21, 3)
        for j in range(21):
            mat = Rotation.from_rotvec(bp_local[j]).as_matrix()  # (3, 3)
            rotations_6d[t, j] = np.concatenate([mat[:, 0], mat[:, 1]])

    # ── Assemble HML263 ───────────────────────────────────────────────────────
    features = np.zeros((n_frames, 263), dtype=np.float32)
    features[:, 0] = rot_vel
    features[:, 1] = root_vel_local_x
    features[:, 2] = root_vel_local_z
    features[:, 3] = positions[:, 0, 1] - ground_y
    features[:, 4:67] = local_positions[:, 1:].reshape(n_frames, 63)
    features[:, 67:193] = rotations_6d.reshape(n_frames, 126)
    features[:, 193:259] = joint_vel.reshape(n_frames, 66)
    features[:, 259:263] = foot_contacts

    # ── Normalize ─────────────────────────────────────────────────────────────
    features = (features - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-8)
    return features


def positions_to_hml263(  # pylint: disable=too-many-locals,too-many-statements
    positions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    tpose_22: np.ndarray | None = None,
    foot_height_thresh: float = 0.05,
    foot_vel_thresh: float = 0.05,
) -> np.ndarray:
    """Convert world-space 22-joint positions directly to normalized HML263 features.

    This is the positions-first variant of :func:`smpl_params_to_hml263`, skipping
    the SMPL FK step.  Use it when joint positions come from direct keypoint mapping
    (e.g. MHR-70 → SMPL-22) rather than from SMPL body_pose parameters.

    Args:
        positions:          ``(N, 22, 3)`` world-space SMPL-22 joint positions.
        mean:               ``(263,)`` HumanML3D mean (from ``Mean.npy``).
        std:                ``(263,)`` HumanML3D std  (from ``Std.npy``).
        tpose_22:           ``(22, 3)`` T-pose joint positions for IK (6D rotations).
                            Defaults to :attr:`SmplLeftArmFK.tpose_all_joints`.
        foot_height_thresh: Height (SMPL units) below which a foot joint is
                            considered in contact with the ground.
        foot_vel_thresh:    Per-frame displacement below which a foot is stationary.

    Returns:
        ``(N, 263)`` normalized HML263 feature array, ``float32``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_frames = positions.shape[0]

    if tpose_22 is None:
        tpose_22 = SmplLeftArmFK().tpose_all_joints
    tpose_22 = np.asarray(tpose_22, dtype=np.float64)

    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames, got {n_frames}.")
    if n_frames < _MIN_FRAMES_AT_20FPS:
        warnings.warn(
            f"Sequence has only {n_frames} frames (< {_MIN_FRAMES_AT_20FPS} recommended "
            "at 20 fps for HumanML3D)."
        )

    # ── Heading direction ─────────────────────────────────────────────────────
    hip_line = positions[:, 1] - positions[:, 2]  # l_hip - r_hip
    hip_line[:, 1] = 0.0
    facing_x = -hip_line[:, 2]
    facing_z = hip_line[:, 0]
    norm = np.sqrt(facing_x**2 + facing_z**2) + 1e-8
    facing_x, facing_z = facing_x / norm, facing_z / norm
    theta = np.arctan2(facing_x, facing_z)

    # ── Angular velocity and accumulated heading ──────────────────────────────
    rot_vel = np.zeros(n_frames, dtype=np.float64)
    rot_vel[1:] = theta[1:] - theta[:-1]
    rot_vel[1:] = (rot_vel[1:] + np.pi) % (2.0 * np.pi) - np.pi
    heading = np.cumsum(rot_vel)

    # ── Local-frame positions ─────────────────────────────────────────────────
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rel = positions - positions[:, 0:1]
    local_x = cos_h[:, None] * rel[:, :, 0] - sin_h[:, None] * rel[:, :, 2]
    local_y = rel[:, :, 1]
    local_z = sin_h[:, None] * rel[:, :, 0] + cos_h[:, None] * rel[:, :, 2]
    local_positions = np.stack([local_x, local_y, local_z], axis=-1)

    # ── Ground height ─────────────────────────────────────────────────────────
    ground_y = float(np.min(positions[:, :, 1]))

    # ── Root linear velocity in local frame ───────────────────────────────────
    root_vel_world = np.zeros((n_frames, 3), dtype=np.float64)
    root_vel_world[1:] = positions[1:, 0] - positions[:-1, 0]
    root_vel_world[:, 1] = 0.0
    root_vel_local_x = cos_h * root_vel_world[:, 0] - sin_h * root_vel_world[:, 2]
    root_vel_local_z = sin_h * root_vel_world[:, 0] + cos_h * root_vel_world[:, 2]

    # ── Joint velocities (world frame) ────────────────────────────────────────
    joint_vel = np.zeros((n_frames, 22, 3), dtype=np.float64)
    joint_vel[1:] = positions[1:] - positions[:-1]

    # ── Foot contacts ─────────────────────────────────────────────────────────
    foot_contacts = np.zeros((n_frames, 4), dtype=np.float32)
    for fi, ji in enumerate(_FOOT_JOINT_INDICES):
        height = positions[:, ji, 1] - ground_y
        speed = np.linalg.norm(joint_vel[:, ji], axis=-1)
        foot_contacts[:, fi] = (
            (height < foot_height_thresh) & (speed < foot_vel_thresh)
        ).astype(np.float32)

    # ── 6D rotations via IK on local positions ────────────────────────────────
    rotations_6d = np.zeros((n_frames, 21, 6), dtype=np.float64)
    for t in range(n_frames):
        bp_local = positions_to_smpl_body_pose(local_positions[t], tpose_22)
        for j in range(21):
            mat = Rotation.from_rotvec(bp_local[j]).as_matrix()
            rotations_6d[t, j] = np.concatenate([mat[:, 0], mat[:, 1]])

    # ── Assemble and normalize ────────────────────────────────────────────────
    features = np.zeros((n_frames, 263), dtype=np.float32)
    features[:, 0] = rot_vel
    features[:, 1] = root_vel_local_x
    features[:, 2] = root_vel_local_z
    features[:, 3] = positions[:, 0, 1] - ground_y
    features[:, 4:67] = local_positions[:, 1:].reshape(n_frames, 63)
    features[:, 67:193] = rotations_6d.reshape(n_frames, 126)
    features[:, 193:259] = joint_vel.reshape(n_frames, 66)
    features[:, 259:263] = foot_contacts

    features = (features - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-8)
    return features


def load_hml_stats(stats_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HumanML3D normalization statistics.

    Args:
        stats_dir: Directory containing ``Mean.npy`` and ``Std.npy``.

    Returns:
        ``(mean, std)`` each of shape ``(263,)``.
    """
    stats_dir = Path(stats_dir)
    mean = np.load(stats_dir / "Mean.npy")  # (263,)
    std = np.load(stats_dir / "Std.npy")  # (263,)
    return mean.astype(np.float32), std.astype(np.float32)
