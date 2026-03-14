"""HML263 ↔ SMPL body_pose conversion utilities.

HML → SMPL  (MDM output decoding):

    ``hml263_to_smpl_body_pose``
        Full pipeline: MDM de-normalization + ``recover_from_ric`` + ``rot2xyz``
        (all provided by the MDM repo) followed by minimum-rotation IK.

    ``positions_to_smpl_body_pose``
        Core IK: ``(22, 3)`` global XYZ positions → ``(21, 3)`` SMPL local
        axis-angle body_pose.

    ``smpl_body_pose_to_arm_aa``
        Extract the 4 left-arm joints from a ``(21, 3)`` body_pose array.

SMPL → HML  (inpainting / input conditioning):

    ``smpl_arm_aa_to_hml263_frame``
        Patch left-arm axis-angles back into a normalized HML263 frame.  Used
        to condition the MDM inpainting start frame on the current MPC state.

    ``HmlArmFeatureInfo``
        Precomputed HML263 feature offsets for the arm joints; bundled into
        one dataclass so callers do not need to track four separate lists.

FK utility:

    ``smpl_body_pose_to_positions``
        Full-body FK: ``(21, 3)`` SMPL body_pose → ``(22, 3)`` global XYZ.
        Distinct from :class:`~uncertain_feedback.planners.mpc.kinematics.SmplLeftArmFK`
        which only controls the 4 left-arm joints.

Conversion pipeline (HML → SMPL)::

    (n_frames, 263)  normalized HumanML3D
        → inv_transform + recover_from_ric + rot2xyz   [MDM-provided]
        → (n_frames, 22, 3)  global XYZ joint positions
        → positions_to_smpl_body_pose  [custom IK, per frame]
        → (n_frames, 21, 3)  SMPL body_pose local axis-angles
        → smpl_body_pose_to_arm_aa  [select 4 arm joints]
        → (n_frames, 4, 3)  [left_collar, left_shoulder, left_elbow, left_wrist]

Inverse pipeline (SMPL → HML, for inpainting)::

    (4, 3)  left-arm axis-angles
        → rotation matrix → 6D representation       [patch 6D rotation block]
        → collar world rot from base 6D chain        [spine1→spine2→spine3→collar]
        → arm FK relative to actual collar pose      [patch RIC position block]
        → zero velocity features
        → re-normalize
        → (263,)  patched normalized HML263 frame
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Make uncertain_feedback importable when the file is run directly.
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.planners.mpc.kinematics import (  # pylint: disable=wrong-import-position
    SMPL_PARENTS_22,
    SmplLeftArmFK,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indices into SMPL body_pose (shape (21, 3)) for the left arm joints.
# SMPL joint j maps to body_pose row j-1 (root joint has no body_pose entry).
#   left_collar   → joint 13 → body_pose[12]
#   left_shoulder → joint 16 → body_pose[15]
#   left_elbow    → joint 18 → body_pose[17]
#   left_wrist    → joint 20 → body_pose[19]
ARM_BODY_POSE_INDICES: list[int] = [12, 15, 17, 19]


# ---------------------------------------------------------------------------
# SMPL → HML: feature-offset bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class HmlArmFeatureInfo:
    """Precomputed HML263 feature offsets for the left arm joints.

    HML263 block layout::

        [0:4]     root (rot_vel, lin_vel_x, lin_vel_z, root_y)
        [4:67]    RIC positions  (21 joints × 3)
        [67:193]  6D rotations   (21 joints × 6)
        [193:259] velocities     (22 joints × 3)
        [259:263] foot contacts  (4)

    Attributes:
        l_arm_joints:   HML joint indices for ``[left_shoulder, left_elbow,
                        left_wrist]`` (indices into the 22-joint array;
                        **excludes** left_collar which has no HML rotation
                        feature).
        arm_6d_offsets: Starting offsets into the 6D rotation block for each
                        arm joint.
        arm_vel_offsets: Starting offsets into the velocity block.
    """

    l_arm_joints: list[int]
    arm_6d_offsets: list[int]
    arm_vel_offsets: list[int]


# ---------------------------------------------------------------------------
# SMPL → HML: patch arm axis-angles into a normalized HML263 frame
# ---------------------------------------------------------------------------


def smpl_arm_aa_to_hml263_frame(  # pylint: disable=too-many-locals
    base_norm: np.ndarray,
    arm_aa: np.ndarray,
    arm_info: HmlArmFeatureInfo,
    hml_mean: np.ndarray,
    hml_std: np.ndarray,
    fk: SmplLeftArmFK,
) -> np.ndarray:
    """Patch left-arm axis-angles into a normalized HML263 frame.

    Replaces the 6D rotation features, RIC position features, and velocity
    features for the arm joints (shoulder, elbow, wrist) in a normalized
    HML263 vector with values derived from ``arm_aa``.

    RIC positions are computed by accumulating the collar's world rotation
    from the base pose's own HML263 6D features (spine1 → spine2 → spine3 →
    collar chain), then applying FK along the arm chain.  This avoids the
    T-pose spine assumption of ``SmplLeftArmFK.full_body_positions``.

    This is used to condition MDM's inpainting start frame on the current MPC
    arm state so that the generated motion starts from the actual arm
    configuration rather than the fixed sitting pose.

    Args:
        base_norm:  ``(263,)`` normalized HML263 frame to use as the base
                    (e.g. the sitting pose).
        arm_aa:     ``(4, 3)`` left-arm axis-angles for
                    ``[left_collar, left_shoulder, left_elbow, left_wrist]``.
        arm_info:   Precomputed feature offsets from
                    :class:`HmlArmFeatureInfo`.
        hml_mean:   ``(263,)`` HML263 normalization mean.
        hml_std:    ``(263,)`` HML263 normalization std.
        fk:         :class:`~uncertain_feedback.planners.mpc.kinematics.SmplLeftArmFK`
                    instance used to access T-pose bone lengths.

    Returns:
        ``(263,)`` patched and re-normalized HML263 frame.
    """
    arm_aa = np.asarray(arm_aa, dtype=np.float64)

    # Un-normalize the base frame to raw HML features.
    raw = base_norm * hml_std + hml_mean  # (263,)
    tpose_22 = fk.tpose_all_joints  # (22, 3)

    # Derive collar / spine3 joint indices from SMPL topology — no magic numbers.
    collar_j = SMPL_PARENTS_22[arm_info.l_arm_joints[0]]  # parent of shoulder = 13
    spine3_j = SMPL_PARENTS_22[collar_j]  # parent of collar  =  9

    # --- Patch 6D rotation features: collar + shoulder + elbow + wrist ------
    # arm_aa: [0=collar, 1=shoulder, 2=elbow, 3=wrist].
    # Offsets computed from joint indices (no magic numbers).
    for arm_idx, joint_j in enumerate([collar_j] + list(arm_info.l_arm_joints)):
        rot_mat = Rotation.from_rotvec(arm_aa[arm_idx]).as_matrix()
        r6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])  # first two columns → (6,)
        raw[67 + (joint_j - 1) * 6 : 67 + joint_j * 6] = r6d

    # --- Patch RIC position features for collar + shoulder + elbow + wrist --
    # Build spine chain from root up to (and including) spine3.  Walk upward
    # from spine3, then reverse.  The spine is constrained to the base pose
    # by inpainting, so reading its HML263 6D rotations is correct.
    spine_chain: list[int] = []
    j = spine3_j
    while j > 0:
        spine_chain.append(j)
        j = SMPL_PARENTS_22[j]
    spine_chain.reverse()  # e.g. [3, 6, 9]

    # Accumulate spine3 world rotation from the base HML263 (collar excluded).
    spine3_world_rot = Rotation.identity()
    for j in spine_chain:
        r6d_j = raw[67 + (j - 1) * 6 : 67 + j * 6]
        a1, a2 = r6d_j[:3], r6d_j[3:6]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(a2, b1) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        spine3_world_rot = spine3_world_rot * Rotation.from_matrix(
            np.stack([b1, b2, b3], axis=1)
        )

    # Collar RIC position: spine3 (base pose) + FK using spine3's world rotation
    # (the PARENT's rotation places the child; the child's own rotation only
    # affects where its children end up).
    spine3_ric = raw[4 + (spine3_j - 1) * 3 : 4 + spine3_j * 3].copy()
    collar_ric = spine3_ric + spine3_world_rot.apply(
        tpose_22[collar_j] - tpose_22[spine3_j]
    )
    raw[4 + (collar_j - 1) * 3 : 4 + collar_j * 3] = collar_ric

    # Collar world rotation from arm_aa[0] (not the base pose's collar 6D).
    collar_world_rot = spine3_world_rot * Rotation.from_rotvec(arm_aa[0])

    # Walk collar → shoulder → elbow → wrist.
    # FK rule: child_pos = parent_pos + parent_world_rot.apply(tpose_bone)
    arm_chain_smpl = [collar_j] + list(arm_info.l_arm_joints)
    current_rot = collar_world_rot
    current_ric = collar_ric
    for arm_idx, (parent_j, child_j) in enumerate(
        zip(arm_chain_smpl[:-1], arm_chain_smpl[1:]), start=1
    ):
        tpose_bone = tpose_22[child_j] - tpose_22[parent_j]
        child_ric = current_ric + current_rot.apply(
            tpose_bone
        )  # parent rot → child pos
        child_rot = current_rot * Rotation.from_rotvec(
            arm_aa[arm_idx]
        )  # child world rot
        raw[4 + (child_j - 1) * 3 : 4 + child_j * 3] = child_ric
        current_rot = child_rot
        current_ric = child_ric

    # --- Zero velocity features for collar + shoulder + elbow + wrist -------
    # For a static start pose (the same frame repeated 10×), inter-frame
    # velocity should be zero.  The base frame may carry non-zero velocities
    # from its original motion clip, which would be inconsistent with the
    # patched position/rotation features.
    raw[193 + collar_j * 3 : 193 + collar_j * 3 + 3] = 0.0
    for vel_offset in arm_info.arm_vel_offsets:
        raw[vel_offset : vel_offset + 3] = 0.0

    # Re-normalize and return.
    return (raw - hml_mean) / (hml_std + 1e-8)


# ---------------------------------------------------------------------------
# Core IK: global XYZ positions → SMPL body_pose
# ---------------------------------------------------------------------------


def positions_to_smpl_body_pose(
    positions: np.ndarray,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """Convert global joint XYZ positions to SMPL local axis-angle body_pose.

    Runs a minimum-rotation IK over the 22-joint SMPL skeleton.  For each
    joint the world rotation is found via ``Rotation.align_vectors``, which
    chooses the shortest rotation that maps the T-pose bone direction to the
    actual bone direction.  The local rotation is then extracted by undoing
    the accumulated parent rotation.

    The root joint (pelvis) is assumed to have identity world orientation.
    Joints are processed in index order 1-21, which is valid because SMPL
    guarantees ``parent(j) < j`` for all non-root joints.

    Args:
        positions: ``(22, 3)`` global XYZ coordinates for all 22 SMPL joints.
        tpose_22:  ``(22, 3)`` T-pose joint positions (from
                   :attr:`SmplLeftArmFK.tpose_all_joints`).

    Returns:
        ``(21, 3)`` array of local axis-angle rotations — SMPL body_pose
        format.  Row ``j-1`` is the rotation for SMPL joint ``j``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    tpose_22 = np.asarray(tpose_22, dtype=np.float64)

    world_rots: list[Rotation] = [Rotation.identity()] * 22
    body_pose = np.zeros((21, 3), dtype=np.float64)

    for j in range(1, 22):
        p = SMPL_PARENTS_22[j]
        actual_bone = positions[j] - positions[p]
        tpose_bone = tpose_22[j] - tpose_22[p]

        bone_len = np.linalg.norm(actual_bone)
        if bone_len < 1e-8:
            # Degenerate bone: keep local rotation as identity (zero aa).
            world_rots[j] = world_rots[p]
            continue

        # Minimum rotation that maps the T-pose bone direction to the actual
        # bone direction.  align_vectors(target, source) returns R such that
        # R.apply(source) ≈ target.
        world_rot_j, _ = Rotation.align_vectors(
            [actual_bone],
            [tpose_bone],
        )
        local_rot = world_rots[p].inv() * world_rot_j
        body_pose[j - 1] = local_rot.as_rotvec()
        world_rots[j] = world_rot_j

    return body_pose


# ---------------------------------------------------------------------------
# Full-body FK: SMPL body_pose → global XYZ positions
# ---------------------------------------------------------------------------


def smpl_body_pose_to_positions(
    body_pose: np.ndarray,
    tpose_22: np.ndarray,
    root_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Convert SMPL local axis-angle body_pose to global joint XYZ positions.

    This is the forward-kinematics complement to
    :func:`positions_to_smpl_body_pose`.  It uses the same convention as
    :class:`~uncertain_feedback.planners.mpc.kinematics.SmplLeftArmFK`:
    ``world_rot[j] = world_rot[parent(j)] * local_rot[j]``.

    Unlike :meth:`SmplLeftArmFK.full_body_positions`, which controls only the
    4 left-arm joints, this function accepts a full ``(21, 3)`` body_pose and
    articulates all 22 joints.

    Args:
        body_pose: ``(21, 3)`` local axis-angle rotations (SMPL body_pose).
                   Row ``j-1`` is the rotation for SMPL joint ``j``.
        tpose_22:  ``(22, 3)`` T-pose joint positions.
        root_pos:  ``(3,)`` world position of the root (pelvis).  Defaults
                   to ``tpose_22[0]`` (T-pose pelvis).

    Returns:
        ``(22, 3)`` global XYZ positions for all 22 SMPL joints.
    """
    body_pose = np.asarray(body_pose, dtype=np.float64)
    tpose_22 = np.asarray(tpose_22, dtype=np.float64)

    positions = np.empty((22, 3), dtype=np.float64)
    positions[0] = (
        np.asarray(root_pos, dtype=np.float64) if root_pos is not None else tpose_22[0]
    )

    world_rots: list[Rotation] = [Rotation.identity()] * 22

    for j in range(1, 22):
        p = SMPL_PARENTS_22[j]
        local_rot = Rotation.from_rotvec(body_pose[j - 1])
        world_rots[j] = world_rots[p] * local_rot
        tpose_bone = tpose_22[j] - tpose_22[p]
        positions[j] = positions[p] + world_rots[j].apply(tpose_bone)

    return positions


# ---------------------------------------------------------------------------
# Arm extraction
# ---------------------------------------------------------------------------


def smpl_body_pose_to_arm_aa(body_pose: np.ndarray) -> np.ndarray:
    """Extract left arm axis-angles from SMPL body_pose.

    Args:
        body_pose: ``(..., 21, 3)`` SMPL body_pose.

    Returns:
        ``(..., 4, 3)`` axis-angles for
        ``[left_collar, left_shoulder, left_elbow, left_wrist]``.
    """
    return np.asarray(body_pose)[..., ARM_BODY_POSE_INDICES, :]


# ---------------------------------------------------------------------------
# Full pipeline: HumanML3D 263-dim → SMPL body_pose
# ---------------------------------------------------------------------------


def hml263_to_smpl_body_pose(
    hml_vec: "torch.Tensor",  # type: ignore[name-defined]  # noqa: F821
    dataset,
    model,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """Convert normalized HumanML3D vectors to SMPL body_pose.

    This function chains the MDM de-normalization and reconstruction pipeline
    (``inv_transform`` → ``recover_from_ric`` → ``rot2xyz``) with the
    minimum-rotation IK provided by :func:`positions_to_smpl_body_pose`.

    Args:
        hml_vec:  ``(n_frames, 263)`` normalized HumanML3D motion tensor.
        dataset:  HumanML3D ``DataLoader`` returned by ``get_dataset_loader``
                  — accesses ``dataset.dataset.t2m_dataset.inv_transform``.
        model:    MDM model — must expose ``model.rot2xyz``.
        tpose_22: ``(22, 3)`` T-pose positions, e.g. from
                  :attr:`SmplLeftArmFK.tpose_all_joints`.

    Returns:
        ``(n_frames, 21, 3)`` SMPL body_pose axis-angles.
    """
    # Import here so the module remains importable without MDM on sys.path.
    # pylint: disable=import-outside-toplevel,import-error
    import torch
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    # inv_transform uses CPU numpy arrays internally; keep tensor on CPU.
    hml_vec = hml_vec.float().cpu()
    n_frames = hml_vec.shape[0]

    # --- De-normalize -------------------------------------------------------
    # inv_transform expects (batch, nfeats=1, seq_len, 263).
    # dataset is a DataLoader; the underlying Dataset is at dataset.dataset.
    unnorm = dataset.dataset.t2m_dataset.inv_transform(
        hml_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, n_frames, 263)
    ).float()

    # --- Recover global XYZ positions ---------------------------------------
    # recover_from_ric: (1, 1, n_frames, 263) → (1, 1, n_frames, 22, 3)
    ric = recover_from_ric(unnorm, 22)
    # (1, 22, 3, n_frames)
    ric = ric.view(-1, *ric.shape[2:]).permute(0, 2, 3, 1)

    # rot2xyz applies the full SMPL FK and returns absolute world positions.
    xyz = model.rot2xyz(
        x=ric,
        mask=None,
        pose_rep="xyz",
        glob=True,
        translation=True,
        jointstype="smpl",
        vertstrans=True,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
    )
    # xyz: (1, 22, 3, n_frames) → (n_frames, 22, 3)
    positions_seq: np.ndarray = xyz[0].permute(2, 0, 1).cpu().numpy()

    # --- IK: positions → SMPL body_pose ------------------------------------
    body_pose = np.stack(
        [
            positions_to_smpl_body_pose(positions_seq[t], tpose_22)
            for t in range(n_frames)
        ],
        axis=0,
    )  # (n_frames, 21, 3)

    return body_pose
