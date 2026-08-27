"""HML263 ↔ SMPL body_pose conversion utilities.

HML263 files/model outputs follow the **official** HumanML3D encoding
(``process_file``): the 6D rotation block stores local quaternions relative to
the t2m reference skeleton (arms hanging down), NOT this repo's SMPL T-pose
(arms out).  All decoding therefore goes through the RIC positions —
convention-free — followed by minimum-rotation IK into repo body_pose; the 6D
block is never read directly.  Exception: ``dataset/custom1`` and the
checkpoints fine-tuned on it (including ``save/customv3_fixed``, the default
until 2026-08-09) predate the 2026-07-07 switch to ``process_file`` — see
:func:`smpl_arm_aa_to_hml263_frame`.

HML → SMPL  (MDM output decoding):

    ``hml263_to_smpl_body_pose``
        Full pipeline: MDM de-normalization + ``recover_from_ric`` (MDM repo)
        followed by minimum-rotation IK.

    ``hml263_batch_to_smpl_body_pose``
        Batched variant: recovers positions for all ``n_samples`` trajectories
        at once, then parallelises the per-frame IK across samples.

    ``positions_to_smpl_body_pose``
        Core IK: ``(22, 3)`` global XYZ positions → ``(21, 3)`` SMPL local
        axis-angle body_pose.

    ``smpl_body_pose_to_arm_aa``
        Extract the 3 MPC-controlled left-arm joints from a ``(21, 3)``
        body_pose array.

SMPL → HML  (inpainting / input conditioning):

    ``smpl_arm_aa_seq_to_hml263_frames``
        Patch a *sequence* of left-arm axis-angles into K normalized HML263
        frames encoded as one clip, so the velocity channels describe the
        actual arm motion across the prefix.

    ``smpl_arm_aa_to_hml263_frame``
        Single-frame case (``K = 1``) of the above: a static pinned frame whose
        velocity channels are ~0.  Used to condition the MDM inpainting start
        frame on the current MPC state.

FK utility:

    ``smpl_body_pose_to_positions``
        Full-body FK: ``(21, 3)`` SMPL body_pose → ``(22, 3)`` global XYZ.
        Distinct from :class:`~uncertain_feedback.planners.mpc.kinematics.SmplLeftArmFK`
        which focuses on the left-arm chain.

Conversion pipeline (HML → SMPL)::

    (n_frames, 263)  normalized HumanML3D
        → inv_transform + recover_from_ric             [MDM-provided]
        → (n_frames, 22, 3)  global XYZ joint positions
        → positions_to_smpl_body_pose  [custom IK, per frame]
        → (n_frames, 21, 3)  SMPL body_pose local axis-angles
        → smpl_body_pose_to_arm_aa  [select controlled arm joints]
        → (n_frames, 3, 3)  [left_shoulder, left_elbow, left_wrist]

Inverse pipeline (SMPL → HML, for inpainting)::

    (3, 3)  controlled left-arm axis-angles
        → recover_from_ric on the base frame → (22, 3) positions
        → minimum-rotation IK → repo body_pose, arm slots ← arm_aa
        → full-body FK → patched (22, 3) positions
        → official HumanML3D ``positions_to_hml263`` re-encode
        → (263,)  patched normalized HML263 frame
"""

from __future__ import annotations

import sys
import time
from os import cpu_count
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

# Make uncertain_feedback importable when the file is run directly.
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.planners.mpc.kinematics import (  # pylint: disable=wrong-import-position
    SMPL_PARENTS_22,
    SmplLeftArmFK,
    anatomical_elbow_wrist_slots,
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
COLLAR_BODY_POSE_INDEX: int = 12
ARM_BODY_POSE_INDICES: list[int] = [15, 17, 19]


# ---------------------------------------------------------------------------
# SMPL → HML: patch arm axis-angles into a normalized HML263 frame
# ---------------------------------------------------------------------------


def smpl_arm_aa_seq_to_hml263_frames(
    base_norm: np.ndarray,
    arm_aa_seq: np.ndarray,
    hml_mean: np.ndarray,
    hml_std: np.ndarray,
    fk: SmplLeftArmFK,
) -> np.ndarray:
    """Patch a sequence of left-arm axis-angles into K normalized HML263 frames.

    Same pipeline as :func:`smpl_arm_aa_to_hml263_frame` but for a whole
    inpainting prefix: the base frame is decoded and IK'd **once**, each of the
    ``K`` arm configurations is patched into that body_pose and FK'd, and the
    resulting ``(K, 22, 3)`` position clip is encoded with a **single**
    ``positions_to_hml263`` call.

    The single call is the load-bearing detail.  ``process_file`` computes
    velocities, root features and foot contacts by finite-differencing *within*
    the clip it is given, so encoding the K frames together yields a prefix
    whose velocity channels (``[0:3]``, ``[193:259]``) describe the actual arm
    motion.  Encoding each frame separately and stacking would instead give K
    frames that each say "not moving".

    Root and body joints are static across the prefix — they come from the base
    frame — so only the arm chain carries velocity.  The final returned frame is
    the one duplicated to satisfy ``process_file``'s ``N → N-1`` drop, so its
    own velocity channels are 0; it is the frame the generated motion continues
    from.

    Note that for ``K > 1`` that final frame is **not** bit-identical to
    ``smpl_arm_aa_to_hml263_frame`` applied to the same last arm config.
    ``process_file`` derives the canonical heading from hips + shoulders and
    Gaussian-smooths it along the clip (``sigma = 20`` frames), so a prefix with
    ``K << 20`` gets one clip-averaged heading for all K frames while the
    single-frame pin gets that pose's own heading — and that heading depends on
    the arm configuration.  The two therefore differ by a pure yaw about the
    root: 0.13 rad (0.08 m at the wrist) for a 0.5 rad shoulder sweep.  Root
    position and the body pose itself are unchanged.  The shared heading is what
    keeps the encoded velocities pure arm motion rather than spurious body yaw,
    so comparisons against the single-frame pin belong on decoded positions
    after yaw alignment, not on raw channels.

    Args:
        base_norm:   ``(263,)`` normalized HML263 frame to use as the base.
        arm_aa_seq:  ``(K, 3, 3)`` left-arm axis-angles per prefix frame,
                     oldest → newest, for
                     ``[left_shoulder, left_elbow, left_wrist]``.
        hml_mean:    ``(263,)`` HML263 normalization mean.
        hml_std:     ``(263,)`` HML263 normalization std.
        fk:          :class:`~uncertain_feedback.planners.mpc.kinematics.SmplLeftArmFK`
                     instance used to access T-pose bone lengths.

    Returns:
        ``(K, 263)`` patched and re-normalized HML263 frames.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    from uncertain_feedback.data_collection.common.hml263 import (
        positions_to_hml263,
    )

    recover_from_ric = _import_recover_from_ric()

    arm_aa_seq = np.asarray(arm_aa_seq, dtype=np.float64)
    raw = base_norm * hml_std + hml_mean  # (263,)
    base_positions = (
        recover_from_ric(torch.tensor(raw, dtype=torch.float32).unsqueeze(0), 22)[0]
        .numpy()
        .astype(np.float64)
    )  # (22, 3)

    body_pose = positions_to_smpl_body_pose(base_positions, fk.tpose_all_joints)
    patched_positions = []
    for arm_aa in arm_aa_seq:
        body_pose[ARM_BODY_POSE_INDICES] = arm_aa
        patched_positions.append(
            smpl_body_pose_to_positions(
                body_pose, fk.tpose_all_joints, root_pos=base_positions[0]
            )
        )

    # process_file drops the last frame, so duplicate the newest one.
    frames = np.stack(patched_positions + patched_positions[-1:])  # (K + 1, 22, 3)
    return positions_to_hml263(frames, hml_mean, hml_std).astype(np.float64)


def smpl_arm_aa_to_hml263_frame(
    base_norm: np.ndarray,
    arm_aa: np.ndarray,
    hml_mean: np.ndarray,
    hml_std: np.ndarray,
    fk: SmplLeftArmFK,
) -> np.ndarray:
    """Patch left-arm axis-angles into a normalized HML263 frame.

    Decodes the base frame to global positions (``recover_from_ric``), swaps
    the controlled arm slots into the repo body_pose obtained by
    minimum-rotation IK, runs full-body FK, and re-encodes the patched frame
    with the official HumanML3D ``process_file``.

    This is used to condition MDM's inpainting start frame on the current MPC
    arm state so that the generated motion starts from the actual arm
    configuration rather than the fixed sitting pose.  It is the ``K = 1`` case
    of :func:`smpl_arm_aa_seq_to_hml263_frames`.

    **Encoding match (2026-08-09).** The re-encode makes the output
    official-convention regardless of what ``base_norm`` was encoded with, so
    the frame is only on the *model's* training manifold if that model was
    fine-tuned on official-encoding data.  **The current default checkpoint
    satisfies this**: ``custom_seatedcanon_lr1e7_10k/model000759250.pt``
    (``consts.py``) was fine-tuned on ``dataset/custom1_seatedcanon``, where
    patching a frame's own arm angles back in moves its rot6d arm slots by only
    0.0002 normalized RMS (0.0006 max).  The previous default
    ``save/customv3_fixed`` was fine-tuned on ``dataset/custom1``, which
    predates the 2026-07-07 ``process_file`` switch: the same measurement gives
    1.98 RMS (4.34 max), i.e. that checkpoint could not read the arm rotations
    it was conditioned on.  See ``CODEBASE_MAP.md`` §9 "Dataset encoding
    provenance".

    The output encodes a duplicated static pose, so its velocity blocks
    (``[0:3]``, ``[193:259]``) are ~0, its foot contacts read "in contact", and
    floor grounding is recomputed on the single pose (a constant RIC-Y shift
    relative to a frame grounded on its full clip).

    Args:
        base_norm:  ``(263,)`` normalized HML263 frame to use as the base
                    (e.g. the sitting pose).
        arm_aa:     ``(3, 3)`` left-arm axis-angles for
                    ``[left_shoulder, left_elbow, left_wrist]``.
        hml_mean:   ``(263,)`` HML263 normalization mean.
        hml_std:    ``(263,)`` HML263 normalization std.
        fk:         :class:`~uncertain_feedback.planners.mpc.kinematics.SmplLeftArmFK`
                    instance used to access T-pose bone lengths.

    Returns:
        ``(263,)`` patched and re-normalized HML263 frame.
    """
    arm_aa = np.asarray(arm_aa, dtype=np.float64)
    return smpl_arm_aa_seq_to_hml263_frames(
        base_norm, arm_aa[None], hml_mean, hml_std, fk
    )[0]


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

    # Anatomical override for the left arm: reparameterize the elbow(18) and
    # wrist(20) slots so the elbow carries the recovered shoulder rotation and
    # the wrist becomes a pure hinge (zero pronation).  Positions are preserved
    # because both arm bone directions are reproduced.  See
    # ``.claude/POSE_REPRESENTATION_AUDIT.md``.
    upper_len = np.linalg.norm(positions[18] - positions[16])
    forearm_len = np.linalg.norm(positions[20] - positions[18])
    if upper_len >= 1e-8 and forearm_len >= 1e-8:
        elbow_aa, wrist_aa = anatomical_elbow_wrist_slots(
            positions[16],
            positions[18],
            positions[20],
            world_rots[16],
            tpose_22[18] - tpose_22[16],
            tpose_22[20] - tpose_22[18],
        )
        body_pose[17] = elbow_aa
        body_pose[19] = wrist_aa

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
        ``(..., 3, 3)`` axis-angles for
        ``[left_shoulder, left_elbow, left_wrist]``.
    """
    return np.asarray(body_pose)[..., ARM_BODY_POSE_INDICES, :]


def smpl_body_pose_to_collar_aa(body_pose: np.ndarray) -> np.ndarray:
    """Extract the fixed left-collar axis-angle from SMPL body_pose."""
    return np.asarray(body_pose)[..., COLLAR_BODY_POSE_INDEX, :]


def smpl_body_pose_to_spine3_aa(body_pose: np.ndarray) -> np.ndarray:
    """World axis-angle of spine3 (joint 9), accumulated along the spine chain.

    ``body_pose[j-1]`` holds the local rotation of SMPL joint ``j``; the spine
    chain is root → spine1 (j=3) → spine2 (j=6) → spine3 (j=9).  Use this with
    ``spine3_pos = body_positions[9]`` so arm positions computed from the
    extracted ``arm_aa`` match the full-body positions.

    Args:
        body_pose: ``(21, 3)`` SMPL body_pose axis-angles.

    Returns:
        ``(3,)`` world axis-angle of spine3.
    """
    body_pose = np.asarray(body_pose, dtype=np.float64)
    spine3_world_rot = (
        Rotation.from_rotvec(body_pose[2])  # spine1 (j=3)
        * Rotation.from_rotvec(body_pose[5])  # spine2 (j=6)
        * Rotation.from_rotvec(body_pose[8])  # spine3 (j=9)
    )
    return spine3_world_rot.as_rotvec()


# ---------------------------------------------------------------------------
# Full pipeline: HumanML3D 263-dim → SMPL body_pose
# ---------------------------------------------------------------------------


def _import_recover_from_ric() -> Callable[..., Any]:
    """Import the MDM submodule's ``recover_from_ric`` with sys.path set up."""
    mdm_dir = Path(__file__).resolve().parent / "motion-diffusion-model"
    if str(mdm_dir) not in sys.path:
        sys.path.insert(0, str(mdm_dir))
    # pylint: disable=import-outside-toplevel,import-error
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    return recover_from_ric


def hml263_to_smpl_body_pose(
    hml_vec: "torch.Tensor",  # type: ignore[name-defined]  # noqa: F821
    dataset,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """Convert normalized HumanML3D vectors to SMPL body_pose.

    De-normalizes, recovers global XYZ via the official ``recover_from_ric``,
    then runs minimum-rotation IK per frame.  The 6D rotation block is never
    read directly: officially encoded rotations are relative to the t2m
    reference skeleton (arms hanging down), not this repo's SMPL T-pose.

    Args:
        hml_vec:  ``(n_frames, 263)`` normalized HumanML3D motion tensor.
        dataset:  HumanML3D ``DataLoader`` — accesses
                  ``dataset.dataset.t2m_dataset.inv_transform``.
        tpose_22: ``(22, 3)`` T-pose joint positions (from
                  :attr:`SmplLeftArmFK.tpose_all_joints`).

    Returns:
        ``(n_frames, 21, 3)`` SMPL body_pose axis-angles.
    """
    recover_from_ric = _import_recover_from_ric()
    hml_vec = hml_vec.float().cpu()

    unnorm = dataset.dataset.t2m_dataset.inv_transform(
        hml_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, n_frames, 263)
    ).float()
    positions = recover_from_ric(unnorm, 22)[0, 0].numpy().astype(np.float64)

    return np.stack(
        [positions_to_smpl_body_pose(frame, tpose_22) for frame in positions]
    )  # (n_frames, 21, 3)


def hml263_batch_to_smpl_body_pose(
    hml_vecs: "torch.Tensor",  # type: ignore[name-defined]  # noqa: F821
    dataset,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """Batched variant of :func:`hml263_to_smpl_body_pose`.

    Args:
        hml_vecs: ``(n_samples, n_frames, 263)`` normalized HumanML3D tensor.
        dataset:  HumanML3D ``DataLoader`` — accesses
                  ``dataset.dataset.t2m_dataset.inv_transform``.
        tpose_22: ``(22, 3)`` T-pose joint positions.

    Returns:
        ``(n_samples, n_frames, 21, 3)`` SMPL body_pose axis-angles.
    """
    recover_from_ric = _import_recover_from_ric()
    hml_vecs = hml_vecs.float().cpu()

    unnorm = dataset.dataset.t2m_dataset.inv_transform(
        hml_vecs.unsqueeze(1)  # (n_samples, 1, n_frames, 263)
    ).float()
    positions = recover_from_ric(unnorm, 22)[:, 0].numpy().astype(np.float64)

    return smpl_positions_batch_to_body_pose(positions, tpose_22)


def hml263_batch_to_smpl_positions(  # pylint: disable=too-many-locals
    hml_vecs: "torch.Tensor",  # type: ignore[name-defined]  # noqa: F821
    dataset,
    model,
) -> np.ndarray:
    """Convert normalized HumanML3D vectors to batched SMPL XYZ positions.

    This is the cheap part of the HML → SMPL pipeline.  It avoids the
    per-frame IK needed to recover local axis-angle rotations, so it is the
    preferred representation for UQ clustering and preview rendering.

    Args:
        hml_vecs: ``(n_samples, n_frames, 263)`` normalized HumanML3D tensor.
        dataset: HumanML3D ``DataLoader`` returned by ``get_dataset_loader``.
        model:   MDM model — must expose ``model.rot2xyz``.

    Returns:
        ``(n_samples, n_frames, 22, 3)`` global SMPL joint positions.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    hml_vecs = hml_vecs.float().cpu()

    # --- De-normalize all samples at once -----------------------------------
    # inv_transform expects (batch, nfeats=1, seq_len, 263).
    hml_t0 = time.perf_counter()
    unnorm = dataset.dataset.t2m_dataset.inv_transform(
        hml_vecs.unsqueeze(1)  # (n_samples, 1, n_frames, 263)
    ).float()

    # --- Recover global XYZ positions for all samples -----------------------
    # recover_from_ric: (n_samples, 1, n_frames, 263) → (n_samples, 1, n_frames, 22, 3)
    ric = recover_from_ric(unnorm, 22)
    # (n_samples, 22, 3, n_frames)
    ric = ric.view(-1, *ric.shape[2:]).permute(0, 2, 3, 1)
    print(f"[timing] HML recovery: {time.perf_counter() - hml_t0:.3f}s")

    # Single rot2xyz call for the full batch (was n_samples separate calls).
    rot2xyz_t0 = time.perf_counter()
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
    # xyz: (n_samples, 22, 3, n_frames) → (n_samples, n_frames, 22, 3)
    positions_all: np.ndarray = xyz.permute(0, 3, 1, 2).cpu().numpy()
    print(f"[timing] rot2xyz: {time.perf_counter() - rot2xyz_t0:.3f}s")
    return positions_all


def smpl_positions_batch_to_body_pose(
    positions_all: np.ndarray,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """Convert batched SMPL XYZ positions to local axis-angle body poses.

    Args:
        positions_all: ``(n_samples, n_frames, 22, 3)`` global joint positions.
        tpose_22:      ``(22, 3)`` T-pose positions.

    Returns:
        ``(n_samples, n_frames, 21, 3)`` SMPL body_pose axis-angles.
    """
    # pylint: disable=import-outside-toplevel
    from concurrent.futures import ThreadPoolExecutor

    positions_all = np.asarray(positions_all, dtype=np.float64)
    n_samples, n_frames = positions_all.shape[:2]

    # --- IK: parallelise across samples -------------------------------------
    # positions_to_smpl_body_pose is pure numpy/scipy; Rotation.align_vectors
    # releases the GIL so ThreadPoolExecutor gives real parallelism.
    def _ik_sample(positions_seq: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                positions_to_smpl_body_pose(positions_seq[t], tpose_22)
                for t in range(n_frames)
            ]
        )  # (n_frames, 21, 3)

    max_workers = max(1, min(n_samples, cpu_count() or 1))
    print(
        f"[timing] IK setup: {n_samples} samples x {n_frames} frames "
        f"using {max_workers} workers"
    )
    ik_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        body_poses = list(pool.map(_ik_sample, positions_all))
    print(f"[timing] IK: {time.perf_counter() - ik_t0:.3f}s")

    return np.stack(body_poses)  # (n_samples, n_frames, 21, 3)
