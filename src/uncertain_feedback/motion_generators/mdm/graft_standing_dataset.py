"""Graft the custom1 left-arm motions onto a standing body → ``custom1_standing``.

Every clip in ``dataset/custom1/`` has a seated body (pelvis ~0.6 m, knees ~85°)
with only the left arm moving.  Fine-tuning on it teaches the checkpoint to snap
standing bodies into sitting.  This script rebuilds the same 44 clips on a
neutral **standing** body so the fine-tune can start from the base checkpoint
without a posture shift:

1. Sample one static standing body from the base checkpoint (see
   :func:`standing_template_positions`) and IK it to a template ``body_pose``.
2. For each clip: decode raw HML263 → global positions → per-frame ``body_pose``,
   splice the clip's left collar/shoulder/elbow/wrist rows into the static
   template ``body_pose``, run full-body FK with a static root, and re-encode
   with the official HumanML3D ``process_file`` (via ``positions_to_hml263``).
3. Copy ``texts/``, the split files and the stock ``Mean.npy``/``Std.npy`` across
   unchanged — the stats are the stock HumanML3D ones and must not be recomputed.

``process_file`` drops the last frame, so clips come out ``(49, 263)`` instead of
``(50, 263)``.

Run from the repo root::

    uv run python src/uncertain_feedback/motion_generators/mdm/graft_standing_dataset.py \\
        --verify-dir /tmp/graft_verify
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — must be set before importing pyplot
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.consts import MDM_ROOT

_MDM_SUBDIR = MDM_ROOT / "motion-diffusion-model"
if str(_MDM_SUBDIR) not in sys.path:
    sys.path.insert(0, str(_MDM_SUBDIR))

# pylint: disable=import-error
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    ARM_BODY_POSE_INDICES,
    COLLAR_BODY_POSE_INDEX,
    positions_to_smpl_body_pose,
    smpl_body_pose_to_positions,
)
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SMPL_PARENTS_22,
    SmplLeftArmFK,
)
from uncertain_feedback.utils.plot import ArmVisualizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SRC_DATASET = _MDM_SUBDIR / "dataset" / "custom1"
OUT_DATASET = _MDM_SUBDIR / "dataset" / "custom1_standing"
BASE_CHECKPOINT = _MDM_SUBDIR / "save" / "humanml_enc_512_50steps" / "model000750000.pt"

# body_pose rows taken from the clip: left collar, shoulder, elbow, wrist.
GRAFT_BODY_POSE_INDICES: list[int] = [COLLAR_BODY_POSE_INDEX] + ARM_BODY_POSE_INDICES

# Template sampling.  A 2 s "stand still" holds a clean neutral stand; longer
# samples drift into a deep squat, and the only real HumanML3D clip on disk
# (012314, tennis) has no neutral standing frame.
TEMPLATE_PROMPT = "a person stands still"
TEMPLATE_SECONDS = 2.0
TEMPLATE_SEED = 7
TEMPLATE_FRAME = 30

_ARM_JOINT_SET = set(LEFT_ARM_CHAIN_INDICES)  # {9, 13, 16, 18, 20}


# ---------------------------------------------------------------------------
# Standing template
# ---------------------------------------------------------------------------


def standing_template_positions() -> np.ndarray:
    """Sample a neutral standing ``(22, 3)`` body from the base checkpoint.

    The start pose is the all-zeros normalized frame, i.e. the HumanML3D dataset
    mean (an upright stand), so the sample stays inside the base checkpoint's own
    standing distribution rather than being dragged there from another posture.
    """
    # pylint: disable=import-outside-toplevel
    from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator

    gen = MdmMotionGenerator(
        model_path=BASE_CHECKPOINT, seed=TEMPLATE_SEED, lock_seed=True
    )
    samples = gen.generate_left_arm_position_samples(
        TEMPLATE_PROMPT,
        motion_length_seconds=TEMPLATE_SECONDS,
        start_pose=np.zeros(263),
        num_samples=1,
        frozen_body=False,
    )  # (1, n_frames, 22, 3)
    return samples[0, TEMPLATE_FRAME].astype(np.float64)


# ---------------------------------------------------------------------------
# Graft
# ---------------------------------------------------------------------------


def decode_raw_hml263(raw: np.ndarray) -> np.ndarray:
    """``(T, 263)`` **raw** (unnormalized) HML263 → ``(T, 22, 3)`` positions."""
    ric = recover_from_ric(torch.tensor(raw, dtype=torch.float32), 22)
    return ric.numpy().astype(np.float64)


def splice_and_fk(
    arm_rows: np.ndarray,
    template_body_pose: np.ndarray,
    template_root: np.ndarray,
    tpose_22: np.ndarray,
) -> np.ndarray:
    """FK the static template with ``arm_rows`` spliced into the grafted slots.

    Args:
        arm_rows:           ``(T, 4, 3)`` axis-angles for
                            ``GRAFT_BODY_POSE_INDICES``.
        template_body_pose: ``(21, 3)`` static standing template body_pose.
        template_root:      ``(3,)`` static template pelvis position.
        tpose_22:           ``(22, 3)`` SMPL T-pose joint positions.

    Returns:
        ``(T, 22, 3)`` global positions of the grafted body.
    """
    body_pose = np.repeat(template_body_pose[None], len(arm_rows), axis=0)
    body_pose[:, GRAFT_BODY_POSE_INDICES] = arm_rows
    return np.stack(
        [
            smpl_body_pose_to_positions(bp, tpose_22, root_pos=template_root)
            for bp in body_pose
        ]
    )


def graft_clip(
    raw: np.ndarray,
    template_body_pose: np.ndarray,
    template_root: np.ndarray,
    tpose_22: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splice a clip's left arm onto the static standing template.

    Args:
        raw:                ``(T, 263)`` raw HML263 clip features.
        template_body_pose: ``(21, 3)`` static standing template body_pose.
        template_root:      ``(3,)`` static template pelvis position.
        tpose_22:           ``(22, 3)`` SMPL T-pose joint positions.

    Returns:
        grafted_positions: ``(T, 22, 3)`` global positions of the grafted body.
        clip_body_pose:    ``(T, 21, 3)`` body_pose decoded from the clip.
        clip_positions:    ``(T, 22, 3)`` positions decoded from the clip.
    """
    clip_positions = decode_raw_hml263(raw)
    clip_body_pose = np.stack(
        [positions_to_smpl_body_pose(frame, tpose_22) for frame in clip_positions]
    )  # (T, 21, 3)
    grafted_positions = splice_and_fk(
        clip_body_pose[:, GRAFT_BODY_POSE_INDICES],
        template_body_pose,
        template_root,
        tpose_22,
    )
    return grafted_positions, clip_body_pose, clip_positions


# ---------------------------------------------------------------------------
# Verification metrics
# ---------------------------------------------------------------------------


def _bone_directions(frame: np.ndarray) -> np.ndarray:
    """``(21, 3)`` unit bone directions of a ``(22, 3)`` pose."""
    bones = np.stack([frame[j] - frame[SMPL_PARENTS_22[j]] for j in range(1, 22)])
    return bones / np.linalg.norm(bones, axis=-1, keepdims=True)


def _derotate(round_trip: np.ndarray, grafted: np.ndarray) -> np.ndarray:
    """Rotate the round trip back into the pre-encoding frame.

    ``process_file`` rotates the whole sequence so frame 0 faces Z+, and
    ``positions_to_smpl_body_pose`` takes per-bone *minimum* rotations, so it is
    not rotation-equivariant — that rotation has to be undone before the
    round-tripped axis-angles mean anything.  Fitted on unit bone directions
    rather than on positions or on ``process_file``'s own hip/shoulder axis:
    ``uniform_skeleton`` preserves bone directions to 0.01° but rescales bone
    lengths (which biases a positional fit), and the hip/shoulder axis goes
    ill-conditioned once the grafted arm swings the left shoulder.

    Args:
        round_trip: ``(T, 22, 3)`` positions decoded back from the encoded clip.
        grafted:    ``(T, 22, 3)`` pre-encoding positions.

    Returns:
        ``(T, 22, 3)`` round-trip positions in ``grafted``'s frame.
    """
    rot, _ = Rotation.align_vectors(
        _bone_directions(grafted[0]), _bone_directions(round_trip[0])
    )
    return rot.apply(round_trip.reshape(-1, 3)).reshape(round_trip.shape)


def _geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between two ``(..., 3)`` axis-angle arrays, in radians."""
    rel = Rotation.from_rotvec(a.reshape(-1, 3)).inv() * Rotation.from_rotvec(
        b.reshape(-1, 3)
    )
    return np.linalg.norm(rel.as_rotvec(), axis=-1).reshape(a.shape[:-1])


# ---------------------------------------------------------------------------
# Verification figures
# ---------------------------------------------------------------------------


def _contact_sheet(
    template_positions: np.ndarray,
    grafted: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Template pose plus the mid-frame of six grafted clips."""
    names = sorted(grafted)[:6]
    fig = plt.figure(figsize=(21, 6))
    ax = fig.add_subplot(2, 4, 1, projection="3d")
    ArmVisualizer.draw_smpl_skeleton(
        ax, template_positions, "standing template", _ARM_JOINT_SET
    )
    for i, name in enumerate(names):
        positions = grafted[name]
        mid = positions[len(positions) // 2]
        ax = fig.add_subplot(2, 4, i + 2, projection="3d")
        ArmVisualizer.draw_smpl_skeleton(
            ax, mid, f"{name} (mid, pelvis={mid[0, 1]:.2f})", _ARM_JOINT_SET
        )
    fig.suptitle("custom1_standing: template + grafted clip mid-frames", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _seated_vs_standing(
    name: str,
    clip_positions: np.ndarray,
    grafted_positions: np.ndarray,
    out_path: Path,
) -> None:
    """Original seated frames above the grafted standing frames, 4 timestamps."""
    n = len(grafted_positions)
    frames = [0, n // 3, 2 * n // 3, n - 1]
    fig = plt.figure(figsize=(4.5 * len(frames), 9))
    for i, t in enumerate(frames):
        ax = fig.add_subplot(2, len(frames), i + 1, projection="3d")
        ArmVisualizer.draw_smpl_skeleton(
            ax, clip_positions[t], f"seated f{t}", _ARM_JOINT_SET
        )
        ax = fig.add_subplot(2, len(frames), len(frames) + i + 1, projection="3d")
        ArmVisualizer.draw_smpl_skeleton(
            ax, grafted_positions[t], f"grafted f{t}", _ARM_JOINT_SET
        )
    fig.suptitle(f"{name}: original seated (top) vs grafted standing (bottom)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Build ``custom1_standing`` and report the round-trip deviations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-dir",
        type=Path,
        default=None,
        help="Directory for verification PNGs. Skipped when omitted.",
    )
    args = parser.parse_args()
    verify_dir = args.verify_dir.resolve() if args.verify_dir else None
    if verify_dir is not None:
        verify_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_hml_stats(SRC_DATASET)
    fk = SmplLeftArmFK()
    tpose_22 = fk.tpose_all_joints

    print("Sampling standing template from the base checkpoint…")
    template_positions = standing_template_positions()
    template_body_pose = positions_to_smpl_body_pose(template_positions, tpose_22)
    template_fk = smpl_body_pose_to_positions(
        template_body_pose, tpose_22, root_pos=template_positions[0]
    )
    print(
        f"template pelvis {template_fk[0, 1]:.3f} m "
        f"({template_fk[0, 1] - template_fk[:, 1].min():.3f} m above floor)"
    )

    (OUT_DATASET / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    clip_paths = sorted((SRC_DATASET / "new_joint_vecs").glob("*.npy"))

    grafted_positions_by_name: dict[str, np.ndarray] = {}
    clip_positions_by_name: dict[str, np.ndarray] = {}
    aa_dev: list[np.ndarray] = []
    arm_dev: list[np.ndarray] = []
    root_heights: list[np.ndarray] = []
    root_vel: list[np.ndarray] = []
    contacts: list[np.ndarray] = []

    for path in clip_paths:
        raw = np.load(path)
        grafted, clip_body_pose, clip_positions = graft_clip(
            raw, template_body_pose, template_positions[0], tpose_22
        )
        features = positions_to_hml263(grafted, mean, std, normalize=False)
        np.save(OUT_DATASET / "new_joint_vecs" / path.name, features)

        # --- round trip: decode the encoded clip and re-run the IK ------------
        n = len(features)
        round_trip = _derotate(decode_raw_hml263(features), grafted)
        round_trip_rows = np.stack(
            [positions_to_smpl_body_pose(frame, tpose_22) for frame in round_trip]
        )[:, GRAFT_BODY_POSE_INDICES]
        aa_dev.append(
            _geodesic(clip_body_pose[:n, GRAFT_BODY_POSE_INDICES], round_trip_rows)
        )
        # Re-FK the round-tripped rows on the same template so the arm positions
        # are compared on one skeleton (custom1's own skeleton is neither rigid
        # nor the t2m one that process_file retargets onto).
        round_trip_arm = splice_and_fk(
            round_trip_rows, template_body_pose, template_positions[0], tpose_22
        )
        arm_dev.append(
            np.linalg.norm(
                round_trip_arm[:, LEFT_ARM_CHAIN_INDICES]
                - grafted[:n, LEFT_ARM_CHAIN_INDICES],
                axis=-1,
            )
        )
        root_heights.append(features[:, 3])
        root_vel.append(features[:, 0:3])
        contacts.append(features[:, -4:])

        grafted_positions_by_name[path.stem] = grafted
        clip_positions_by_name[path.stem] = clip_positions
        print(f"  {path.stem}: {raw.shape} → {features.shape} {features.dtype}")

    # --- copy the rest of the dataset unchanged ------------------------------
    shutil.copytree(SRC_DATASET / "texts", OUT_DATASET / "texts", dirs_exist_ok=True)
    for name in ("train.txt", "val.txt", "test.txt", "Mean.npy", "Std.npy"):
        shutil.copy2(SRC_DATASET / name, OUT_DATASET / name)

    # --- report --------------------------------------------------------------
    aa = np.concatenate([d.ravel() for d in aa_dev])
    arm = np.concatenate([d.ravel() for d in arm_dev])
    heights = np.concatenate(root_heights)
    vel = np.concatenate(root_vel)
    contact = np.concatenate(contacts)
    print(f"\n{len(clip_paths)} clips written to {OUT_DATASET}")
    print(
        "left-arm axis-angle round-trip (rad): "
        f"mean={aa.mean():.4f} max={aa.max():.4f}"
    )
    print(
        "left-arm joint position round-trip (cm): "
        f"mean={arm.mean() * 100:.3f} max={arm.max() * 100:.3f}"
    )
    print(
        f"root height (feature 3): mean={heights.mean():.4f} std={heights.std():.5f} "
        f"min={heights.min():.4f} max={heights.max():.4f}"
    )
    print(f"root angular/linear velocity (features 0:3) max|.|={np.abs(vel).max():.5f}")
    print(f"foot contacts (features -4:) mean={contact.mean():.3f}")

    if verify_dir is None:
        return
    _contact_sheet(
        template_fk, grafted_positions_by_name, verify_dir / "01_contact_sheet.png"
    )
    for name in sorted(grafted_positions_by_name)[:2]:
        _seated_vs_standing(
            name,
            clip_positions_by_name[name],
            grafted_positions_by_name[name],
            verify_dir / f"02_seated_vs_standing_{name}.png",
        )
    print(f"Verification images in {verify_dir}")


if __name__ == "__main__":
    main()
