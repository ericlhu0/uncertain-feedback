"""Side-by-side visualization: original image vs. HML263-decoded 3D skeleton.

For each selected frame, the top row shows the original image and the bottom
row shows the 3D skeleton reconstructed from the HML263 feature vector.  The
skeleton is in the heading-local frame (person always facing forward), which
is the canonical representation stored in HML263.

Usage
-----
Step 1 – produce the positions npz with the inference worker::

    uv run python _mhr_inference_worker.py \\
        --image_folder ./frames \\
        --output_path /tmp/smpl_out.npz \\
        --sam_checkpoint_path \\
            src/uncertain_feedback/data_collection/sam-3d-body/checkpoints/\\
            sam-3d-body-dinov3/model.ckpt

Step 2 – generate the visualization::

    uv run python show_conversion.py \\
        --npz_path /tmp/smpl_out.npz \\
        --hml_stats_dir /path/to/HumanML3D/ \\
        [--output_path comparison.png] \\
        [--n_frames 3]
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.planners.mpc.kinematics import SMPL_BONE_PAIRS_22, SmplLeftArmFK

_LEFT_ARM_JOINTS = {13, 16, 18, 20}
_BG = "#1a1a2e"
_FG = "white"


# ---------------------------------------------------------------------------
# HML263 decode
# ---------------------------------------------------------------------------


def _hml263_to_local_positions(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Reconstruct 22-joint positions from normalized HML263 features.

    Positions are in the heading-local frame: root is placed at
    ``[0, root_height_above_ground, 0]`` and all other joints are expressed
    relative to the root in that rotated frame.

    Args:
        features: ``(N, 263)`` normalized HML263 array.
        mean:     ``(263,)`` HumanML3D normalization mean.
        std:      ``(263,)`` HumanML3D normalization std.

    Returns:
        ``(N, 22, 3)`` joint positions in heading-local frame.
    """
    raw = features * std + mean  # (N, 263) — un-normalized raw HML features

    n = features.shape[0]
    root_y = raw[:, 3]  # root height above ground (Y, metres)
    rel_pos = raw[:, 4:67].reshape(n, 21, 3)  # joints 1-21 relative to root

    positions = np.zeros((n, 22, 3), dtype=np.float64)
    positions[:, 0, 1] = root_y  # root XZ = 0, Y = height
    positions[:, 1:] = positions[:, :1] + rel_pos
    return positions


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------


def _draw_skeleton(ax: Any, positions: np.ndarray, title: str = "") -> None:
    """Draw a 3D skeleton on *ax* from 22-joint SMPL positions."""
    for parent, child in SMPL_BONE_PAIRS_22:
        is_arm = parent in _LEFT_ARM_JOINTS or child in _LEFT_ARM_JOINTS
        ax.plot(
            [positions[parent, 0], positions[child, 0]],
            [positions[parent, 2], positions[child, 2]],
            [positions[parent, 1], positions[child, 1]],
            color="#e05c2a" if is_arm else "#4a90d9",
            linewidth=2,
        )
    ax.scatter(
        positions[:, 0],
        positions[:, 2],
        positions[:, 1],
        color=_FG,
        s=12,
        zorder=5,
    )

    # Equal-aspect bounding box
    xyz = np.stack([positions[:, 0], positions[:, 2], positions[:, 1]], axis=1)
    mid = (xyz.max(0) + xyz.min(0)) / 2
    half = (xyz.max(0) - xyz.min(0)).max() / 2 + 0.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_title(title, color=_FG, fontsize=9, pad=4)
    ax.set_xlabel("X", color=_FG, fontsize=7)
    ax.set_ylabel("Z", color=_FG, fontsize=7)
    ax.set_zlabel("Y", color=_FG, fontsize=7)
    ax.tick_params(colors=_FG, labelsize=6)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.view_init(elev=10, azim=-60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pylint: disable=too-many-locals
    """Parse args and render side-by-side image vs. HML263 skeleton figure."""
    parser = argparse.ArgumentParser(
        description="Side-by-side visualization of original image vs HML263 pose"
    )
    parser.add_argument(
        "--npz_path",
        required=True,
        type=Path,
        help="npz produced by _mhr_inference_worker.py",
    )
    parser.add_argument(
        "--hml_stats_dir",
        required=True,
        type=Path,
        help="Directory containing Mean.npy and Std.npy",
    )
    parser.add_argument(
        "--output_path",
        default=Path("comparison.png"),
        type=Path,
        help="Output PNG path (default: comparison.png)",
    )
    parser.add_argument(
        "--n_frames",
        default=3,
        type=int,
        help="Number of evenly-spaced frames to show (default: 3)",
    )
    parser.add_argument(
        "--smpl_model_path",
        default=None,
        type=Path,
        help="Path to SMPL_NEUTRAL.pkl for loading T-pose joint positions.",
    )
    args = parser.parse_args()

    smpl_pkl = (
        Path(args.smpl_model_path).expanduser()
        if args.smpl_model_path
        else Path("~/MHR/tools/mhr_smpl_conversion/data/SMPL_NEUTRAL.pkl").expanduser()
    )
    tpose_22 = SmplLeftArmFK(smpl_pkl).tpose_all_joints  # (22, 3)

    # Load inference output
    data = np.load(args.npz_path, allow_pickle=True)
    frame_paths: np.ndarray = data["frame_paths"]  # (N,) str

    mean, std = load_hml_stats(args.hml_stats_dir)

    # MHR positions → HML263 → local positions
    features = positions_to_hml263(
        positions=data["smpl_positions"],
        mean=mean,
        std=std,
        tpose_22=tpose_22,
    )  # (N, 263) normalized
    positions = _hml263_to_local_positions(features, mean, std)  # (N, 22, 3)

    n_total = len(features)
    n_show = min(args.n_frames, n_total)
    frame_indices = np.linspace(0, n_total - 1, n_show, dtype=int)

    # Figure: 2 rows (image top, skeleton bottom) × n_show columns
    fig = plt.figure(figsize=(5 * n_show, 9), facecolor=_BG)
    fig.suptitle(
        "MHR → HML263: original image (top) vs. decoded 3D pose (bottom)",
        color=_FG,
        fontsize=12,
        y=0.99,
    )

    for col, t in enumerate(frame_indices):
        # ── Top row: original image ──────────────────────────────────────────
        ax_img = fig.add_subplot(2, n_show, col + 1)
        img = plt.imread(str(frame_paths[t]))
        ax_img.imshow(img)
        ax_img.set_title(f"Frame {t}", color=_FG, fontsize=9)
        ax_img.axis("off")

        # ── Bottom row: HML263 skeleton ──────────────────────────────────────
        ax_sk = fig.add_subplot(2, n_show, n_show + col + 1, projection="3d")
        ax_sk.set_facecolor(_BG)
        _draw_skeleton(ax_sk, positions[t], title="HML263 pose (local frame)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(args.output_path), dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
