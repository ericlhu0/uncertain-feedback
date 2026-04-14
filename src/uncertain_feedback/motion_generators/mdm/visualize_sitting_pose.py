# pylint: disable=duplicate-code
"""Visualize the sitting pose before and after HumanML3D → SMPL conversion.

Produces a side-by-side 3D plot saved to ``sitting_pose_comparison.png``:

  Left panel  — Full 22-joint skeleton decoded from ``sitting_pose.pt`` via
                the MDM pipeline (inv_transform → recover_from_ric → rot2xyz).
                This is the ground-truth sitting pose as MDM sees it.

  Right panel — Same skeleton, but with the arm joints replaced by positions
                reconstructed through our IK pipeline:
                  hml263_to_smpl_body_pose → smpl_body_pose_to_arm_aa
                  → SmplLeftArmFK.fk
                Non-arm joints are shown at T-pose.  This lets us verify that
                our conversion faithfully captures the arm configuration.

Run from the repo root::

    uv run --extra develop python \\
        src/uncertain_feedback/motion_generators/mdm/visualize_sitting_pose.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — must be set before importing pyplot
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup — mirror sample_leftarm.py
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.consts import MDM_MODEL_WEIGHTS_PATH, MDM_ROOT

_MDM_SUBDIR = MDM_ROOT / "motion-diffusion-model"
if str(_MDM_SUBDIR) not in sys.path:
    sys.path.insert(0, str(_MDM_SUBDIR))

os.chdir(_MDM_SUBDIR)

# ---------------------------------------------------------------------------
# MDM imports (only valid after sys.path setup + chdir)
# ---------------------------------------------------------------------------

# pylint: disable=import-error
import torch
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel

from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    hml263_to_smpl_body_pose,
    smpl_body_pose_to_positions,
)
from uncertain_feedback.motion_generators.mdm.mdm_parser_util import edit_args
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SMPL_BONE_PAIRS_22,
    SmplLeftArmFK,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARM_JOINT_SET = set(LEFT_ARM_CHAIN_INDICES)  # {9, 13, 16, 18, 20}


def _load_model_and_data():
    """Load MDM model + dataset (same pattern as sample_leftarm.py)."""
    orig_argv = sys.argv
    sys.argv = [
        "visualize_sitting_pose.py",
        "--model_path",
        str(MDM_MODEL_WEIGHTS_PATH),
        "--text_condition",
        "",
    ]
    try:
        args = edit_args()
    finally:
        sys.argv = orig_argv

    args.batch_size = 1
    args.num_samples = 1
    args.num_repetitions = 1
    args.no_video = True

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    data = get_dataset_loader(
        name="humanml",
        batch_size=1,
        num_frames=196,
        split="test",
        hml_mode="text_only",
        fixed_len=0,
        pred_len=0,
        device=dist_util.dev(),
    )
    model, _ = create_model_and_diffusion(args, data)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    return model, data, args


def _decode_hml263_to_xyz(hml_vec: torch.Tensor, dataset) -> np.ndarray:
    """Decode a single (1, 263) normalized HML vector to (22, 3) XYZ positions.

    Uses recover_from_ric directly (no rot2xyz) — consistent with how
    sample_leftarm.py visualizes ground-truth input_motions.  rot2xyz would
    add a second vertical translation on top of the one already embedded in
    the HML263 root-height feature, lifting the pelvis erroneously.
    """
    hml_vec = hml_vec.float().cpu()
    unnorm = dataset.dataset.t2m_dataset.inv_transform(
        hml_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 263)
    ).float()
    ric = recover_from_ric(unnorm, 22)  # (1, 1, 1, 22, 3)
    return ric[0, 0, 0].cpu().numpy()  # (22, 3)


def _draw_skeleton(  # pylint: disable=too-many-locals
    ax, positions: np.ndarray, title: str, highlight_joints=None
) -> None:
    """Draw a 22-joint skeleton on a 3D axes.

    Args:
        ax:               matplotlib 3D Axes.
        positions:        ``(22, 3)`` joint world positions.
        title:            Axes title.
        highlight_joints: Set of joint indices to draw in a distinct color
                          (used to emphasise the arm chain).
    """
    highlight_joints = highlight_joints or set()

    # Draw bones
    for parent, child in SMPL_BONE_PAIRS_22:
        is_arm_bone = parent in highlight_joints or child in highlight_joints
        color = "#e05c2a" if is_arm_bone else "#4a90d9"
        lw = 2.5 if is_arm_bone else 1.5
        xs = [positions[parent, 0], positions[child, 0]]
        ys = [positions[parent, 1], positions[child, 1]]
        zs = [positions[parent, 2], positions[child, 2]]
        ax.plot(xs, zs, ys, color=color, linewidth=lw)

    # Draw joints
    for j in range(22):
        is_arm = j in highlight_joints
        color = "#e05c2a" if is_arm else "#4a90d9"
        size = 40 if is_arm else 20
        ax.scatter(
            positions[j, 0],
            positions[j, 2],
            positions[j, 1],
            c=color,
            s=size,
            zorder=5,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=10, azim=-60)

    # Equal aspect ratio
    ranges = np.ptp(positions, axis=0)
    max_range = max(ranges) / 2
    mid = positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[2] - max_range, mid[2] + max_range)
    ax.set_zlim(mid[1] - max_range, mid[1] + max_range)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load sitting pose, decode via MDM and via IK→FK, and save comparison."""
    print("Loading MDM model and dataset…")
    model, data, _ = _load_model_and_data()  # type: ignore[no-untyped-call]
    fk = SmplLeftArmFK()

    # Load the sitting pose (263-dim, normalized)
    sitting_pt = torch.load(MDM_ROOT / "demo_final_pose.pt", map_location=dist_util.dev())
    sitting = sitting_pt.squeeze(-1).unsqueeze(0)  # (1, 263)

    # --- Panel 1: original pose decoded by MDM ------------------------------
    print("Decoding sitting pose via MDM pipeline…")
    original_xyz = _decode_hml263_to_xyz(sitting, data)  # (22, 3)

    # --- Panel 2: full body reconstructed through our IK → FK pipeline ------
    print("Converting via hml263_to_smpl_body_pose → IK → FK…")
    body_pose = hml263_to_smpl_body_pose(sitting, data, model, fk.tpose_all_joints)
    # body_pose[0] is (21, 3) — full SMPL body_pose, not just the arm.
    # Run full-body FK to recover all 22 joint positions.
    reconstructed_xyz = smpl_body_pose_to_positions(
        body_pose[0], fk.tpose_all_joints, root_pos=original_xyz[0]
    )  # (22, 3)

    # --- Plot ---------------------------------------------------------------
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Sitting pose: MDM decode vs. HML→SMPL conversion", fontsize=13)

    ax1 = fig.add_subplot(121, projection="3d")
    _draw_skeleton(ax1, original_xyz, "Original (MDM decode)", _ARM_JOINT_SET)

    ax2 = fig.add_subplot(122, projection="3d")
    _draw_skeleton(ax2, reconstructed_xyz, "Converted (IK → FK)", _ARM_JOINT_SET)

    # Add a legend note
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel

    legend_elements = [
        Line2D([0], [0], color="#e05c2a", lw=2.5, label="Left arm"),
        Line2D([0], [0], color="#4a90d9", lw=1.5, label="Rest of body"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    out_path = Path(__file__).resolve().parent / "sitting_pose_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
