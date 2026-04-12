"""Example script: run the full pipeline and plot a few frames of the result."""

# pylint: disable=wrong-import-position

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # or "TkAgg" / "Qt5Agg" for interactive

import matplotlib.pyplot as plt  # noqa: E402

from uncertain_feedback.data_collection import (
    MhrEstimatorConfig,
    MhrToHml263Config,
    MhrToHml263Pipeline,
)
from uncertain_feedback.planners.mpc.kinematics import SMPL_BONE_PAIRS_22


# ── Copy _draw_skeleton inline (it's not exported) ───────────────────────────
def draw_skeleton(skeleton_ax, joint_positions, title="", highlight_joints=None):
    """Draw a 3D skeleton on *skeleton_ax* from 22-joint SMPL positions."""
    highlight_joints = highlight_joints or set()
    for parent, child in SMPL_BONE_PAIRS_22:
        is_arm = parent in highlight_joints or child in highlight_joints
        skeleton_ax.plot(
            [joint_positions[parent, 0], joint_positions[child, 0]],
            [joint_positions[parent, 2], joint_positions[child, 2]],
            [joint_positions[parent, 1], joint_positions[child, 1]],
            color="#e05c2a" if is_arm else "#4a90d9",
        )
    skeleton_ax.set_title(title)
    skeleton_ax.set_xlabel("X")
    skeleton_ax.set_ylabel("Z")
    skeleton_ax.set_zlabel("Y")
    skeleton_ax.view_init(elev=10, azim=-60)


# ── Run pipeline ──────────────────────────────────────────────────────────────
config = MhrToHml263Config(
    mhr_estimator_config=MhrEstimatorConfig(
        sam_checkpoint_path=Path(__file__).parent
        / "sam-3d-body"
        / "checkpoints"
        / "sam-3d-body-dinov3"
        / "model.ckpt",
    ),
    hml_stats_dir=Path("path/to/HumanML3D/Mean_Std/"),
)
pipeline = MhrToHml263Pipeline(config)

# Get 22-joint world positions (N, 22, 3)
positions = pipeline.run_to_smpl_positions(Path("./video_frames/"))

# ── Plot a few frames ─────────────────────────────────────────────────────────
frames_to_plot = [0, len(positions) // 2, -1]
fig = plt.figure(figsize=(5 * len(frames_to_plot), 5))
for i, t in enumerate(frames_to_plot):
    ax = fig.add_subplot(1, len(frames_to_plot), i + 1, projection="3d")
    draw_skeleton(ax, positions[t], title=f"Frame {t}")  # type: ignore[no-untyped-call]
plt.tight_layout()
plt.savefig("smpl_poses.png", dpi=150)
print("Saved smpl_poses.png")
