"""Example script: run the full pipeline and visualize a few frames."""

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
from uncertain_feedback.utils.plot import ArmVisualizer

# ── Run pipeline ──────────────────────────────────────────────────────────────
config = MhrToHml263Config(
    mhr_estimator_config=MhrEstimatorConfig(
        sam_checkpoint_path=str(
            Path(__file__).parent.parent
            / "sam-3d-body"
            / "checkpoints"
            / "sam-3d-body-dinov3"
            / "model.ckpt",
        )
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
    ArmVisualizer.draw_smpl_skeleton(ax, positions[t], title=f"Frame {t}")
plt.tight_layout()
plt.savefig("smpl_poses.png", dpi=150)
print("Saved smpl_poses.png")
