"""HML263 conversion + side-by-side visualization for the demo image.

Prerequisite: run the MHR inference worker to produce smpl_out.npz::

    uv run python \\
        src/uncertain_feedback/data_collection/_mhr_inference_worker.py \\
        --image_folder src/uncertain_feedback/data_collection/demo/images \\
        --output_path src/uncertain_feedback/data_collection/demo/smpl_out.npz \\
        --sam_checkpoint_path \\
            src/uncertain_feedback/data_collection/sam-3d-body/checkpoints/\\
            sam-3d-body-dinov3/model.ckpt

Then run this script (no special env needed)::

    uv run python src/uncertain_feedback/data_collection/demo/run_demo.py
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from uncertain_feedback.data_collection.show_conversion import (
    _BG,
    _FG,
    _draw_skeleton,
    _hml263_to_local_positions,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_DEMO_DIR = Path(__file__).parent
_MDM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "motion_generators"
    / "mdm"
    / "motion-diffusion-model"
)
_SMPL_PKL = _MDM_ROOT / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"
_HML_STATS = _MDM_ROOT / "dataset" / "HumanML3D"
_NPZ = _DEMO_DIR / "smpl_out.npz"
_OUT = _DEMO_DIR / "comparison.png"


def main() -> None:
    """Convert smpl_out.npz to HML263 and render a side-by-side comparison figure."""
    if not _NPZ.exists():
        raise FileNotFoundError(
            f"{_NPZ} not found.\n"
            "Run _mhr_inference_worker.py first to produce it (see docstring above)."
        )

    print(f"Loading {_NPZ} …")
    data = np.load(_NPZ, allow_pickle=True)

    positions_arr = data["smpl_positions"]  # (N, 22, 3)
    frame_paths = data["frame_paths"]  # (N,)
    n_frames = positions_arr.shape[0]

    tpose_22 = SmplLeftArmFK(_SMPL_PKL).tpose_all_joints
    mean, std = load_hml_stats(_HML_STATS)

    print("Converting MHR positions → HML263 …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = positions_to_hml263(
            positions=positions_arr,
            mean=mean,
            std=std,
            tpose_22=tpose_22,
        )  # (N, 263)

    positions = _hml263_to_local_positions(features, mean, std)  # (N, 22, 3)

    print("Rendering comparison figure …")
    fig = plt.figure(figsize=(5 * n_frames, 9), facecolor=_BG)
    fig.suptitle(
        "MHR → HML263: input image (top) vs. HML263-decoded 3D pose (bottom)",
        color=_FG,
        fontsize=12,
        y=0.99,
    )

    for col, t in enumerate(range(n_frames)):
        ax_img = fig.add_subplot(2, n_frames, col + 1)
        ax_img.imshow(plt.imread(str(frame_paths[t])))
        ax_img.set_title(f"Frame {t}  (input image)", color=_FG, fontsize=9)
        ax_img.axis("off")

        ax_sk = fig.add_subplot(2, n_frames, n_frames + col + 1, projection="3d")
        ax_sk.set_facecolor(_BG)
        _draw_skeleton(ax_sk, positions[t], title=f"Frame {t}  HML263 pose")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(_OUT), dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    main()
