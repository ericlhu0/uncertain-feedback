"""Extract one frame of a raw HML263 motion file and save it as a normalized
``(263, 1)`` pose tensor usable as ``start_pose`` / YAML ``pose``.

Usage:

    uv run python src/uncertain_feedback/motion_generators/mdm/make_initial_pose.py \\
        --motion path/to/new_joint_vecs/000001.npy \\
        --frame -1 \\
        --out src/uncertain_feedback/motion_generators/mdm/my_pose.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_STATS_DIR = HERE / "motion-diffusion-model" / "dataset" / "HumanML3D"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--motion",
    required=True,
    type=Path,
    help="Path to a raw (unnormalized) HML263 .npy motion file (N, 263).",
)
parser.add_argument(
    "--frame",
    type=int,
    default=0,
    help="Frame index to extract (supports negative indexing; default: 0).",
)
parser.add_argument("--out", required=True, type=Path, help="Output .pt path.")
parser.add_argument(
    "--stats_dir",
    type=Path,
    default=DEFAULT_STATS_DIR,
    help="Directory containing Mean.npy and Std.npy for normalization.",
)
args = parser.parse_args()

data = np.load(args.motion)  # (N, 263) raw HML263
print(f"Loaded {args.motion}  shape={data.shape}")

# MDM expects normalized vectors as model input. Convert raw frame to normalized.
mean = np.load(args.stats_dir / "Mean.npy").astype(np.float32)  # (263,)
std = np.load(args.stats_dir / "Std.npy").astype(np.float32)  # (263,)
frame_raw = data[args.frame].astype(np.float32)  # (263,)
frame_norm = (frame_raw - mean) / (std + 1e-8)  # (263,)

pose = torch.from_numpy(frame_norm).unsqueeze(-1)  # (263, 1)
torch.save(pose, args.out)
print(f"Saved frame {args.frame} (normalized, shape {tuple(pose.shape)}) to {args.out}")
