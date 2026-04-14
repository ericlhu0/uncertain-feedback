"""Extract a normalized first-frame pose and save it as ``demo_final_pose.pt``.

Run from any directory with the mdm conda env:

    conda activate mdm
    python src/uncertain_feedback/motion_generators/mdm/make_initial_pose.py

Output: ``src/uncertain_feedback/motion_generators/mdm/demo_final_pose.pt``  (263, 1)
"""

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "motion-diffusion-model" / "dataset" / "HumanML3D"
OUTPUT_PATH = HERE / "demo_pose.pt"

# Load the first training trajectory (shape: N x 263, raw HML263)
npy_path = DATASET_DIR / "new_joint_vecs" / "000001.npy"
data = np.load(npy_path)  # (N, 263)
print(f"Loaded {npy_path}  shape={data.shape}")

# MDM expects normalized vectors as model input. Convert raw frame to normalized.
mean = np.load(DATASET_DIR / "Mean.npy").astype(np.float32)  # (263,)
std = np.load(DATASET_DIR / "Std.npy").astype(np.float32)  # (263,)
first_frame_raw = data[0].astype(np.float32)  # (263,)
first_frame_norm = (first_frame_raw - mean) / (std + 1e-8)  # (263,)

# Reshape to (263, 1) as expected by sample_leftarm.py
first_frame = torch.from_numpy(first_frame_norm).unsqueeze(-1)  # (263, 1)
print(f"First frame shape (normalized): {first_frame.shape}")

torch.save(first_frame, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
