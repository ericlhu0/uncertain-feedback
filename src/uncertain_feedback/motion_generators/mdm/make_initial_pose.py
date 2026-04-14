"""Extract the first frame from training data and save it as sitting_pose.pt.

Run from any directory with the mdm conda env:

    conda activate mdm
    python src/uncertain_feedback/motion_generators/mdm/make_initial_pose.py

Output: src/uncertain_feedback/motion_generators/mdm/sitting_pose.pt  (263, 1)
"""

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "motion-diffusion-model" / "dataset" / "HumanML3D"
OUTPUT_PATH = HERE / "demo_final_pose.pt"

# Load the first training trajectory (shape: N x 263, normalized HML263)
npy_path = DATASET_DIR / "new_joint_vecs" / "000001.npy"
data = np.load(npy_path)  # (N, 263)
print(f"Loaded {npy_path}  shape={data.shape}")

# Take the first frame and reshape to (263, 1) as expected by sample_leftarm.py
first_frame = torch.from_numpy(data[-1]).unsqueeze(-1)  # (263, 1)
print(f"First frame shape: {first_frame.shape}")

torch.save(first_frame, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
