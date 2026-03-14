"""Script to generate 't_pose.pt' in the HumanML3D format (263 dimensions).

Run this script to create the file, then move it to your MDM_ROOT
directory.
"""

import numpy as np
import torch

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


def generate_tpose_file(
    output_path: str = "t_pose.pt",
):
    """Generate a T-pose HumanML3D 263-dim feature vector and save it to a .pt
    file."""
    # 1. Load SMPL kinematics to get the default T-pose joints
    fk = SmplLeftArmFK()
    tpose_joints = fk.tpose_all_joints  # (22, 3)

    # 2. Initialize the 263-dimensional feature vector
    # Format:
    # [0:1]     Root angular velocity (Y)
    # [1:3]     Root linear velocity (XZ)
    # [3:4]     Root height (Y)
    # [4:67]    Local joint positions (21 joints * 3)
    # [67:193]  Local joint rotations (21 joints * 6)
    # [193:259] Local joint velocities (22 joints * 3)
    # [259:263] Foot contacts (4)
    vec = np.zeros(263, dtype=np.float32)

    # 3. Set Root Height (Y)
    # Normalize so the lowest joint (usually feet) is at y=0
    min_y = np.min(tpose_joints[:, 1])
    root_y_ground = tpose_joints[0, 1] - min_y
    vec[3] = root_y_ground

    # 4. Set Local Joint Positions (relative to root)
    # Exclude root (joint 0), take joints 1..21
    root_pos = tpose_joints[0]
    local_pos = tpose_joints[1:] - root_pos  # (21, 3)
    vec[4:67] = local_pos.flatten()

    # 5. Set Local Joint Rotations (6D)
    # T-pose has identity rotations. 6D identity is [1, 0, 0, 0, 1, 0].
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    vec[67:193] = np.tile(identity_6d, 21)

    # Velocities and contacts remain 0.

    # 6. Save to .pt file
    torch.save(torch.from_numpy(vec), output_path)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    generate_tpose_file()
