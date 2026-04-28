"""Decode a normalized demo.pt (single-frame HML263) to SMPL-22 world positions.

Uses only the RIC block (dims 4:67) and root height (dim 3), which gives
approximate world positions for heading ≈ 0.  No GPU or MDM model required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from uncertain_feedback.data_collection.smpl_to_hml263 import load_hml_stats
from uncertain_feedback.planners.mpc.kinematics import (
    SMPL_BONE_PAIRS_22,
    SMPL_PARENTS_22,
    SmplLeftArmFK,
)

# Joints the user can edit (arms + torso spine)
EDITABLE_JOINTS: list[int] = [3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21]
FIXED_JOINTS: list[int] = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15]

JOINT_NAMES: list[str] = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
    "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
    "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist",
]


def demo_pt_to_positions(pt_path: Path, hml_stats_dir: Path) -> np.ndarray:
    """Return (22, 3) float32 SMPL world positions from a normalized demo.pt.

    Assumes heading ≈ 0 (skeleton facing forward) for the base pose.
    """
    mean, std = load_hml_stats(hml_stats_dir)
    norm_vec = (
        torch.load(Path(pt_path), map_location="cpu", weights_only=True)
        .squeeze()
        .numpy()
        .astype(np.float64)
    )  # (263,)
    raw = norm_vec * std + mean  # (263,) unnormalized

    # root height above ground (dim 3)
    root_y = float(raw[3])

    # RIC positions: joints 1-21 relative to root, in local frame (dims 4:67)
    # For heading=0: local_x = world_x - root_x, local_y = world_y, local_z = world_z - root_z
    ric = raw[4:67].reshape(21, 3)

    world_pos = np.zeros((22, 3), dtype=np.float32)
    world_pos[0] = [0.0, root_y, 0.0]
    world_pos[1:] = ric.astype(np.float32)
    return world_pos


def get_tpose_bone_lengths() -> list[float]:
    """Return T-pose bone length for each of the 22 joints (0.0 for root)."""
    tpose = SmplLeftArmFK().tpose_all_joints  # (22, 3)
    lengths = [0.0] * 22
    for parent, child in SMPL_BONE_PAIRS_22:
        lengths[child] = float(np.linalg.norm(tpose[child] - tpose[parent]))
    return lengths
