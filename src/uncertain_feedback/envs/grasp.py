"""Forward kinematics to the robot grasp frame on the human forearm.

Shared by simulated and (future) real-robot envs so both realize the same
human-q → grasp-point-pose mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa

GRASP_FRACTION = 0.15

_SMPL_UP = np.array([0.0, 1.0, 0.0])


def grasp_pose_fk(
    fk: SmplLeftArmFK,
    q: np.ndarray,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    fraction: float = GRASP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose of the gripper grasping the forearm, in SMPL (Y-up) coordinates.

    The grasp point sits ``fraction`` of the way along the elbow→wrist
    segment. The frame follows the Panda hand convention (z = approach axis,
    y = finger-closing axis), so the forearm lies along the gripper x-axis and
    the gripper approaches top-down.

    Args:
        q: ``(7,)`` planner arm configuration.
        fraction: Position along the forearm, 0 = elbow, 1 = wrist.

    Returns:
        Tuple of ``(3,)`` position and ``(4,)`` xyzw quaternion.
    """
    arm_aa = q_to_arm_aa(q, fk.elbow_hinge_axis)
    positions = fk.fk(arm_aa, spine3_pos, spine3_aa)
    elbow, wrist = positions[3], positions[4]
    grasp_pos = elbow + fraction * (wrist - elbow)

    x_axis = wrist - elbow
    x_axis = x_axis / np.linalg.norm(x_axis)
    up = (
        _SMPL_UP
        if abs(float(np.dot(x_axis, _SMPL_UP))) <= 0.99
        else np.array([1.0, 0.0, 0.0])
    )
    z_axis = -up - float(np.dot(-up, x_axis)) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()
    return grasp_pos, quat
