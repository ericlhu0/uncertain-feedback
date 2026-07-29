"""Forward kinematics to the robot grasp frame on the human forearm.

Two grasps live here. :func:`grasp_pose_fk` is the *assumed* one the simulated
envs place their robot at: a fixed fraction along the forearm, approached
top-down. :class:`MeasuredGrasp` is for the real world, where the grasp is
already established before the MPC starts and so is *measured* instead — the
gripper's actual pose relative to the forearm, from which the gripper pose for
another arm configuration follows. It is rigid over one step, not over a run:
:class:`~uncertain_feedback.envs.real.RealEnv` re-measures it each step, since
the forearm shifts inside the fingers as the trajectory runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa

GRASP_FRACTION = 0.15

_SMPL_UP = np.array([0.0, 1.0, 0.0])


def forearm_frame_fk(
    fk: SmplLeftArmFK,
    q: np.ndarray,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> tuple[np.ndarray, Rotation]:
    """Forearm frame at the elbow, in SMPL (Y-up) coordinates.

    The reference frame a measured grasp is expressed in. It is the FK's own
    forearm bone rotation (see
    :meth:`SmplLeftArmFK.bone_world_rotations`), not a frame built from the
    elbow→wrist direction plus a world up-reference: the latter flips its roll
    when the forearm passes through vertical, which would swing a rigidly held
    gripper by half a turn.

    Args:
        q: ``(7,)`` planner arm configuration.

    Returns:
        Tuple of the ``(3,)`` elbow position and the forearm rotation.
    """
    arm_aa = q_to_arm_aa(q, fk.elbow_hinge_axis)
    positions = fk.fk(arm_aa, spine3_pos, spine3_aa)
    return positions[3], fk.bone_world_rotations(arm_aa, spine3_aa)[3]


@dataclass(frozen=True)
class MeasuredGrasp:
    """Gripper-on-forearm transform, measured from one forearm/gripper pose pair.

    Replaces the assumed grasp geometry of :func:`grasp_pose_fk` for the real
    robot: where the gripper actually holds the forearm is whatever the operator
    grasped, so it is read off the end-effector pose and the measured forearm
    pose rather than dictated. Rigid by construction, so re-measure it as often
    as the physical grasp moves.
    """

    # Gripper origin in the forearm frame.
    position: np.ndarray
    # Forearm frame → gripper frame. Absorbs the robot's tool-frame convention
    # (``_RobotSpec.tool_quat``), since it is measured on the ee link itself.
    rotation: Rotation

    @classmethod
    def measure(
        cls,
        forearm_pos: np.ndarray,
        forearm_rot: Rotation,
        ee_pos: np.ndarray,
        ee_rot: Rotation,
    ) -> "MeasuredGrasp":
        """Capture the transform from one simultaneous forearm/gripper pose pair.

        Both poses must be in the same frame (the caller's world frame); the
        result is frame-independent.
        """
        return cls(
            position=forearm_rot.inv().apply(
                np.asarray(ee_pos, np.float64) - forearm_pos
            ),
            rotation=forearm_rot.inv() * ee_rot,
        )

    def gripper_pose(
        self, forearm_pos: np.ndarray, forearm_rot: Rotation
    ) -> tuple[np.ndarray, Rotation]:
        """Gripper pose implied by a forearm pose, holding the grasp rigid."""
        return (
            forearm_pos + forearm_rot.apply(self.position),
            forearm_rot * self.rotation,
        )


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
