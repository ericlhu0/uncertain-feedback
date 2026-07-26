"""Map mocap keypoints into the planner's arm configuration.

Registration is *measured*, not assumed. Two rigid bodies do the work:

- The **robot base** body gives its own orientation and position, so the robot's
  place in the scene is measured rather than the hardcoded guess
  ``SimMannequinEnv`` uses.
- The **collar** body closes the last degree of freedom. The collar rotation is
  locked for a run (``fk.collar_aa``, decoded from the start pose), so the
  collar->shoulder bone direction in the torso frame is *known*; matching it
  against the measured direction pins the mocap->pybullet yaw. Without it the
  person's orientation relative to the robot is unobservable and has to be
  assumed, which silently rotates every measured bone direction.

Both the person and the robot are placed where they were *measured*: pybullet
is the mocap world turned by the solved yaw, so the collar sits at the measured
collar and the robot base at the measured base. The startup pose supplies the
torso's shape and orientation, not its position — a person who sits somewhere
else next run moves with their markers instead of the robot moving around a
pinned torso. The anchor this yields is frozen for the run (see
:attr:`ArmRegistration.spine3_smpl`), which is what :class:`MpcCostContext`
requires; Cartesian goals are spine3-relative, so they follow the person.

Within a run only bone *directions* come from mocap. They are re-anchored at
the frozen collar and rescaled to SMPL bone lengths. Rescaling is required for
correctness, not cosmetics — the returned ``q`` is consumed through SMPL FK, so
it must be a valid SMPL arm configuration whatever the person's real limb
lengths are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.sim_mannequin import _SMPL_TO_PB
from uncertain_feedback.mocap.natnet import RigidBodyPose
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa

# A clavicle pointing straight up or down has no horizontal component to take a
# yaw from. Anatomically it never should, so this is a wiring/mounting error.
_MIN_HORIZONTAL = 1e-3


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _horizontal_angle(vector: np.ndarray, what: str) -> float:
    if float(np.hypot(vector[0], vector[1])) < _MIN_HORIZONTAL:
        raise ValueError(
            f"{what} is vertical ({np.round(vector, 4)}), so the registration yaw "
            "is undetermined. Check the collar and shoulder rigid-body ids and "
            "their marker placement."
        )
    return float(np.arctan2(vector[1], vector[0]))


def arm_keypoints(
    bodies: dict[int, RigidBodyPose],
    collar_id: int,
    shoulder_id: int,
    elbow_id: int,
    wrist_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Collar/shoulder/elbow/wrist positions, or ``None`` if any is untracked."""
    positions: list[np.ndarray] = []
    for body_id in (collar_id, shoulder_id, elbow_id, wrist_id):
        pose = bodies.get(body_id)
        if pose is None or not pose.valid:
            return None
        positions.append(pose.position)
    return positions[0], positions[1], positions[2], positions[3]


@dataclass(frozen=True)
class ArmRegistration:
    """Frozen mocap→planner registration captured once at startup."""

    fk: SmplLeftArmFK
    spine3_aa: np.ndarray | None
    # Torso anchor the run plans against: the startup pose translated so its
    # collar lands on the measured one. Callers must plan with this rather than
    # the config's `spine3_pos`, or the person's placement and the goals will
    # disagree.
    spine3_smpl: np.ndarray
    collar_smpl: np.ndarray
    # SMPL-frame translation from the startup pose to the measured placement.
    # Apply it to anything else expressed in that pose's frame — `body_pos`.
    translation_smpl: np.ndarray
    collar_pb: np.ndarray
    base_pb: np.ndarray
    # Mocap -> pybullet: a pure yaw about the shared vertical.
    rotation: Rotation
    # Pybullet yaw to load the robot base at. Includes the base body's measured
    # orientation, so it is *not* the yaw of `rotation`.
    robot_base_yaw: float
    clavicle_length: float
    upper_arm_length: float
    forearm_length: float

    @classmethod
    def calibrate(
        cls,
        fk: SmplLeftArmFK,
        q0: np.ndarray,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        base_position: np.ndarray,
        base_orientation: np.ndarray,
        collar_mocap: np.ndarray,
        shoulder_mocap: np.ndarray,
    ) -> "ArmRegistration":
        """Register the mocap world against the startup configuration ``q0``.

        The mocap->pybullet rotation is *solved*, not configured: the locked
        collar rotation fixes the clavicle's direction in the torso frame, so
        requiring the measured clavicle to match it determines the yaw exactly.
        Mocap, the Kinova base, and pybullet are all Z-up, so the fit stays a
        rotation about the shared vertical — any leftover angle is the person's
        real clavicle elevation differing from the start pose's, which must
        *not* tilt the world. Keep Motive streaming Z-up for that reason: a
        Y-up stream would need an extra fixed conversion and would stop sharing
        an "up" axis with the robot base and pybullet.

        The base body's orientation is used *only* for the robot's facing in the
        scene, never for the measured bone directions — so it must be aligned in
        Motive to the Kinova base frame (x forward, z up). If it is not, the
        robot is loaded rotated away from reality and IK solves against the
        wrong pose, while the person's placement stays correct, which makes the
        error hard to see.

        Only ``q0``'s clavicle slot (``arm_aa[0]``, the collar->shoulder bone)
        is read. Its elbow and wrist slots are irrelevant: the caller gets the
        person's real ones back from :meth:`q_from_keypoints`. The clavicle
        cannot be measured the same way — it is the reference the yaw is
        solved against, so at startup :meth:`q_from_keypoints` returns ``q0``'s
        clavicle slot by construction.

        The caller must build its scene with :attr:`robot_base_yaw`, or the
        scene and the measured directions will disagree, and must plan against
        :attr:`spine3_smpl` rather than the ``spine3_pos`` passed in: the pose
        is translated onto the measured collar, so the two differ by
        :attr:`translation_smpl` whenever the person is not sitting exactly
        where the startup pose puts them (i.e. always).

        Args:
            base_position:    ``(3,)`` base rigid-body position in mocap world.
            base_orientation: ``(4,)`` base rigid-body xyzw quaternion, aligned
                              to the Kinova base frame.
            collar_mocap:     ``(3,)`` startup collar position in mocap world.
            shoulder_mocap:   ``(3,)`` startup shoulder position in mocap world.
        """
        positions = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), spine3_pos, spine3_aa)
        clavicle_start = _unit(_SMPL_TO_PB @ (positions[2] - positions[1]))

        # Mocap -> pybullet is a pure yaw about the shared vertical, solved from
        # the clavicle. The base body's own orientation plays no part: it
        # describes the marker plate, not the person, so a plate that was never
        # aligned cannot corrupt the measured bone directions.
        clavicle_measured = _unit(
            np.asarray(shoulder_mocap, dtype=np.float64) - collar_mocap
        )
        yaw = _horizontal_angle(clavicle_start, "start-pose clavicle") - (
            _horizontal_angle(clavicle_measured, "measured clavicle")
        )
        yaw = float((yaw + np.pi) % (2.0 * np.pi) - np.pi)
        rotation = Rotation.from_euler("z", yaw)

        # The robot's facing in the scene is a separate question: take the
        # plate's measured orientation into pybullet, then correct for how the
        # plate is mounted on the base. Unlike the mapping above, an unaligned
        # plate *does* land here — the robot would be loaded rotated away from
        # reality and IK would solve against the wrong pose.
        robot_base_yaw = float(
            (rotation * Rotation.from_quat(base_orientation)).as_euler("zyx")[0]
        )

        # Place the person on their *measured* collar: with the yaw solved,
        # pybullet is just the mocap world turned by it, so the collar and the
        # robot base both land where they were measured and the startup pose
        # contributes only the torso's shape and orientation.
        collar_pb = rotation.apply(np.asarray(collar_mocap, dtype=np.float64))
        translation_smpl = _SMPL_TO_PB.T @ collar_pb - positions[1]

        tpose = fk.tpose_joints
        return cls(
            fk=fk,
            spine3_aa=spine3_aa,
            spine3_smpl=positions[0] + translation_smpl,
            collar_smpl=positions[1] + translation_smpl,
            translation_smpl=translation_smpl,
            collar_pb=collar_pb,
            base_pb=rotation.apply(np.asarray(base_position, dtype=np.float64)),
            rotation=rotation,
            robot_base_yaw=robot_base_yaw,
            clavicle_length=float(np.linalg.norm(tpose[2] - tpose[1])),
            upper_arm_length=float(np.linalg.norm(tpose[3] - tpose[2])),
            forearm_length=float(np.linalg.norm(tpose[4] - tpose[3])),
        )

    def q_from_keypoints(
        self,
        collar: np.ndarray,
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
    ) -> np.ndarray:
        """Convert one mocap-world arm measurement to the planner's ``(7,)`` q.

        The chain is anchored at the collar frozen at calibration — the measured
        collar of *that* frame, so mid-run torso translation is discarded. The
        clavicle, upper-arm, and forearm directions are all measured, so the
        clavicle block of ``q`` tracks the person's shoulder girdle instead of
        staying at the start pose's value.
        """
        clavicle = self.rotation.apply(_unit(np.asarray(shoulder, np.float64) - collar))
        upper = self.rotation.apply(_unit(np.asarray(elbow, np.float64) - shoulder))
        forearm = self.rotation.apply(_unit(np.asarray(wrist, np.float64) - elbow))
        shoulder_pb = self.collar_pb + clavicle * self.clavicle_length
        elbow_pb = shoulder_pb + upper * self.upper_arm_length
        wrist_pb = elbow_pb + forearm * self.forearm_length
        positions = np.stack(
            [
                self.spine3_smpl,
                self.collar_smpl,
                _SMPL_TO_PB.T @ shoulder_pb,
                _SMPL_TO_PB.T @ elbow_pb,
                _SMPL_TO_PB.T @ wrist_pb,
            ]
        )
        arm_aa = self.fk.arm_aa_from_positions(positions, self.spine3_aa)
        return self.fk.arm_aa_to_q(arm_aa, self.spine3_aa)
