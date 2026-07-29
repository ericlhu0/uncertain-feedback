"""Posed SMPL body mesh as a pybullet visual body.

Shared by the envs that draw the person as a mesh rather than a skeleton
(:mod:`~uncertain_feedback.envs.sim_robot_visual` offscreen,
:mod:`~uncertain_feedback.envs.real` in the live GUI). Pybullet has no API for
updating a mesh's vertices, so each pose replaces the body.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

# SMPL world is Y-up; pybullet is Z-up. Proper rotation (x, y, z) -> (x, -z, y).
_SMPL_TO_PB = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
BODY_COLOR = (0.62, 0.71, 0.82, 1.0)
# Goal ghosts: green and see-through, so the person's own mesh reads through it.
GOAL_COLOR = (0.25, 0.85, 0.35, 0.35)
# The planner's own chain, in a colour no body mesh uses.
SKELETON_COLOR = (0.95, 0.35, 0.1)
# The person, when the arm chain is drawn inside them: the skeleton is the point
# of that view and an opaque body would simply hide it.
BODY_XRAY_COLOR = (0.62, 0.71, 0.82, 0.4)


class HumanMeshBody:
    """A body-mesh view of the arm chain, re-posed on demand.

    Args:
        cid:       Pybullet client to draw in.
        cache:     Mesh generator fitted to the run's body pose. Callers drawing
                   more than one body (e.g. the person plus a goal ghost) share
                   one cache — building it loads the SMPL model and fits the
                   torso.
        color:     rgba of the mesh. Alpha below 1 renders translucent in the GUI
                   (the offscreen TinyRenderer ignores it).
        arm_only:  Draw the left arm alone. For a second body over the first: the
                   arm is all the MPC controls, so the rest would coincide with
                   the person's own mesh and z-fight.
    """

    def __init__(
        self,
        cid: int,
        cache: SmplMeshCache,
        color: tuple[float, float, float, float] = BODY_COLOR,
        arm_only: bool = False,
    ) -> None:
        self._cid = cid
        self._color = color
        self._cache = cache
        self._faces = np.asarray(
            cache.left_arm_faces if arm_only else cache.faces, dtype=np.int64
        )
        self._body: int = -1

    def update(self, arm_positions: np.ndarray) -> None:
        """Re-pose the mesh to ``(5, 3)`` SMPL arm-chain world positions."""
        import trimesh  # pylint: disable=import-outside-toplevel

        vertices = self._cache.preview(arm_positions).astype(np.float64) @ _SMPL_TO_PB.T
        normals = trimesh.Trimesh(vertices, self._faces, process=False).vertex_normals
        if self._body >= 0:
            p.removeBody(self._body, physicsClientId=self._cid)
        vis = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vertices.tolist(),
            indices=self._faces.flatten().tolist(),
            normals=normals.tolist(),
            rgbaColor=self._color,
            physicsClientId=self._cid,
        )
        self._body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=(0.0, 0.0, 0.0),
            physicsClientId=self._cid,
        )


class ArmSkeletonBody:
    """The planner's arm chain itself: a bone per segment, a ball per joint.

    :class:`HumanMeshBody` shows a *body* posed from this chain, which makes the
    person legible but puts a shape fit and a skinned surface between the viewer
    and the numbers. This is the chain — the five joints (spine3, collar, shoulder,
    elbow, wrist) :meth:`SmplLeftArmFK.fk` returns and the costs read, drawn where
    they are.

    Real geometry rather than debug lines, because debug items are a GUI overlay:
    they are absent from ``getCameraImage``, so a skeleton drawn that way would go
    missing from exactly the screenshots and videos a run is checked from later.
    The bones are rigid, so each is built once at its own length and afterwards
    only moved — which costs nothing next to a mesh rebuild, so this can refresh
    every step while the mesh stays rate-limited.

    Args:
        cid:          Pybullet client to draw in.
        color:        rgb of the bones and joint balls.
        radius:       Bone radius in metres.
        joint_radius: Joint ball radius in metres.
    """

    def __init__(
        self,
        cid: int,
        color: tuple[float, float, float] = SKELETON_COLOR,
        radius: float = 0.008,
        joint_radius: float = 0.014,
    ) -> None:
        self._cid = cid
        self._color = color
        self._radius = radius
        self._joint_radius = joint_radius
        self._bones: list[int] = []
        self._joints: list[int] = []

    def update(self, arm_positions: np.ndarray) -> None:
        """Move the chain to ``(5, 3)`` SMPL arm-chain world positions."""
        points = np.asarray(arm_positions, dtype=np.float64) @ _SMPL_TO_PB.T
        segments = list(zip(points[:-1], points[1:]))
        if not self._joints:
            self._joints = [self._ball() for _ in points]
            self._bones = [
                self._bone(float(np.linalg.norm(end - start)))
                for start, end in segments
            ]
        for joint, point in zip(self._joints, points):
            p.resetBasePositionAndOrientation(
                joint, tuple(point), (0.0, 0.0, 0.0, 1.0), physicsClientId=self._cid
            )
        for bone, (start, end) in zip(self._bones, segments):
            # A cylinder is symmetric about its axis, so the twist align_vectors
            # leaves free does not matter.
            rotation, _ = Rotation.align_vectors([end - start], [[0.0, 0.0, 1.0]])
            p.resetBasePositionAndOrientation(
                bone,
                tuple((start + end) / 2.0),
                tuple(rotation.as_quat()),
                physicsClientId=self._cid,
            )

    def _ball(self) -> int:
        return self._body(
            p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self._joint_radius,
                rgbaColor=(*self._color, 1.0),
                physicsClientId=self._cid,
            )
        )

    def _bone(self, length: float) -> int:
        return self._body(
            p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=self._radius,
                length=length,
                rgbaColor=(*self._color, 1.0),
                physicsClientId=self._cid,
            )
        )

    def _body(self, visual_shape: int) -> int:
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=(0.0, 0.0, 0.0),
            physicsClientId=self._cid,
        )
