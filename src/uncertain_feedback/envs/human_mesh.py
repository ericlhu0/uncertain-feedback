"""Posed SMPL body mesh as a pybullet visual body.

Shared by the envs that draw the person as a mesh rather than a skeleton
(:mod:`~uncertain_feedback.envs.sim_robot_visual` offscreen,
:mod:`~uncertain_feedback.envs.real` in the live GUI). Pybullet has no API for
updating a mesh's vertices, so each pose replaces the body.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

# SMPL world is Y-up; pybullet is Z-up. Proper rotation (x, y, z) -> (x, -z, y).
_SMPL_TO_PB = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
BODY_COLOR = (0.62, 0.71, 0.82, 1.0)
# Goal ghosts: green and see-through, so the person's own mesh reads through it.
GOAL_COLOR = (0.25, 0.85, 0.35, 0.35)


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
