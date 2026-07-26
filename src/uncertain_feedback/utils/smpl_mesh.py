"""SMPL mesh generation and binary caching for the demo runner."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx import create

from uncertain_feedback.planners.mpc.kinematics import anatomical_elbow_wrist_slots

_BODY_MODELS_DIR = (
    Path(__file__).parents[1]
    / "motion_generators/mdm/motion-diffusion-model/body_models"
)
_LEG_BONES = ((1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11))
# Left shoulder, elbow, wrist, hand in the SMPL 24-joint skinning order. The
# collar is left out: its rotation is locked for a run, so those vertices are the
# same in every pose.
_LEFT_ARM_LBS_JOINTS = (16, 18, 20, 22)


class SmplMeshCache:
    """Generate coherent SMPL meshes from FK arm-chain world positions.

    The repo FK convention differs from native SMPL (a joint's rotation
    transforms the bone ending at it vs. the bones leaving it), and native
    SMPL cannot reproduce repo poses exactly at branch joints, so the demo
    body is fitted once by joint-position optimization. Each frame the
    collar/shoulder/elbow locals are re-derived from the FK arm-chain
    positions (bone directions are convention-free and the two skeletons are
    identical), so the mesh arm tracks the skeleton drawn in the UI up to the
    torso fit residual at the collar.
    """

    def __init__(self, body_positions: np.ndarray, max_entries: int = 64) -> None:
        self._model = create(
            str(_BODY_MODELS_DIR), model_type="smpl", gender="neutral", ext="pkl"
        )
        self._model.eval()
        with torch.no_grad():
            self._rest = (
                self._model(
                    body_pose=torch.zeros(1, 69),
                    global_orient=torch.zeros(1, 3),
                    betas=torch.zeros((1, 10), dtype=torch.float32),
                    transl=torch.zeros(1, 3),
                )
                .joints[0, :24]
                .numpy()
                .astype(np.float64)
            )
        self._global_orient, self._baseline, self._translation = self._fit_demo_pose(
            np.asarray(body_positions, dtype=np.float32)
        )
        self._spine3_world = self._world_rotations()[9]
        self._max_entries = max_entries
        self._next_id = 0
        self._poses: OrderedDict[str, np.ndarray] = OrderedDict()
        self._vertices: dict[str, np.ndarray] = {}
        self._pinned: set[str] = set()

    def _fit_demo_pose(
        self, target_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit one coherent native-SMPL pose to the demo's 22 joints."""
        target = torch.from_numpy(target_positions)
        global_orient = torch.zeros((1, 3), requires_grad=True)
        body_pose = torch.zeros((1, 69), requires_grad=True)
        translation = torch.tensor(
            target_positions[0][None], dtype=torch.float32, requires_grad=True
        )
        optimizer = torch.optim.Adam([global_orient, body_pose, translation], lr=0.05)
        for _ in range(400):
            optimizer.zero_grad()
            output = self._model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=torch.zeros((1, 10), dtype=torch.float32),
                transl=translation,
            )
            joint_error = output.joints[0, :22] - target
            loss = (joint_error * joint_error).mean() + 1e-5 * (
                body_pose * body_pose
            ).mean()
            loss.backward()
            optimizer.step()
        orient = global_orient.detach().numpy()[0].astype(np.float32)
        baseline = body_pose.detach().numpy()[0].reshape(23, 3).astype(np.float32)
        baseline[19] = 0.0
        return (
            orient,
            self._detwist_legs(baseline, orient),
            translation.detach().numpy()[0].astype(np.float32),
        )

    def _world_rotations(self) -> list[Rotation]:
        """Chained world rotations of the fitted pose (24 joints)."""
        parents = self._model.parents.numpy()
        local = Rotation.from_rotvec(
            np.vstack([self._global_orient[None], self._baseline]).astype(np.float64)
        )
        world = [local[0]]
        for j in range(1, 24):
            world.append(world[parents[j]] * local[j])
        return world

    def _detwist_legs(
        self, body_pose: np.ndarray, global_orient: np.ndarray
    ) -> np.ndarray:
        """Zero the position-loss null space (bone twist) in the leg chains.

        Replaces each leg joint's local rotation with the minimum rotation that
        reproduces the fitted bone direction, keeping joint positions intact.
        """
        parents = self._model.parents.numpy()
        local = Rotation.from_rotvec(
            np.vstack([global_orient[None], body_pose]).astype(np.float64)
        )
        fitted_world = [local[0]]
        for j in range(1, 24):
            fitted_world.append(fitted_world[parents[j]] * local[j])
        world = list(fitted_world)
        for joint, child in _LEG_BONES:
            bone = self._rest[child] - self._rest[joint]
            direction = fitted_world[joint].apply(bone)
            new_world, _ = Rotation.align_vectors([direction], [bone])
            body_pose[joint - 1] = (
                (world[parents[joint]].inv() * new_world).as_rotvec().astype(np.float32)
            )
            world[joint] = new_world
        return body_pose

    @property
    def faces(self) -> np.ndarray:
        """The SMPL body model's triangle indices."""
        return np.asarray(self._model.faces, dtype=np.int32)

    @property
    def left_arm_faces(self) -> np.ndarray:
        """Triangles skinned to the left arm, by LBS weight.

        For drawing a *second* body — a goal ghost — over the first: only the
        left arm is controlled, so a whole-body ghost would coincide with the
        person's own mesh everywhere else, and coincident surfaces z-fight into
        speckle. Selecting by skinning weight rather than a hardcoded vertex
        list keeps this tied to the model actually loaded.
        """
        weights = self._model.lbs_weights.detach().cpu().numpy()
        on_arm = weights[:, list(_LEFT_ARM_LBS_JOINTS)].sum(axis=1) > 0.5
        faces = np.asarray(self._model.faces, dtype=np.int32)
        return faces[on_arm[faces].all(axis=1)]

    def register(self, arm_positions: np.ndarray, pin: bool = False) -> str:
        """Register a pose sequence and return its id.

        ``pin`` exempts the entry from eviction: reference rollouts live as long
        as their trajectory, and one live MPC rollout registers a frame per step
        -- far more than ``max_entries`` -- so an unpinned reference id is dead
        by the time the user ticks it on.
        """
        mesh_id = str(self._next_id)
        self._next_id += 1
        self._poses[mesh_id] = np.asarray(arm_positions, dtype=np.float64).reshape(
            -1, 5, 3
        )
        if pin:
            self._pinned.add(mesh_id)
        while len(self._poses) - len(self._pinned) > self._max_entries:
            stale_id = next(k for k in self._poses if k not in self._pinned)
            del self._poses[stale_id]
            self._vertices.pop(stale_id, None)
        return mesh_id

    def unpin(self, mesh_id: str) -> None:
        """Make a pinned entry evictable again."""
        self._pinned.discard(mesh_id)

    def vertices(self, mesh_id: str) -> np.ndarray:
        """Mesh vertices for every frame of a registered pose sequence."""
        if mesh_id not in self._poses:
            raise KeyError(f"Unknown or stale mesh id {mesh_id!r}.")
        self._poses.move_to_end(mesh_id)
        if mesh_id not in self._vertices:
            self._vertices[mesh_id] = self._generate(self._poses[mesh_id])
        return self._vertices[mesh_id]

    def vertices_at(self, mesh_id: str, frame: int) -> np.ndarray:
        """Vertices for a single frame, shaped ``(1, n_verts, 3)``.

        The magnitude slider mints a mesh per position but only ever draws the
        frame on screen, so generating that frame alone keeps a drag off the
        whole-trajectory path. Deliberately uncached: these are transient.
        """
        if mesh_id not in self._poses:
            raise KeyError(f"Unknown or stale mesh id {mesh_id!r}.")
        self._poses.move_to_end(mesh_id)
        cached = self._vertices.get(mesh_id)
        poses = self._poses[mesh_id]
        index = min(max(frame, 0), poses.shape[0] - 1)
        if cached is not None:
            return cached[index : index + 1]
        return self._generate(poses[index : index + 1])

    def preview(self, arm_positions: np.ndarray) -> np.ndarray:
        """Mesh vertices for a single un-registered arm pose."""
        return self._generate(np.asarray(arm_positions).reshape(1, 5, 3))[0]

    def _generate(self, arm_positions: np.ndarray) -> np.ndarray:
        n_frames = arm_positions.shape[0]
        body_pose = np.repeat(self._baseline[None], n_frames, axis=0)
        for f in range(n_frames):
            # Collar (13) via shortest arc onto the clavicle bone (unchanged).
            collar_world, _ = Rotation.align_vectors(
                [arm_positions[f, 2] - arm_positions[f, 1]],
                [self._rest[16] - self._rest[13]],
            )
            body_pose[f, 12] = (self._spine3_world.inv() * collar_world).as_rotvec()
            # Native SMPL orients the bone *leaving* a joint, so the anatomical
            # elbow/wrist slots shift up one row: the recovered upper-arm
            # orientation (with shoulder twist) drives the shoulder(16) row and
            # the pure forearm hinge drives the elbow(18) row, so the
            # hand-carrying frame at joint 18 is stable.  parent = collar world.
            shoulder_row, elbow_row = anatomical_elbow_wrist_slots(
                arm_positions[f, 2],
                arm_positions[f, 3],
                arm_positions[f, 4],
                collar_world,
                self._rest[18] - self._rest[16],
                self._rest[20] - self._rest[18],
            )
            body_pose[f, 15] = shoulder_row
            body_pose[f, 17] = elbow_row
        with torch.no_grad():
            output = self._model(
                body_pose=torch.from_numpy(body_pose).reshape(n_frames, -1),
                global_orient=torch.from_numpy(
                    np.repeat(self._global_orient[None], n_frames, axis=0)
                ),
                betas=torch.zeros((n_frames, 10), dtype=torch.float32),
                transl=torch.from_numpy(
                    np.repeat(self._translation[None], n_frames, axis=0)
                ),
            )
        vertices = output.vertices.detach().cpu().numpy().astype("<f4")
        return np.ascontiguousarray(vertices)
