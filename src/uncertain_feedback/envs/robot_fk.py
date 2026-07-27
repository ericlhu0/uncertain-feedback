"""Batched analytic forward kinematics for a robot's end-effector chain.

Extracted once from a pybullet body and evaluated in vectorized numpy, so an
MPC can roll out thousands of joint-space samples per step without touching
pybullet (``getLinkState`` is one configuration at a time). The chain is the
base→ee ancestry only; joints off the chain (gripper fingers) do not move the
end effector and are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


def _axis_rotations(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues rotations about a fixed ``(3,)`` axis for ``(...,)`` angles."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    sin = np.sin(theta)[..., None, None]
    cos = np.cos(theta)[..., None, None]
    return np.eye(3) + sin * k + (1.0 - cos) * (k @ k)


@dataclass(frozen=True)
class RobotChainFK:
    """World-frame ee pose from robot joint angles, batched.

    ``rel_pos``/``rel_rot`` are each chain link's URDF joint-origin transform in
    its parent link frame, measured numerically from pybullet at the all-zero
    configuration (the first entry is world→first-link, absorbing the loaded
    base pose). A joint's rotation applies in its child link frame
    (``T_i = T_{i-1} · T_origin_i · R(axis_i, q_i)``), matching the URDF
    convention pybullet's world link frames follow.
    """

    rel_pos: np.ndarray  # (J, 3)
    rel_rot: np.ndarray  # (J, 3, 3)
    axes: np.ndarray  # (J, 3) joint axis in the child link frame; zero = fixed
    movable: np.ndarray  # (J,) bool, base→ee order

    @classmethod
    def from_pybullet(cls, body: int, ee_index: int, cid: int) -> "RobotChainFK":
        """Measure the ee chain of a loaded body at its current base pose."""
        chain: list[int] = []
        link = ee_index
        while link != -1:
            chain.append(link)
            link = p.getJointInfo(body, link, physicsClientId=cid)[16]
        chain.reverse()

        infos = [p.getJointInfo(body, j, physicsClientId=cid) for j in chain]
        movable = np.array([info[2] != p.JOINT_FIXED for info in infos])
        movable_joints = [j for j, m in zip(chain, movable) if m]
        saved = [
            p.getJointState(body, j, physicsClientId=cid)[:2] for j in movable_joints
        ]
        for j in movable_joints:
            p.resetJointState(body, j, 0.0, physicsClientId=cid)

        world_pos = np.zeros(3)
        world_rot = np.eye(3)
        rel_pos = np.zeros((len(chain), 3))
        rel_rot = np.zeros((len(chain), 3, 3))
        for i, j in enumerate(chain):
            pos, orn = p.getLinkState(
                body, j, computeForwardKinematics=True, physicsClientId=cid
            )[4:6]
            pos = np.asarray(pos, dtype=np.float64)
            rot = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            rel_pos[i] = world_rot.T @ (pos - world_pos)
            rel_rot[i] = world_rot.T @ rot
            world_pos, world_rot = pos, rot

        for j, (position, velocity) in zip(movable_joints, saved):
            p.resetJointState(body, j, position, velocity, physicsClientId=cid)

        axes = np.array(
            [info[13] if m else (0.0, 0.0, 0.0) for info, m in zip(infos, movable)]
        )
        return cls(rel_pos=rel_pos, rel_rot=rel_rot, axes=axes, movable=movable)

    @property
    def n_movable(self) -> int:
        """Number of actuated joints on the chain (the length of ``q``)."""
        return int(self.movable.sum())

    def ee_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """End-effector pose for ``(..., n_movable)`` joint angles.

        Returns:
            Tuple of ``(..., 3)`` world positions and ``(..., 3, 3)`` world
            rotation matrices.
        """
        q = np.asarray(q, dtype=np.float64)
        batch = q.shape[:-1]
        pos = np.zeros((*batch, 3))
        rot = np.broadcast_to(np.eye(3), (*batch, 3, 3)).copy()
        k = 0
        for i, is_movable in enumerate(self.movable):
            pos = pos + np.einsum("...ij,j->...i", rot, self.rel_pos[i])
            rot = rot @ self.rel_rot[i]
            if is_movable:
                rot = rot @ _axis_rotations(self.axes[i], q[..., k])
                k += 1
        return pos, rot

    def ee_pose_jacobian(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """End-effector pose and geometric Jacobian, batched.

        Returns:
            Tuple of ``(..., 3)`` positions, ``(..., 3, 3)`` rotations, and
            the ``(..., 6, n_movable)`` world-frame geometric Jacobian —
            rows are [linear; angular], column *k* is
            ``(z_k × (p_ee − p_k), z_k)`` for joint *k*'s world axis ``z_k``
            and origin ``p_k``.
        """
        q = np.asarray(q, dtype=np.float64)
        batch = q.shape[:-1]
        pos = np.zeros((*batch, 3))
        rot = np.broadcast_to(np.eye(3), (*batch, 3, 3)).copy()
        origins = np.zeros((*batch, self.n_movable, 3))
        world_axes = np.zeros((*batch, self.n_movable, 3))
        k = 0
        for i, is_movable in enumerate(self.movable):
            pos = pos + np.einsum("...ij,j->...i", rot, self.rel_pos[i])
            rot = rot @ self.rel_rot[i]
            if is_movable:
                origins[..., k, :] = pos
                world_axes[..., k, :] = np.einsum("...ij,j->...i", rot, self.axes[i])
                rot = rot @ _axis_rotations(self.axes[i], q[..., k])
                k += 1
        linear = np.cross(world_axes, pos[..., None, :] - origins)
        jac = np.swapaxes(np.concatenate([linear, world_axes], axis=-1), -1, -2)
        return pos, rot, jac
