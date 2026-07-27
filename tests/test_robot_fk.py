"""Tests for the batched robot end-effector FK against pybullet ground truth."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
import pytest
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.robot_fk import RobotChainFK
from uncertain_feedback.envs.sim_mannequin import _KINOVA_URDF, _PANDA_URDF


def _load(urdf: Path, ee_link: bytes, cid: int, **kwargs) -> tuple[int, int]:
    body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid, **kwargs)
    ee_index = next(
        j
        for j in range(p.getNumJoints(body, physicsClientId=cid))
        if p.getJointInfo(body, j, physicsClientId=cid)[12] == ee_link
    )
    return body, ee_index


def _assert_matches_pybullet(
    body: int, ee_index: int, chain: RobotChainFK, cid: int
) -> None:
    movable = [
        j
        for j in range(p.getNumJoints(body, physicsClientId=cid))
        if p.getJointInfo(body, j, physicsClientId=cid)[2] != p.JOINT_FIXED
    ]
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, size=len(movable))
        for joint, value in zip(movable, q):
            p.resetJointState(body, joint, float(value), physicsClientId=cid)
        ref_pos, ref_orn = p.getLinkState(
            body, ee_index, computeForwardKinematics=True, physicsClientId=cid
        )[4:6]
        pos, rot = chain.ee_pose(q)
        np.testing.assert_allclose(pos, ref_pos, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            rot, Rotation.from_quat(ref_orn).as_matrix(), rtol=0.0, atol=1e-6
        )


def test_panda_ee_pose_matches_pybullet() -> None:
    cid = p.connect(p.DIRECT)
    try:
        body, ee_index = _load(
            _PANDA_URDF,
            b"ee_link",
            cid,
            basePosition=(0.3, -0.2, 0.1),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, 1.2)),
        )
        chain = RobotChainFK.from_pybullet(body, ee_index, cid)
        assert chain.n_movable == 7
        _assert_matches_pybullet(body, ee_index, chain, cid)
    finally:
        p.disconnect(cid)


@pytest.mark.skipif(not _KINOVA_URDF.exists(), reason="kortex URDF not installed")
def test_kinova_ee_pose_matches_pybullet() -> None:
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath("/home/emprise", physicsClientId=cid)
        body, ee_index = _load(_KINOVA_URDF, b"tool_frame", cid)
        chain = RobotChainFK.from_pybullet(body, ee_index, cid)
        assert chain.n_movable == 7
        _assert_matches_pybullet(body, ee_index, chain, cid)
    finally:
        p.disconnect(cid)


def test_ee_pose_jacobian_matches_finite_differences() -> None:
    cid = p.connect(p.DIRECT)
    try:
        body, ee_index = _load(_PANDA_URDF, b"ee_link", cid)
        chain = RobotChainFK.from_pybullet(body, ee_index, cid)
    finally:
        p.disconnect(cid)

    q = np.random.default_rng(2).uniform(-1.2, 1.2, size=(5, 7))
    pos, rot, jac = chain.ee_pose_jacobian(q)
    assert jac.shape == (5, 6, 7)
    ref_pos, ref_rot = chain.ee_pose(q)
    np.testing.assert_allclose(pos, ref_pos)
    np.testing.assert_allclose(rot, ref_rot)

    eps = 1e-7
    for k in range(7):
        dq = np.zeros(7)
        dq[k] = eps
        pos1, rot1 = chain.ee_pose(q + dq)
        linear = (pos1 - pos) / eps
        angular = (
            Rotation.from_matrix(rot1 @ np.swapaxes(rot, -1, -2)).as_rotvec() / eps
        )
        np.testing.assert_allclose(jac[:, 0:3, k], linear, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(jac[:, 3:6, k], angular, rtol=0.0, atol=1e-5)


def test_ee_pose_batched_shapes_and_state_restored() -> None:
    cid = p.connect(p.DIRECT)
    try:
        body, ee_index = _load(_PANDA_URDF, b"ee_link", cid)
        movable = [
            j
            for j in range(p.getNumJoints(body, physicsClientId=cid))
            if p.getJointInfo(body, j, physicsClientId=cid)[2] != p.JOINT_FIXED
        ]
        before = [0.1 * (j + 1) for j in range(len(movable))]
        for joint, value in zip(movable, before):
            p.resetJointState(body, joint, value, physicsClientId=cid)

        chain = RobotChainFK.from_pybullet(body, ee_index, cid)
        after = [
            p.getJointState(body, j, physicsClientId=cid)[0] for j in movable
        ]
        np.testing.assert_allclose(after, before)

        q = np.random.default_rng(1).uniform(-1.0, 1.0, size=(4, 3, 7))
        pos, rot = chain.ee_pose(q)
        assert pos.shape == (4, 3, 3)
        assert rot.shape == (4, 3, 3, 3)
        single_pos, single_rot = chain.ee_pose(q[2, 1])
        np.testing.assert_allclose(pos[2, 1], single_pos)
        np.testing.assert_allclose(rot[2, 1], single_rot)
    finally:
        p.disconnect(cid)
