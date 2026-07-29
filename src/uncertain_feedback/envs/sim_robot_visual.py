"""Visualization-only simulated robot env: no physics, kinematic human.

The human arm achieves every commanded configuration exactly (as in
:class:`KinematicEnv`); a Franka Panda is posed via IK each step so its
end-effector holds the grasp point on the human forearm. The human renders as
a posed SMPL mesh fitted to the run's decoded initial body pose (same fit the
demo runner shows). PyBullet runs in DIRECT mode purely for IK and camera
rendering — ``stepSimulation`` is never called.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.grasp import GRASP_FRACTION, grasp_pose_fk
from uncertain_feedback.envs.human_mesh import HumanMeshBody
from uncertain_feedback.planners.mpc.kinematics import (
    _SMPL_PKL_DEFAULT,
    SmplLeftArmFK,
    q_to_arm_aa,
)
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

_PANDA_URDF = Path(__file__).parent / "assets" / "panda" / "panda.urdf"

# SMPL world is Y-up; pybullet is Z-up. Proper rotation (x, y, z) -> (x, -z, y).
_SMPL_TO_PB = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

_ROBOT_BASE_OFFSET = np.array([0.6, -0.15, -0.35], dtype=np.float64)
_ROBOT_BASE_YAW = np.deg2rad(166.0)
_ROBOT_HOME = (
    -0.54193711,
    -1.07197495,
    -2.70514736,
    -2.81591873,
    -1.6951869,
    2.48184051,
    -1.43600207,
)

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480
_CAMERA_DISTANCE = 1.9
_CAMERA_YAW = -30.0
_CAMERA_PITCH = -12.0
_CAMERA_TOP_PITCH = -89.0
_CAMERA_FOV = 60.0


class SimRobotVisualEnv(ExecutionEnv):
    """Kinematic pass-through env rendering a Panda grasping the forearm."""

    def __init__(self) -> None:
        super().__init__()
        self._cid: int = p.connect(p.DIRECT)
        self._history: list[tuple[np.ndarray, np.ndarray]] = []
        self._robot: int = -1
        self._movable_joints: list[int] = []
        self._ee_index: int = -1
        self._spine3_pb: np.ndarray = np.zeros(3, dtype=np.float64)
        self._human_mesh: HumanMeshBody | None = None

    def set_pose_context(
        self,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        body_pos: np.ndarray | None = None,
    ) -> None:
        super().set_pose_context(fk, spine3_pos, spine3_aa, body_pos)
        self._build_scene()

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        q = np.asarray(q_cmd, dtype=np.float64)
        q_robot = self._sync(q)
        self._history.append((q.copy(), q_robot))
        return q_cmd

    def hold(self, q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64)
        q_robot = self._sync(q_arr)
        self._history.append((q_arr.copy(), q_robot))
        return q

    def visualize(self, path: Path | None = None) -> np.ndarray:
        frame = self._frame()
        if path is not None:
            import imageio  # pylint: disable=import-outside-toplevel

            imageio.imwrite(str(path), frame)
        return frame

    def save_video(self, path: str | Path, fps: int = 20) -> None:
        import imageio  # pylint: disable=import-outside-toplevel

        frames: list[np.ndarray] = []
        for q_human, q_robot in self._history:
            self._sync_human(q_human)
            self._set_robot_joints(q_robot)
            frames.append(self._frame())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), np.stack(frames), fps=fps)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        assert self._fk is not None
        spine3 = (
            np.asarray(self._spine3_pos, dtype=np.float64)
            if self._spine3_pos is not None
            else self._fk.tpose_spine3_pos
        )
        self._spine3_pb = _SMPL_TO_PB @ spine3

        self._robot = p.loadURDF(
            str(_PANDA_URDF),
            basePosition=tuple(self._spine3_pb + _ROBOT_BASE_OFFSET),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, _ROBOT_BASE_YAW)),
            useFixedBase=True,
            physicsClientId=self._cid,
        )
        self._movable_joints = [
            j
            for j in range(p.getNumJoints(self._robot, physicsClientId=self._cid))
            if p.getJointInfo(self._robot, j, physicsClientId=self._cid)[2]
            != p.JOINT_FIXED
        ]
        self._ee_index = next(
            j
            for j in range(p.getNumJoints(self._robot, physicsClientId=self._cid))
            if p.getJointInfo(self._robot, j, physicsClientId=self._cid)[12]
            == b"ee_link"
        )
        self._set_robot_joints(np.asarray(_ROBOT_HOME, dtype=np.float64))

        if not _SMPL_PKL_DEFAULT.exists():
            raise FileNotFoundError(f"SMPL_NEUTRAL.pkl not found: {_SMPL_PKL_DEFAULT}")
        body_pos = (
            np.asarray(self._body_pos, dtype=np.float64)
            if self._body_pos is not None
            else self._fk.tpose_all_joints
        )
        self._human_mesh = HumanMeshBody(self._cid, SmplMeshCache(body_pos))

    def _set_robot_joints(self, q_robot: np.ndarray) -> None:
        for joint, value in zip(self._movable_joints, q_robot):
            p.resetJointState(
                self._robot, joint, float(value), physicsClientId=self._cid
            )

    def _sync_human(self, q: np.ndarray) -> None:
        assert self._fk is not None and self._human_mesh is not None
        self._human_mesh.update(
            self._fk.fk(
                q_to_arm_aa(q, self._fk.elbow_hinge_axis),
                self._spine3_pos,
                self._spine3_aa,
            )
        )

    def _sync(self, q: np.ndarray) -> np.ndarray:
        assert self._fk is not None
        self._sync_human(q)
        grasp_pos, grasp_quat = grasp_pose_fk(
            self._fk, q, self._spine3_pos, self._spine3_aa, GRASP_FRACTION
        )
        target_pos = _SMPL_TO_PB @ grasp_pos
        target_quat = (
            Rotation.from_matrix(_SMPL_TO_PB) * Rotation.from_quat(grasp_quat)
        ).as_quat()
        solution = p.calculateInverseKinematics(
            self._robot,
            self._ee_index,
            tuple(target_pos),
            tuple(target_quat),
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self._cid,
        )
        q_robot = np.asarray(solution, dtype=np.float64)
        self._set_robot_joints(q_robot)
        return q_robot

    def _frame(self) -> np.ndarray:
        return np.concatenate(
            (self._camera_frame(_CAMERA_PITCH), self._camera_frame(_CAMERA_TOP_PITCH)),
            axis=0,
        )

    def _camera_frame(self, pitch: float) -> np.ndarray:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=tuple(self._spine3_pb),
            distance=_CAMERA_DISTANCE,
            yaw=_CAMERA_YAW,
            pitch=pitch,
            roll=0.0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            _CAMERA_FOV, _IMAGE_WIDTH / _IMAGE_HEIGHT, 0.1, 10.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            _IMAGE_WIDTH,
            _IMAGE_HEIGHT,
            viewMatrix=view,
            projectionMatrix=proj,
            physicsClientId=self._cid,
        )
        image = np.reshape(
            np.asarray(rgb, dtype=np.uint8), (_IMAGE_HEIGHT, _IMAGE_WIDTH, 4)
        )
        return image[..., :3].copy()
