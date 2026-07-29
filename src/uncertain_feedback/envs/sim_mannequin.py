"""Physics-sim env: a robot physically drags a passive 4-DOF mannequin arm.

Real-world proxy built on the articulated mannequin from
empriselab/limb-manipulation: the robot (Franka Panda by default, or a Kinova
Gen3 7-DOF via ``robot="kinova_gen3"``) starts grasping the mannequin forearm
(fixed constraint) and tracks the commanded grasp pose under position control
with ``stepSimulation`` and gravity. Each step targets the grasp-pose *delta*
between the command and the current read-back, applied to the physical ee pose
(see :meth:`_drive`). The achieved arm configuration is measured back from the
mannequin's link positions (perception proxy) rather than assumed, so
commanded and achieved configs diverge.

Limitations of the 4-DOF proxy: the mannequin shoulder is bolted in place
(no clavicle), so the achieved clavicle block stays ~constant; the closed MPC
loop absorbs this. Mannequin joints are unbounded in the URDF; nominal box
limits are base_x [-1.57, 1.57], base_y [-1.57, 0], base_z [0, 1.57], elbow
[-2.0944, 0].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.grasp import (
    GRASP_FRACTION,
    MeasuredGrasp,
    forearm_frame_fk,
    grasp_pose_fk,
)
from uncertain_feedback.envs.robot_fk import RobotChainFK
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa

_PANDA_URDF = Path(__file__).parent / "assets" / "panda" / "panda.urdf"
_HUMAN_ASSETS = Path(__file__).parent / "assets" / "human"
_MANNEQUIN_URDF = _HUMAN_ASSETS / "left_arm_4dof_continuous.urdf"

# Decorative body parts (visual context only) and their mount offsets in the
# torso frame, from limb-manipulation's HumanSceneConfig. The torso frame is
# x = person's left, z = up; zero-pose limbs hang along -z.
_TORSO_TO_LEFT_ARM = np.array([0.24, 0.048, 0.45], dtype=np.float64)
_BODY_PART_URDFS: tuple[tuple[str, np.ndarray], ...] = (
    ("torso_and_head.urdf", np.zeros(3)),
    ("right_arm_4dof_continuous.urdf", np.array([-0.24, 0.048, 0.45])),
    ("left_leg_4dof_continuous.urdf", np.array([0.11, 0.0, 0.0])),
    ("right_leg_4dof_continuous.urdf", np.array([-0.11, 0.0, 0.0])),
)
_BODY_PART_HOLD_FORCE = 100.0

# SMPL world is Y-up; pybullet is Z-up. Proper rotation (x, y, z) -> (x, -z, y).
_SMPL_TO_PB = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

# Mannequin zero pose = SMPL T-pose: base-local -z (arm) -> world +x (left),
# base-local +y (elbow-bend direction) -> world -y (person faces -y).
_MANNEQUIN_BASE_ROT = np.array(
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)

_ROBOT_BASE_OFFSET = (0.6, -0.15, -0.35)


@dataclass(frozen=True)
class _RobotSpec:
    """Robot model embedded in the scene: URDF, frames, and control limits."""

    urdf: Path
    ee_link: bytes
    home: tuple[float, ...]
    joint_forces: tuple[float, ...]
    base_yaw: float
    # xyzw rotation from the grasp-frame convention (grasp_pose_fk: z =
    # approach, y = finger-closing) to this robot's ee-link frame.
    tool_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Root for resolving package:// mesh URIs in the URDF.
    mesh_search_path: str | None = None


_KINOVA_URDF = (
    Path.home() / "kortex_description/robots/gen3_7dof_no_vision_robotiq_2f_85.urdf"
)

_ROBOT_SPECS: dict[str, _RobotSpec] = {
    "panda": _RobotSpec(
        urdf=_PANDA_URDF,
        ee_link=b"ee_link",
        home=(
            -0.54193711,
            -1.07197495,
            -2.70514736,
            -2.81591873,
            -1.6951869,
            2.48184051,
            -1.43600207,
        ),
        joint_forces=(87.0,) * 7,
        base_yaw=np.deg2rad(166.0),
    ),
    # Robotiq fingers close along the tool-frame x-axis (gripper base mounted
    # at yaw 90 deg), so the tool frame is the grasp frame rolled 90 deg
    # about the approach axis.
    "kinova_gen3": _RobotSpec(
        urdf=_KINOVA_URDF,
        ee_link=b"tool_frame",
        home=(0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57),
        joint_forces=(39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
        base_yaw=np.deg2rad(166.0),
        tool_quat=(0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
        mesh_search_path=str(Path.home()),
    ),
}

_MANNEQUIN_JOINT_DAMPING = 0.2
_CONSTRAINT_MAX_FORCE = 1000.0
_EXECUTE_SUBSTEPS = 48
# Per-execute cap on each robot joint's travel (rad). Keeps the arm on the
# nearby IK branch when the solver jumps to a distant one — the physical
# analogue of a real controller's velocity limit.
_ROBOT_MAX_JOINT_DELTA = 0.2

_GHOST_COLOR = (0.25, 0.75, 0.35, 0.7)
_GHOST_RADIUS = 0.015
_GHOST_JOINT_RADIUS = 0.035

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480
_CAMERA_DISTANCE = 1.9
_CAMERA_YAW = -30.0
_CAMERA_PITCH = -12.0
_CAMERA_TOP_PITCH = -89.0
_CAMERA_FOV = 60.0


def _mannequin_joints(
    shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray
) -> np.ndarray:
    """Mannequin ``(4,)`` [base_x, base_y, base_z, elbow] matching arm points.

    The URDF chain is Rot(-x, q0) . Rot(-y, q1) . Rot(z, q2) at the shoulder,
    then the elbow hinge; zero pose points the arm along base-local -z with
    the forearm bending toward -y for positive elbow angles.

    Three points reproduce the arm in two mirror configurations (the upper-arm
    roll differs by 180 deg, absorbed by the elbow-angle sign). The one with
    ``n = f x u`` matches the SMPL arm's roll (palm faces down for an arm held
    out to the side); ``u x f`` gives the palm-up mirror.
    """

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    u = _MANNEQUIN_BASE_ROT.T @ unit(elbow - shoulder)
    f = _MANNEQUIN_BASE_ROT.T @ unit(wrist - elbow)
    n = np.cross(f, u)
    if np.linalg.norm(n) < 1e-8:
        n = np.cross(np.array([0.0, 0.0, 1.0]), u)
        if np.linalg.norm(n) < 1e-8:
            n = np.cross(np.array([0.0, 1.0, 0.0]), u)
    n = unit(n)
    elbow_angle = float(np.arccos(np.clip(np.dot(u, f), -1.0, 1.0)))
    shoulder_rot = np.column_stack([n, np.cross(n, u), -u])
    ex, ey, ez = Rotation.from_matrix(shoulder_rot).as_euler("XYZ")
    return np.array([-ex, -ey, ez, elbow_angle], dtype=np.float64)


def _quat_z_to(direction: np.ndarray) -> np.ndarray:
    """xyzw quaternion rotating the +z axis onto ``direction`` (unit)."""
    rot, _ = Rotation.align_vectors([direction], [[0.0, 0.0, 1.0]])
    return rot.as_quat()


class SimMannequinEnv(ExecutionEnv):
    """Robot repositioning a passive mannequin arm under pybullet physics."""

    def __init__(
        self,
        robot: str = "panda",
        robot_base_offset: tuple[float, float, float] = _ROBOT_BASE_OFFSET,
        robot_max_joint_delta: float = _ROBOT_MAX_JOINT_DELTA,
        robot_joint_limit_padding: float = 0.0,
        real_mirror_host: str | None = None,
        real_mirror_confirm_start: bool = True,
    ) -> None:
        super().__init__()
        if robot not in _ROBOT_SPECS:
            raise ValueError(
                f"Unknown robot '{robot}'. Available: {sorted(_ROBOT_SPECS)}"
            )
        self._mirror = None
        if real_mirror_host is not None:
            if robot != "kinova_gen3":
                raise ValueError("real_mirror_host requires robot='kinova_gen3'")
            from uncertain_feedback.envs.real_mirror import (  # pylint: disable=import-outside-toplevel
                RealArmMirror,
            )

            self._mirror = RealArmMirror.connect(
                real_mirror_host, confirm_start=real_mirror_confirm_start
            )
        self._spec = _ROBOT_SPECS[robot]
        self._robot_base_offset = np.asarray(robot_base_offset, dtype=np.float64)
        self._robot_max_joint_delta = float(robot_max_joint_delta)
        self._robot_joint_limit_padding = float(robot_joint_limit_padding)
        self._joint_lower: np.ndarray = np.zeros(0, dtype=np.float64)
        self._joint_upper: np.ndarray = np.zeros(0, dtype=np.float64)
        self._cid: int = p.connect(p.DIRECT)
        self._history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._ghost_bodies: list[int] = []
        self._robot: int = -1
        self._movable_joints: list[int] = []
        self._continuous_joints: np.ndarray = np.zeros(0, dtype=bool)
        self._ee_index: int = -1
        self._spine3_pb: np.ndarray = np.zeros(3, dtype=np.float64)
        self._mannequin: int = -1
        self._mannequin_joints_idx: list[int] = []
        self._lower_arm_index: int = -1
        self._hand_index: int = -1
        self._mannequin_base_pb: np.ndarray = np.zeros(3, dtype=np.float64)
        self._spine3_smpl: np.ndarray = np.zeros(3, dtype=np.float64)
        self._collar_smpl: np.ndarray = np.zeros(3, dtype=np.float64)
        self._attached = False
        self._robot_chain: RobotChainFK | None = None

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
        if not self._attached:
            self._attach(q)
        self._drive(q)
        if self._mirror is not None:
            self._mirror.send(self._robot_q())
        self._sync_ghost(q)
        self._record(q)
        return self._read_back_q()

    def hold(self, q: np.ndarray) -> np.ndarray:
        return self.execute(q)

    def robot_fk(self) -> RobotChainFK:
        assert self._robot != -1, "scene not built; set_pose_context first"
        if self._robot_chain is None:
            self._robot_chain = RobotChainFK.from_pybullet(
                self._robot, self._ee_index, self._cid
            )
        return self._robot_chain

    def current_robot_q(self) -> np.ndarray:
        return self._robot_q()

    def robot_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._joint_lower.copy(), self._joint_upper.copy()

    def robot_max_joint_delta(self) -> float:
        return self._robot_max_joint_delta

    def solve_robot_ik_exact(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> np.ndarray | None:
        q_seed = np.asarray(q_seed, dtype=np.float64)
        lower = np.where(self._continuous_joints, q_seed - np.pi, self._joint_lower)
        upper = np.where(self._continuous_joints, q_seed + np.pi, self._joint_upper)
        solution = p.calculateInverseKinematics(
            self._robot,
            self._ee_index,
            tuple(target_pos),
            tuple(target_quat),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=(upper - lower).tolist(),
            restPoses=q_seed.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self._cid,
        )
        return np.clip(np.asarray(solution, dtype=np.float64), lower, upper)

    def current_grasp(self, q: np.ndarray) -> MeasuredGrasp:
        q = np.asarray(q, dtype=np.float64)
        if not self._attached:
            self._attach(q)
        return MeasuredGrasp.measure(*self._forearm_frame_pb(q), *self._ee_pose_pb())

    def execute_robot(self, target: np.ndarray) -> np.ndarray:
        if not self._attached:
            raise RuntimeError(
                "execute_robot before the grasp exists; call current_grasp first"
            )
        target = np.asarray(target, dtype=np.float64)
        q_now = self._robot_q()
        delta = target - q_now
        wrapped = self._continuous_joints
        delta[wrapped] = np.arctan2(np.sin(delta[wrapped]), np.cos(delta[wrapped]))
        # Uniform scaling keeps the commanded ee direction when one joint
        # saturates, unlike the per-joint clip of the IK path.
        largest = float(np.max(np.abs(delta)))
        if largest > self._robot_max_joint_delta:
            delta *= self._robot_max_joint_delta / largest
        target = np.clip(q_now + delta, self._joint_lower, self._joint_upper)
        self._drive_joints(target)
        if self._mirror is not None:
            self._mirror.send(self._robot_q())
        q_meas = self._read_back_q()
        self._sync_ghost(q_meas)
        self._record(q_meas)
        return q_meas

    def visualize(self, path: Path | None = None) -> np.ndarray:
        frame = self._frame()
        if path is not None:
            import imageio  # pylint: disable=import-outside-toplevel

            imageio.imwrite(str(path), frame)
        return frame

    def save_video(self, path: str | Path, fps: int = 20) -> None:
        import imageio  # pylint: disable=import-outside-toplevel

        frames: list[np.ndarray] = []
        for q_robot, q_mannequin, q_cmd in self._history:
            self._set_joints(self._robot, self._movable_joints, q_robot)
            self._set_joints(self._mannequin, self._mannequin_joints_idx, q_mannequin)
            self._sync_ghost(q_cmd)
            frames.append(self._frame())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), np.stack(frames), fps=fps)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        assert self._fk is not None
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._cid)
        spine3 = (
            np.asarray(self._spine3_pos, dtype=np.float64)
            if self._spine3_pos is not None
            else self._fk.tpose_spine3_pos
        )
        self._spine3_pb = _SMPL_TO_PB @ spine3

        if self._spec.mesh_search_path is not None:
            p.setAdditionalSearchPath(
                self._spec.mesh_search_path, physicsClientId=self._cid
            )
        self._robot = p.loadURDF(
            str(self._spec.urdf),
            basePosition=tuple(self._spine3_pb + self._robot_base_offset),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, self._spec.base_yaw)),
            useFixedBase=True,
            physicsClientId=self._cid,
        )
        self._movable_joints = [
            j
            for j in range(p.getNumJoints(self._robot, physicsClientId=self._cid))
            if p.getJointInfo(self._robot, j, physicsClientId=self._cid)[2]
            != p.JOINT_FIXED
        ]
        self._continuous_joints = np.array(
            [
                p.getJointInfo(self._robot, j, physicsClientId=self._cid)[8]
                > p.getJointInfo(self._robot, j, physicsClientId=self._cid)[9]
                for j in self._movable_joints
            ]
        )
        # Soft joint limits: URDF limits shrunk by the padding (matching the
        # real controller's joint_limit_padding_deg); continuous joints stay
        # unbounded. When mirroring, the real controller's enforced limits
        # (narrower than the URDF's) replace the URDF limits.
        if self._mirror is not None:
            from uncertain_feedback.envs.real_mirror import (  # pylint: disable=import-outside-toplevel
                GEN3_JOINT_LIMITS,
            )

            lower, upper = (np.array(x) for x in zip(*GEN3_JOINT_LIMITS))
        else:
            lower = np.array(
                [
                    p.getJointInfo(self._robot, j, physicsClientId=self._cid)[8]
                    for j in self._movable_joints
                ]
            )
            upper = np.array(
                [
                    p.getJointInfo(self._robot, j, physicsClientId=self._cid)[9]
                    for j in self._movable_joints
                ]
            )
        self._joint_lower = np.where(
            self._continuous_joints, -np.inf, lower + self._robot_joint_limit_padding
        )
        self._joint_upper = np.where(
            self._continuous_joints, np.inf, upper - self._robot_joint_limit_padding
        )
        self._ee_index = next(
            j
            for j in range(p.getNumJoints(self._robot, physicsClientId=self._cid))
            if p.getJointInfo(self._robot, j, physicsClientId=self._cid)[12]
            == self._spec.ee_link
        )
        self._set_joints(
            self._robot, self._movable_joints, np.asarray(self._spec.home, np.float64)
        )

    def _set_joints(self, body: int, joints: list[int], values: np.ndarray) -> None:
        for joint, value in zip(joints, values):
            p.resetJointState(body, joint, float(value), physicsClientId=self._cid)

    # ------------------------------------------------------------------
    # Grasp acquisition
    # ------------------------------------------------------------------

    def _attach(self, q0: np.ndarray) -> None:
        assert self._fk is not None
        positions = self._fk.fk(
            q_to_arm_aa(q0, self._fk.elbow_hinge_axis),
            self._spine3_pos,
            self._spine3_aa,
        )
        self._spine3_smpl = positions[0].copy()
        self._collar_smpl = positions[1].copy()
        shoulder, elbow, wrist = (_SMPL_TO_PB @ pos for pos in positions[2:5])
        self._mannequin_base_pb = shoulder

        self._mannequin = p.loadURDF(
            str(_MANNEQUIN_URDF),
            basePosition=tuple(shoulder),
            baseOrientation=tuple(Rotation.from_matrix(_MANNEQUIN_BASE_ROT).as_quat()),
            useFixedBase=True,
            physicsClientId=self._cid,
        )
        n_joints = p.getNumJoints(self._mannequin, physicsClientId=self._cid)
        infos = [
            p.getJointInfo(self._mannequin, j, physicsClientId=self._cid)
            for j in range(n_joints)
        ]
        self._mannequin_joints_idx = [
            info[0] for info in infos if info[2] != p.JOINT_FIXED
        ]
        self._lower_arm_index = next(
            info[0] for info in infos if info[1] == b"upper_to_lower"
        )
        self._hand_index = next(
            info[0] for info in infos if info[1] == b"lower_to_hand_x"
        )
        self._set_joints(
            self._mannequin,
            self._mannequin_joints_idx,
            _mannequin_joints(shoulder, elbow, wrist),
        )
        for joint in self._mannequin_joints_idx:
            p.changeDynamics(
                self._mannequin,
                joint,
                jointDamping=_MANNEQUIN_JOINT_DAMPING,
                physicsClientId=self._cid,
            )
        for link in range(-1, n_joints):
            p.setCollisionFilterGroupMask(
                self._mannequin, link, 0, 0, physicsClientId=self._cid
            )
        self._load_body_parts(shoulder)

        self._set_joints(
            self._robot,
            self._movable_joints,
            np.clip(self._ik(q0), self._joint_lower, self._joint_upper),
        )

        # createConstraint frames are relative to link CoM frames, so measure
        # CoM poses ([0:2]), not URDF link frames ([4:6]).
        ee_pos, ee_orn = p.getLinkState(
            self._robot,
            self._ee_index,
            computeForwardKinematics=True,
            physicsClientId=self._cid,
        )[0:2]
        arm_pos, arm_orn = p.getLinkState(
            self._mannequin,
            self._lower_arm_index,
            computeForwardKinematics=True,
            physicsClientId=self._cid,
        )[0:2]
        rel_pos, rel_orn = p.multiplyTransforms(
            *p.invertTransform(arm_pos, arm_orn), ee_pos, ee_orn
        )
        constraint = p.createConstraint(
            self._robot,
            self._ee_index,
            self._mannequin,
            self._lower_arm_index,
            jointType=p.JOINT_FIXED,
            jointAxis=(0.0, 0.0, 0.0),
            parentFramePosition=(0.0, 0.0, 0.0),
            parentFrameOrientation=(0.0, 0.0, 0.0, 1.0),
            childFramePosition=rel_pos,
            childFrameOrientation=rel_orn,
            physicsClientId=self._cid,
        )
        p.changeConstraint(
            constraint, maxForce=_CONSTRAINT_MAX_FORCE, physicsClientId=self._cid
        )
        p.setJointMotorControlArray(
            self._mannequin,
            self._mannequin_joints_idx,
            p.VELOCITY_CONTROL,
            forces=[0.0] * len(self._mannequin_joints_idx),
            physicsClientId=self._cid,
        )
        if self._mirror is not None:
            self._mirror.start(self._robot_q())
        self._attached = True

    def _load_body_parts(self, shoulder: np.ndarray) -> None:
        """Load the torso/head and remaining limbs as static visual context."""
        torso_pos = shoulder - _TORSO_TO_LEFT_ARM
        for urdf_name, offset in _BODY_PART_URDFS:
            body = p.loadURDF(
                str(_HUMAN_ASSETS / urdf_name),
                basePosition=tuple(torso_pos + offset),
                useFixedBase=True,
                physicsClientId=self._cid,
            )
            n_joints = p.getNumJoints(body, physicsClientId=self._cid)
            movable = [
                j
                for j in range(n_joints)
                if p.getJointInfo(body, j, physicsClientId=self._cid)[2]
                != p.JOINT_FIXED
            ]
            if movable:
                p.setJointMotorControlArray(
                    body,
                    movable,
                    p.POSITION_CONTROL,
                    targetPositions=[0.0] * len(movable),
                    forces=[_BODY_PART_HOLD_FORCE] * len(movable),
                    physicsClientId=self._cid,
                )
            for link in range(-1, n_joints):
                p.setCollisionFilterGroupMask(
                    body, link, 0, 0, physicsClientId=self._cid
                )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _grasp_pose_pb(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        assert self._fk is not None
        grasp_pos, grasp_quat = grasp_pose_fk(
            self._fk, q, self._spine3_pos, self._spine3_aa, GRASP_FRACTION
        )
        return (
            _SMPL_TO_PB @ grasp_pos,
            Rotation.from_matrix(_SMPL_TO_PB) * Rotation.from_quat(grasp_quat),
        )

    def _forearm_frame_pb(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """The frame a measured grasp is expressed in, in pybullet coords."""
        assert self._fk is not None
        pos, rot = forearm_frame_fk(self._fk, q, self._spine3_pos, self._spine3_aa)
        return _SMPL_TO_PB @ pos, Rotation.from_matrix(_SMPL_TO_PB) * rot

    def _ee_pose_pb(self) -> tuple[np.ndarray, Rotation]:
        """The robot's current end-effector pose."""
        ee_pos, ee_orn = p.getLinkState(
            self._robot,
            self._ee_index,
            computeForwardKinematics=True,
            physicsClientId=self._cid,
        )[4:6]
        return np.asarray(ee_pos, dtype=np.float64), Rotation.from_quat(ee_orn)

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        solution = self.solve_robot_ik_exact(target_pos, target_quat, self._robot_q())
        assert solution is not None
        return solution

    def _ik(self, q: np.ndarray) -> np.ndarray:
        target_pos, target_rot = self._grasp_pose_pb(q)
        target_rot = target_rot * Rotation.from_quat(self._spec.tool_quat)
        return self._solve_ik(target_pos, target_rot.as_quat())

    def _drive(self, q_cmd: np.ndarray) -> None:
        # Target the grasp-pose *delta* (model pose of q_cmd relative to the
        # model pose of the current read-back) applied to the physical ee pose.
        # The SMPL FK and the mannequin disagree on segment lengths, so an
        # absolute grasp-pose target re-anchors on the read-back bias every
        # step and integrates it into unbounded drift; the relative target
        # makes executing the read-back q a no-op.
        ref_pos, ref_rot = self._grasp_pose_pb(self._read_back_q())
        cmd_pos, cmd_rot = self._grasp_pose_pb(q_cmd)
        ee_pos, ee_orn = p.getLinkState(
            self._robot,
            self._ee_index,
            computeForwardKinematics=True,
            physicsClientId=self._cid,
        )[4:6]
        target_pos = np.asarray(ee_pos) + (cmd_pos - ref_pos)
        # Orientation delta: minimal rotation between the read-back and
        # commanded forearm directions (the grasp-frame x-axis). The full
        # grasp frame flips its up-reference near a vertical forearm
        # (grasp_pose_fk), which would command spurious ~1 rad rotations;
        # forearm roll is unobservable from the read-back anyway.
        delta_rot, _ = Rotation.align_vectors(
            [cmd_rot.apply([1.0, 0.0, 0.0])], [ref_rot.apply([1.0, 0.0, 0.0])]
        )
        target_rot = delta_rot * Rotation.from_quat(ee_orn)
        q_now = self._robot_q()
        delta = self._solve_ik(target_pos, target_rot.as_quat()) - q_now
        # Continuous joints have no limits to anchor the IK solution, so it
        # may come back unwound by full turns; take the short way around.
        wrapped = self._continuous_joints
        delta[wrapped] = np.arctan2(np.sin(delta[wrapped]), np.cos(delta[wrapped]))
        delta = np.clip(
            delta, -self._robot_max_joint_delta, self._robot_max_joint_delta
        )
        delta = np.clip(q_now + delta, self._joint_lower, self._joint_upper) - q_now
        self._drive_joints(q_now + delta)

    def _drive_joints(self, target: np.ndarray) -> None:
        """Ramp the robot to a joint target under position control."""
        q_now = self._robot_q()
        delta = target - q_now
        for k in range(1, _EXECUTE_SUBSTEPS + 1):
            step_target = q_now + delta * (k / _EXECUTE_SUBSTEPS)
            p.setJointMotorControlArray(
                self._robot,
                self._movable_joints,
                p.POSITION_CONTROL,
                targetPositions=step_target.tolist(),
                forces=list(self._spec.joint_forces),
                physicsClientId=self._cid,
            )
            p.stepSimulation(physicsClientId=self._cid)

    def _robot_q(self) -> np.ndarray:
        return np.array(
            [
                p.getJointState(self._robot, j, physicsClientId=self._cid)[0]
                for j in self._movable_joints
            ]
        )

    def _read_back_q(self) -> np.ndarray:
        assert self._fk is not None
        elbow_pb = np.asarray(
            p.getLinkState(
                self._mannequin,
                self._lower_arm_index,
                computeForwardKinematics=True,
                physicsClientId=self._cid,
            )[4]
        )
        wrist_pb = np.asarray(
            p.getLinkState(
                self._mannequin,
                self._hand_index,
                computeForwardKinematics=True,
                physicsClientId=self._cid,
            )[4]
        )
        positions = np.stack(
            [
                self._spine3_smpl,
                self._collar_smpl,
                _SMPL_TO_PB.T @ self._mannequin_base_pb,
                _SMPL_TO_PB.T @ elbow_pb,
                _SMPL_TO_PB.T @ wrist_pb,
            ]
        )
        arm_aa = self._fk.arm_aa_from_positions(positions, self._spine3_aa)
        return self._fk.arm_aa_to_q(arm_aa, self._spine3_aa)

    def _record(self, q_cmd: np.ndarray) -> None:
        q_robot = self._robot_q()
        q_mannequin = np.array(
            [
                p.getJointState(self._mannequin, j, physicsClientId=self._cid)[0]
                for j in self._mannequin_joints_idx
            ]
        )
        self._history.append((q_robot, q_mannequin, q_cmd.copy()))

    def _sync_ghost(self, q_cmd: np.ndarray) -> None:
        """Pose the commanded-arm ghost: capsules along shoulder→elbow→wrist."""
        assert self._fk is not None
        positions = self._fk.fk(
            q_to_arm_aa(q_cmd, self._fk.elbow_hinge_axis),
            self._spine3_pos,
            self._spine3_aa,
        )
        points = [_SMPL_TO_PB @ pos for pos in positions[2:5]]
        segments = list(zip(points[:-1], points[1:]))
        if not self._ghost_bodies:
            for start, end in segments:
                vis = p.createVisualShape(
                    p.GEOM_CAPSULE,
                    radius=_GHOST_RADIUS,
                    length=float(np.linalg.norm(end - start)),
                    rgbaColor=_GHOST_COLOR,
                    physicsClientId=self._cid,
                )
                self._ghost_bodies.append(
                    p.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=vis,
                        physicsClientId=self._cid,
                    )
                )
            for _ in points[1:]:
                vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=_GHOST_JOINT_RADIUS,
                    rgbaColor=_GHOST_COLOR,
                    physicsClientId=self._cid,
                )
                self._ghost_bodies.append(
                    p.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=vis,
                        physicsClientId=self._cid,
                    )
                )
        for body, (start, end) in zip(self._ghost_bodies, segments):
            direction = end - start
            quat = _quat_z_to(direction / np.linalg.norm(direction))
            p.resetBasePositionAndOrientation(
                body, tuple((start + end) / 2.0), tuple(quat), physicsClientId=self._cid
            )
        for body, point in zip(self._ghost_bodies[len(segments) :], points[1:]):
            p.resetBasePositionAndOrientation(
                body, tuple(point), (0.0, 0.0, 0.0, 1.0), physicsClientId=self._cid
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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
