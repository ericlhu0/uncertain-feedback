"""Real-world env: a mocap-measured human arm moved by the real Kinova Gen3.

The closed-loop counterpart to
:class:`~uncertain_feedback.envs.sim_mannequin.SimMannequinEnv`. There the robot
drags a simulated mannequin under physics and the real arm only shadows the sim;
here the human arm state is *measured* — OptiTrack rigid bodies on the collar,
shoulder, elbow, and wrist are streamed in over NatNet and converted to the
planner's ``(7,)`` configuration — so the MPC closes the loop on the actual
person. That includes the configuration the run *starts* from
(:meth:`RealEnv.initial_q`), so the person does not have to match the config's
start pose. A right-collar body (read at calibration only) makes the
registration yaw — the person's facing — measurable rather than assumed, and
the left collar fixes *where* the person is: the torso anchor is the measured
collar, not the config pose's, so the scene tracks a person who sits somewhere
different between runs (see :mod:`uncertain_feedback.mocap.registration`). The
run must therefore read the anchor back through :meth:`ExecutionEnv.pose_context`
after :meth:`initial_q`, since every Cartesian goal is relative to it.

The grasp is *measured*, not assumed — and re-measured every step. The operator
establishes it before the run (gripper closed on the forearm), and from then on
the gripper's pose relative to the measured forearm is read off the real arm's
own configuration at each step, because the real grasp shifts as the trajectory
runs. The simulated envs instead place their robot at the nominal grasp of
:func:`grasp_pose_fk` and keep it.

Pybullet runs in ``DIRECT`` for inverse kinematics only: no ``stepSimulation``,
no cameras, no ghost bodies (the same no-physics use as
:mod:`~uncertain_feedback.envs.sim_robot_visual`). Rollouts are inspected
through :meth:`ArmVisualizer.render_rollout_video`.

With ``real_mirror_host=None`` the loop runs on the live person and solves IK
but sends nothing to the arm — the verification mode to use before anything
moves. There is no real gripper to measure then, so the calibration falls back
to the nominal grasp.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pybullet as p
from scipy.optimize import OptimizeResult, least_squares
from scipy.spatial.transform import Rotation
from ssik.prebuilt import gen3_ik

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.grasp import (
    GRASP_FRACTION,
    MeasuredGrasp,
    forearm_frame_fk,
    grasp_pose_fk,
)
from uncertain_feedback.envs.human_mesh import (
    BODY_XRAY_COLOR,
    GOAL_COLOR,
    ArmSkeletonBody,
    HumanMeshBody,
)
from uncertain_feedback.envs.sim_mannequin import _ROBOT_SPECS, _SMPL_TO_PB
from uncertain_feedback.mocap.natnet import (
    NatNetReceiver,
    RigidBodyPose,
    require_fresh,
)
from uncertain_feedback.mocap.registration import ArmRegistration, arm_keypoints
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

# Rigid-body roles the config must supply streaming ids for. The right-collar
# body is what makes the registration yaw measurable rather than assumed (see
# `mocap.registration`), so it is required, not optional — but it is read only
# at calibration, so it is not part of the live arm chain.
_ARM_KEYS = ("collar", "shoulder", "elbow", "wrist")
_BASE_KEY = "robot_base"
_COLLAR_RIGHT_KEY = "collar_right"
# The person must be tracked before the scene can be built at all, so allow a
# generous window for markers to come into view.
_CALIBRATION_TIMEOUT_S = 20.0
# Re-reading tracking after the scene build should succeed on the first poll;
# anything longer means markers were lost while the scene came up.
_RESYNC_TIMEOUT_S = 2.0
# How the IK trades gripper orientation error against position error when the
# exact pose is infeasible (metres of position per radian of orientation).
# Small, so the gripper stays on the forearm and the wrist absorbs the error —
# the same compromise the physical grasp forces.
_IK_ORIENTATION_WEIGHT_M_PER_RAD = 0.02
# How many analytical branches to refine from. They are ranked nearest-first, so
# the ones past the first few are branches the arm would have to cross itself to
# reach.
_IK_SEEDS = 2
# Live-view camera, framed on the person's collar (where the registration is
# anchored) with the robot in shot.
_LIVE_CAMERA_DISTANCE = 2.0
_LIVE_CAMERA_YAW = -30.0
_LIVE_CAMERA_PITCH = -12.0
# The link `gen3_ik` solves for, which is not the one the grasp is measured at.
_SSIK_EE_LINK = b"end_effector_link"


def _pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """A ``(4, 4)`` homogeneous transform from a position and an xyzw quat."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


class RealEnv(ExecutionEnv):
    """MPC execution against a mocap-tracked person and the real Gen3."""

    def __init__(
        self,
        mocap_host: str,
        mocap_rigid_bodies: dict[str, int],
        robot: str = "kinova_gen3",
        robot_max_joint_delta: float = 0.01,
        robot_joint_limit_padding: float = 0.27,
        real_mirror_host: str | None = None,
        real_mirror_confirm_start: bool = True,
        mocap_hold_timeout: float = 0.5,
        live_view: bool = False,
        live_view_fps: float = 5.0,
        preview_plan: bool = True,
    ) -> None:
        super().__init__()
        # The IK is `ssik`'s Gen3 artifact, baked against that arm's geometry.
        if robot != "kinova_gen3":
            raise ValueError(f"RealEnv solves IK for kinova_gen3 only, not '{robot}'")
        missing = {_BASE_KEY, _COLLAR_RIGHT_KEY, *_ARM_KEYS} - set(mocap_rigid_bodies)
        if missing:
            raise ValueError(f"mocap_rigid_bodies is missing {sorted(missing)}")
        self._spec = _ROBOT_SPECS[robot]
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
        self._body_ids = {key: int(v) for key, v in mocap_rigid_bodies.items()}
        self._receiver = NatNetReceiver.connect(mocap_host)
        self._hold_timeout = float(mocap_hold_timeout)
        self._robot_max_joint_delta = float(robot_max_joint_delta)
        self._robot_joint_limit_padding = float(robot_joint_limit_padding)
        self._live_view = bool(live_view)
        self._preview_plan = bool(preview_plan)
        self._live_mesh_period = 1.0 / live_view_fps if live_view_fps > 0.0 else 0.0
        self._last_live_mesh_s = 0.0
        self._cid: int = p.connect(p.GUI if self._live_view else p.DIRECT)
        if self._live_view:
            # The scene is the whole point of the window; pybullet's parameter
            # panes and preview tiles only take space away from it.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._cid)
        self._mesh_cache: SmplMeshCache | None = None
        self._human_mesh: HumanMeshBody | None = None
        self._arm_skeleton: ArmSkeletonBody | None = None
        self._goal_mesh: HumanMeshBody | None = None
        self._goal_q: np.ndarray | None = None
        self._registration: ArmRegistration | None = None
        self._robot: int = -1
        self._movable_joints: list[int] = []
        self._continuous_joints: np.ndarray = np.zeros(0, dtype=bool)
        self._joint_lower: np.ndarray = np.zeros(0, dtype=np.float64)
        self._joint_upper: np.ndarray = np.zeros(0, dtype=np.float64)
        self._ee_index: int = -1
        self._world_to_base: np.ndarray = np.eye(4)
        self._tool_to_ee: np.ndarray = np.eye(4)
        self._last_q: np.ndarray = np.zeros(0, dtype=np.float64)
        self._last_valid_s: float = 0.0
        self._grasp: MeasuredGrasp | None = None
        self._measured: list[np.ndarray] = []

    def initial_q(self, q_nominal: np.ndarray) -> np.ndarray:
        """Register against the person and report their *measured* arm config.

        Talks to mocap only — the grasp is measured on the first :meth:`execute`,
        once the planner (which may load a diffusion model) is built and about to
        command something.

        ``q_nominal`` is ignored: the registration yaw comes from the measured
        collar-to-collar axis (see :meth:`ArmRegistration.calibrate`), so every
        slot of the returned configuration — clavicle included — is the
        person's.
        """
        self._register()
        return self._last_q.copy()

    def show_goal(self, q_goal: np.ndarray) -> None:
        """Draw the goal configuration as a translucent green ghost body.

        Costs one mesh build, so call it when the goal changes, not per step. It
        is drawn in the same registered frame as the person, so how far the
        person's own mesh is from the ghost *is* the remaining error.
        """
        self._goal_q = np.asarray(q_goal, dtype=np.float64)
        if self._mesh_cache is not None:
            self._draw_goal_ghost()

    def preview(self, plan: Callable[[], np.ndarray]) -> bool:
        """Play the whole planned trajectory in the live scene, then ask.

        Everything in the window is the measured world: the person's mesh at the
        registered anchor on their own segment lengths, the Gen3 at its measured
        base, and the gripper carried by the grasp measured off the real arm — so
        the operator sees the trajectory that is about to be run on the person,
        against the geometry it will be run against, while nothing is moving. A
        registration error, an unreachable stretch, or a plan that drags the arm
        somewhere unacceptable is visible here and answering ``n`` ends the run
        before the first command.

        The plan comes from a kinematic rollout, so it assumes the arm tracks
        each command exactly; the real run closes the loop on mocap and will
        drift from it. The mesh is re-posed every frame (~140 ms each, so a long
        plan takes a minute), which starves the mocap receive thread — hence the
        :meth:`_resync` before handing back.

        The grasp error over the plan is reported before the prompt, since how
        far the gripper strays from the grasp is the part of "can the arm
        actually do this" that is hard to see in the window.
        """
        if not self._preview_plan:
            return True
        if self._registration is None:
            self._register()
        if not self._live_view:
            print(
                "[real] plan preview skipped: needs live_view for a window to "
                "draw it in",
                flush=True,
            )
            return True
        print("[real] planning the trajectory to preview", flush=True)
        traj = np.asarray(plan(), dtype=np.float64)
        q_meas = self._read_back_q()
        self._capture_grasp(q_meas)
        print(
            f"[real] previewing {len(traj)} planned steps in the live view "
            "(full mesh, ~140 ms/frame)",
            flush=True,
        )
        errors = []
        for q in traj:
            self._drive(q, mirror=False)
            errors.append(self._grasp_error(q))
            self._pose_arm_skeleton(q)
            self._pose_human_mesh(q)
        self._print_grasp_error(np.asarray(errors))
        answer = input("[real] preview done. Enter = run on the robot, n = abort: ")
        # The grasp measured above was for the animation only. Dropping it sends
        # the first real step back through `_establish_grasp`, whose handover to
        # the controller's position mode is first-step work the preview must not
        # do or skip.
        self._grasp = None
        self._set_joints(self._current_q())
        self._resync()
        self._pose_arm_skeleton(self._last_q)
        self._pose_human_mesh(self._last_q)
        return not answer.strip().lower().startswith("n")

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        q = np.asarray(q_cmd, dtype=np.float64)
        if self._registration is None:
            self._register()
        q_meas = self._read_back_q()
        if self._grasp is None:
            self._establish_grasp(q_meas)
        else:
            self._measure_grasp(q_meas)
        self._drive(q)
        if self._human_mesh is not None:
            self._update_live_view(q_meas)
        self._measured.append(q_meas.copy())
        return q_meas

    def hold(self, q: np.ndarray) -> np.ndarray:
        return self.execute(q)

    def visualize(self, path: Path | None = None) -> np.ndarray:
        """Render the last measured arm configuration."""
        from uncertain_feedback.envs.kinematic import (  # pylint: disable=import-outside-toplevel
            KinematicEnv,
        )

        assert self._fk is not None
        render_env = KinematicEnv()
        render_env.set_pose_context(
            self._fk, self._spine3_pos, self._spine3_aa, self._body_pos
        )
        render_env.execute(self._measured[-1])
        return render_env.visualize(path)

    def save_video(self, path: str | Path, fps: int = 20) -> None:
        """Render the *measured* arm trajectory — what the person actually did."""
        from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
            ArmVisualizer,
        )

        assert self._fk is not None
        rollout = q_to_arm_aa(np.stack(self._measured), self._fk.elbow_hinge_axis)
        ArmVisualizer(self._fk).render_rollout_video(
            rollout,
            path,
            spine3_pos=self._spine3_pos,
            spine3_aa=self._spine3_aa,
            body_pos=self._body_pos,
            fps=fps,
        )

    # ------------------------------------------------------------------
    # Calibration and scene
    # ------------------------------------------------------------------

    def _register(self) -> None:
        """Solve the mocap registration, build the scene, and measure the arm.

        Runs from :meth:`initial_q` before planning, because the robot base
        pose comes from mocap and the planner's start configuration is the
        measured one. Falls back to the first :meth:`execute` for callers that
        plan without asking for a start configuration.
        """
        assert self._fk is not None
        wanted = [
            self._body_ids[key] for key in (_BASE_KEY, _COLLAR_RIGHT_KEY, *_ARM_KEYS)
        ]
        print(
            f"[real] waiting for mocap rigid bodies {wanted} "
            f"(up to {_CALIBRATION_TIMEOUT_S:.0f}s)",
            flush=True,
        )
        bodies = self._receiver.wait_for(wanted, _CALIBRATION_TIMEOUT_S)
        base = bodies[self._body_ids[_BASE_KEY]]
        collar_right = bodies[self._body_ids[_COLLAR_RIGHT_KEY]]
        keypoints = self._arm_keypoints(bodies)
        assert keypoints is not None
        # Calibrate the skeleton to the person before anything reads lengths
        # off it: the fk is the same instance the planner and the grasp FK use,
        # so scaling it here puts the whole run — measured q, Cartesian costs,
        # forearm frame — on the person's segment lengths instead of SMPL
        # neutral's. The lengths are marker distances from this one frame;
        # segments are rigid, so there is nothing to track live.
        collar, shoulder, elbow, wrist = keypoints
        self._fk.scale_arm_lengths(
            float(np.linalg.norm(shoulder - collar)),
            float(np.linalg.norm(elbow - shoulder)),
            float(np.linalg.norm(wrist - elbow)),
        )
        self._registration = ArmRegistration.calibrate(
            fk=self._fk,
            spine3_pos=self._spine3_pos,
            spine3_aa=self._spine3_aa,
            base_position=base.position,
            base_orientation=base.orientation,
            collar_mocap=keypoints[0],
            collar_right_mocap=collar_right.position,
        )
        # The registration puts the person on their measured collar, so the
        # config's torso anchor is superseded. Everything downstream — the
        # grasp FK, the live mesh, and (through `pose_context`) the planner's
        # spine3-relative goals — has to use the measured one.
        self._spine3_pos = self._registration.spine3_smpl
        self._body_pos = (
            self._body_pos if self._body_pos is not None else self._fk.tpose_all_joints
        ) + self._registration.translation_smpl
        self._last_q = self._registration.q_from_keypoints(*keypoints)
        self._last_valid_s = time.monotonic()
        self._build_scene()
        if self._live_view:
            # Up before the first command, so the registration can be eyeballed
            # while nothing is moving.
            self._start_live_view()
            self._update_live_view(self._last_q)
        self._resync()

    def _resync(self) -> None:
        """Re-read tracking once the scene is built, and re-stamp its age.

        Building the scene takes seconds — loading the SMPL body model, and in
        the live view the GUI meshes — and holds the GIL through much of it, so
        the receive thread may not have run since the registration frame. The
        first step would then find the last valid pose ``mocap_hold_timeout``
        old and halt on a perfectly good stream. Polling here yields to that
        thread and refreshes the pose the run starts from.
        """
        assert self._registration is not None
        bodies = self._receiver.wait_for(
            [self._body_ids[key] for key in _ARM_KEYS], _RESYNC_TIMEOUT_S
        )
        keypoints = self._arm_keypoints(bodies)
        assert keypoints is not None
        self._last_q = self._registration.q_from_keypoints(*keypoints)
        self._last_valid_s = time.monotonic()

    def _establish_grasp(self, q_meas: np.ndarray) -> None:
        """First step: read the grasp the operator already took, then hand over.

        The gripper is already closed on the forearm when the run starts, so the
        grasp is a measurement, not a choice. Nothing moves. Only the handover to
        position mode is first-step work — the transform itself is re-read every
        step by :meth:`_measure_grasp`.
        """
        self._capture_grasp(q_meas)
        if self._mirror is not None:
            self._mirror.start_from_grasp()

    def _capture_grasp(self, q_meas: np.ndarray) -> None:
        """Measure the grasp without the handover — also what a preview needs."""
        if self._mirror is None:
            self._pose_robot_at_nominal_grasp(q_meas)
        self._measure_grasp(q_meas)

    def _measure_grasp(self, q_meas: np.ndarray) -> None:
        """Re-read the gripper-on-forearm transform, every step.

        The grasp is not rigid in practice: over a trajectory the forearm turns
        and slides a little inside the fingers, and sleeve and skin move over the
        bone. A transform captured once therefore goes stale, and it is the lever
        arm :meth:`_drive` swings the gripper on — a stale one converts the
        commanded forearm motion into the wrong gripper motion, worst where the
        offset is largest. So it is measured afresh here: the real joint angles
        give the gripper pose, ``q_meas`` gives the forearm pose, and their
        relative transform is this step's grasp. Nothing moves.

        Re-measuring absorbs the SMPL-FK-vs-real bone-length disagreement each
        step instead of letting it accumulate, which is what lets :meth:`_drive`
        command an absolute target.
        """
        self._set_joints(self._current_q())
        self._grasp = MeasuredGrasp.measure(
            *self._forearm_frame_pb(q_meas), *self._ee_pose_pb()
        )

    def _pose_robot_at_nominal_grasp(self, q_meas: np.ndarray) -> None:
        """Dry run only: there is no real gripper pose to measure.

        Placing the IK robot at :func:`grasp_pose_fk`'s nominal grasp on the
        measured forearm gives the calibration something to read, so the
        mocap-only run exercises the same code path as the live one. The solve is
        bounded by the controller's joint limits, so a nominal grasp the robot
        cannot reach from where it stands is missed rather than faked; nothing
        moves.
        """
        target_pos, target_rot = self._nominal_grasp_pose_pb(q_meas)
        solution = self._solve_ik(
            target_pos,
            (target_rot * Rotation.from_quat(self._spec.tool_quat)).as_quat(),
        )
        self._set_joints(np.clip(solution, self._joint_lower, self._joint_upper))

    def _build_scene(self) -> None:
        """Load the robot at the measured base pose.

        The yaw comes from the registration, not from ``_RobotSpec.base_yaw``
        (a sim scene choice): the scene and the measured bone directions have to
        share one frame.
        """
        assert self._registration is not None
        if self._spec.mesh_search_path is not None:
            p.setAdditionalSearchPath(
                self._spec.mesh_search_path, physicsClientId=self._cid
            )
        self._robot = p.loadURDF(
            str(self._spec.urdf),
            basePosition=tuple(self._registration.base_pb),
            baseOrientation=p.getQuaternionFromEuler(
                (0.0, 0.0, self._registration.robot_base_yaw)
            ),
            useFixedBase=True,
            physicsClientId=self._cid,
        )
        infos = [
            p.getJointInfo(self._robot, j, physicsClientId=self._cid)
            for j in range(p.getNumJoints(self._robot, physicsClientId=self._cid))
        ]
        self._movable_joints = [info[0] for info in infos if info[2] != p.JOINT_FIXED]
        movable = [infos[j] for j in self._movable_joints]
        self._continuous_joints = np.array([info[8] > info[9] for info in movable])
        # The real controller enforces limits narrower than the kortex URDF's,
        # so plan against those even when no command is being mirrored — the
        # mocap-only dry run must produce the same solutions as the live one.
        from uncertain_feedback.envs.real_mirror import (  # pylint: disable=import-outside-toplevel
            GEN3_JOINT_LIMITS,
        )

        lower, upper = (np.array(x) for x in zip(*GEN3_JOINT_LIMITS))
        self._joint_lower = np.where(
            self._continuous_joints, -np.inf, lower + self._robot_joint_limit_padding
        )
        self._joint_upper = np.where(
            self._continuous_joints, np.inf, upper - self._robot_joint_limit_padding
        )
        self._ee_index = next(
            info[0] for info in infos if info[12] == self._spec.ee_link
        )
        # `gen3_ik` solves in the robot's own base frame while every target is
        # built in the pybullet world, so cache the inverse of where the robot
        # was just measured to stand.
        self._world_to_base = np.linalg.inv(
            _pose_matrix(
                self._registration.base_pb,
                p.getQuaternionFromEuler((0.0, 0.0, self._registration.robot_base_yaw)),
            )
        )
        # It also solves for `end_effector_link`, while every pose here is the
        # `tool_frame` between the gripper's fingers that the grasp is measured
        # at — 12 cm apart on this arm. Read the offset off the URDF, since it is
        # a property of the loaded gripper rather than of the arm.
        ee_frame_index = next(info[0] for info in infos if info[12] == _SSIK_EE_LINK)
        self._tool_to_ee = np.linalg.inv(self._link_pose_pb(self._ee_index)) @ (
            self._link_pose_pb(ee_frame_index)
        )
        self._set_joints(np.asarray(self._spec.home, dtype=np.float64))

    def _link_pose_pb(self, link_index: int) -> np.ndarray:
        """A link's world pose as a ``(4, 4)`` transform."""
        return _pose_matrix(
            *p.getLinkState(
                self._robot,
                link_index,
                computeForwardKinematics=True,
                physicsClientId=self._cid,
            )[4:6]
        )

    def _start_live_view(self) -> None:
        """Open the live scene: the measured person as a mesh, plus the robot.

        The robot's meshes come from the URDF already loaded for IK, so only the
        human needs adding. Both are drawn in the registered pybullet frame, so
        the window shows the geometry the MPC is actually solving against — a
        registration that put the person or the robot in the wrong place is
        visible here before the numbers say anything.

        The mesh is shaped to the arm the planner is planning for: only bone
        directions survive into a mesh pose, so a neutral-shape body would draw a
        neutral-length arm while the FK is on the person's measured segments, and
        the drawn wrist would miss the wrist the Cartesian cost measures. Handing
        the calibrated lengths to the cache fits ``betas`` to them instead.
        """
        assert self._fk is not None and self._body_pos is not None
        assert self._registration is not None
        clavicle, upper_arm, forearm = np.linalg.norm(
            np.diff(self._fk.tpose_joints, axis=0), axis=1
        )[1:]
        self._mesh_cache = SmplMeshCache(
            np.asarray(self._body_pos, dtype=np.float64),
            arm_lengths=(float(clavicle), float(upper_arm), float(forearm)),
        )
        self._human_mesh = HumanMeshBody(self._cid, self._mesh_cache, BODY_XRAY_COLOR)
        self._arm_skeleton = ArmSkeletonBody(self._cid)
        if self._goal_q is not None:
            self._draw_goal_ghost()
        p.resetDebugVisualizerCamera(
            cameraDistance=_LIVE_CAMERA_DISTANCE,
            cameraYaw=_LIVE_CAMERA_YAW,
            cameraPitch=_LIVE_CAMERA_PITCH,
            cameraTargetPosition=tuple(self._registration.collar_pb),
            physicsClientId=self._cid,
        )

    def _draw_goal_ghost(self) -> None:
        """(Re)pose the translucent green body at the goal configuration."""
        assert self._fk is not None and self._mesh_cache is not None
        assert self._goal_q is not None
        if self._goal_mesh is None:
            self._goal_mesh = HumanMeshBody(
                self._cid, self._mesh_cache, GOAL_COLOR, arm_only=True
            )
        self._goal_mesh.update(
            self._fk.fk(
                q_to_arm_aa(self._goal_q, self._fk.elbow_hinge_axis),
                self._spine3_pos,
                self._spine3_aa,
            )
        )

    def _update_live_view(self, q_meas: np.ndarray) -> None:
        """Redraw the person at the measured configuration; mesh rate-limited.

        Replacing the 6890-vertex body in the GUI costs ~140 ms (measured) —
        pybullet has no vertex-update call, and it holds the GIL across the
        remove/create pair, which stalls the MPC step *and* delays the mocap
        receive thread toward ``mocap_hold_timeout``. So the person's mesh
        refreshes at ``live_view_fps``; the planner's arm chain and the robot are
        redrawn every step, since debug lines and ``resetJointState`` are free by
        comparison — the numbers the MPC is working from stay live even when the
        body around them lags.
        """
        self._pose_arm_skeleton(q_meas)
        now = time.monotonic()
        if now - self._last_live_mesh_s < self._live_mesh_period:
            return
        self._last_live_mesh_s = now
        self._pose_human_mesh(q_meas)

    def _pose_human_mesh(self, q: np.ndarray) -> None:
        """Re-pose the human mesh to an arm configuration, unconditionally.

        The rate limit belongs to the live view during a run, where each rebuild
        stalls an MPC step; a preview has no step to stall.
        """
        assert self._human_mesh is not None
        self._human_mesh.update(self._arm_positions(q))

    def _pose_arm_skeleton(self, q: np.ndarray) -> None:
        """Redraw the planner's arm chain at ``q``."""
        assert self._arm_skeleton is not None
        self._arm_skeleton.update(self._arm_positions(q))

    def _arm_positions(self, q: np.ndarray) -> np.ndarray:
        """The ``(5, 3)`` arm chain the planner's FK gives for ``q``."""
        assert self._fk is not None
        return self._fk.fk(
            q_to_arm_aa(q, self._fk.elbow_hinge_axis),
            self._spine3_pos,
            self._spine3_aa,
        )

    def _set_joints(self, values: np.ndarray) -> None:
        for joint, value in zip(self._movable_joints, values):
            p.resetJointState(
                self._robot, joint, float(value), physicsClientId=self._cid
            )

    # ------------------------------------------------------------------
    # Mocap read-back
    # ------------------------------------------------------------------

    def _arm_keypoints(
        self, bodies: dict[int, RigidBodyPose]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        return arm_keypoints(
            bodies,
            self._body_ids["collar"],
            self._body_ids["shoulder"],
            self._body_ids["elbow"],
            self._body_ids["wrist"],
        )

    def _read_back_q(self) -> np.ndarray:
        """Measure the human arm, holding the last valid pose through dropouts."""
        assert self._registration is not None
        _, bodies = self._receiver.latest()
        keypoints = self._arm_keypoints(bodies)
        if keypoints is not None:
            self._last_q = self._registration.q_from_keypoints(*keypoints)
            self._last_valid_s = time.monotonic()
        require_fresh(time.monotonic() - self._last_valid_s, self._hold_timeout)
        return self._last_q

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _forearm_frame_pb(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """The frame the measured grasp is expressed in, in pybullet coords."""
        assert self._fk is not None
        pos, rot = forearm_frame_fk(self._fk, q, self._spine3_pos, self._spine3_aa)
        to_pb = Rotation.from_matrix(_SMPL_TO_PB)
        return _SMPL_TO_PB @ pos, to_pb * rot

    def _nominal_grasp_pose_pb(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """The assumed grasp the simulated envs use — dry run only."""
        assert self._fk is not None
        grasp_pos, grasp_quat = grasp_pose_fk(
            self._fk, q, self._spine3_pos, self._spine3_aa, GRASP_FRACTION
        )
        return (
            _SMPL_TO_PB @ grasp_pos,
            Rotation.from_matrix(_SMPL_TO_PB) * Rotation.from_quat(grasp_quat),
        )

    def _grasp_pose_pb(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """Gripper pose the measured grasp implies for an arm configuration."""
        assert self._grasp is not None
        return self._grasp.gripper_pose(*self._forearm_frame_pb(q))

    def _ee_pose_pb(self) -> tuple[np.ndarray, Rotation]:
        """The IK robot's current end-effector pose."""
        ee_pos, ee_orn = p.getLinkState(
            self._robot,
            self._ee_index,
            computeForwardKinematics=True,
            physicsClientId=self._cid,
        )[4:6]
        return np.asarray(ee_pos, dtype=np.float64), Rotation.from_quat(ee_orn)

    def _grasp_error(self, q_cmd: np.ndarray) -> tuple[float, float]:
        """How far the gripper ended up from the grasp ``q_cmd`` asked for.

        Position in metres and attitude in radians, read after :meth:`_drive`
        has moved the robot. Two things put error here and both matter to a
        preview: a pose outside the padded joint box, where
        :meth:`_nearest_infeasible_ik` deliberately spends attitude to keep
        position; and ``robot_max_joint_delta``, which rate-limits the step and
        so shows up as the arm lagging a plan that moves faster than it may.
        """
        target_pos, target_rot = self._grasp_pose_pb(q_cmd)
        ee_pos, ee_rot = self._ee_pose_pb()
        return (
            float(np.linalg.norm(ee_pos - target_pos)),
            float((ee_rot * target_rot.inv()).magnitude()),
        )

    @staticmethod
    def _print_grasp_error(errors: np.ndarray) -> None:
        """Summarise a plan's worth of :meth:`_grasp_error` for the operator."""
        position, attitude = errors[:, 0], np.rad2deg(errors[:, 1])
        print(
            f"[real] grasp error over the plan: position mean "
            f"{1e3 * position.mean():.1f} mm, max {1e3 * position.max():.1f} mm "
            f"at step {int(np.argmax(position))}; attitude mean "
            f"{attitude.mean():.1f} deg, max {attitude.max():.1f} deg at step "
            f"{int(np.argmax(attitude))}",
            flush=True,
        )

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """Continue the branch the arm is on; enumerate only if it cannot reach.

        Continuation first, because this arm is redundant: its exact solutions
        for a pose form a self-motion manifold, so any number of far-apart
        configurations hit the target just as exactly and *which* one a solver
        returns is not pinned down by the pose. Taking whichever scores best
        walks the gripper across the person for nothing — over one plan that
        moved the solution up to 1.5 rad in a step against a
        ``robot_max_joint_delta`` of 0.01, which the arm then spends tens of
        steps chasing with the pose wrong the whole way. Following the branch
        the arm is already on removes the choice instead of making it well.

        Enumeration is the fallback, for when continuation genuinely cannot
        reach — the pose left the current branch, or left the controller's
        padded box. ``ssik`` returns *every* analytical branch ranked against
        the current configuration, so the nearest reachable one is known
        outright rather than searched for; that is what the previous solve could
        only approach heuristically, from three seeds chosen to bait the right
        basin. Only when nothing exact fits does anything numerical run.
        """
        rest = self._sim_q()
        target = self._ee_target_in_base(target_pos, target_quat)
        for track in (True, False):
            reachable = [
                solution
                for solution in self._ik_solutions(target, rest, track=track)
                if np.all(solution >= self._joint_lower)
                and np.all(solution <= self._joint_upper)
            ]
            if reachable:
                return reachable[0]
        return self._nearest_infeasible_ik(target, rest)

    def _ee_target_in_base(
        self, target_pos: np.ndarray, target_quat: np.ndarray
    ) -> np.ndarray:
        """Move a world-frame ``tool_frame`` pose into ``gen3_ik``'s frames.

        The artifact is baked against the nominal kortex chain and works in
        ``end_effector_link`` relative to ``base_link``, so a target has to leave
        both frames it arrives in: the pybullet world the scene is built in (the
        robot stands at its *measured* base pose) and the ``tool_frame`` between
        the gripper's fingers that the grasp is measured at.
        """
        return (
            self._world_to_base
            @ _pose_matrix(np.asarray(target_pos, dtype=np.float64), target_quat)
            @ self._tool_to_ee
        )

    def _ik_solutions(
        self, target: np.ndarray, q_seed: np.ndarray, *, track: bool
    ) -> list[np.ndarray]:
        """Exact configurations for a base-frame ee pose, nearest ``q_seed`` first.

        With ``track`` this asks only for the Newton continuation from
        ``q_seed`` — the artifact takes that fast path when a seed is given and a
        single solution wanted, and it rejects its own step if it lands on a
        different branch, which is the continuity the trajectory needs and a
        tenth the cost of resolving the redundancy. Without it, every analytical
        branch comes back, which is what a genuine branch change needs.

        ``respect_limits`` is off throughout because the artifact filters against
        the URDF's limits, which are not the ones the arm has to satisfy.
        Solutions come back on whatever winding the algebra produced, so the
        continuous joints are re-branched onto the half-turn either side of
        ``q_seed`` — the same configuration, expressed as the move the arm would
        actually make to reach it.
        """
        solutions = []
        for solution in gen3_ik.solve(
            target,
            q_seed=q_seed,
            max_solutions=1 if track else None,
            respect_limits=False,
        ):
            joints = np.asarray(solution.q, dtype=np.float64)
            delta = joints[self._continuous_joints] - q_seed[self._continuous_joints]
            joints[self._continuous_joints] = q_seed[self._continuous_joints] + (
                np.arctan2(np.sin(delta), np.cos(delta))
            )
            solutions.append(joints)
        return solutions

    def _nearest_infeasible_ik(
        self, target: np.ndarray, rest: np.ndarray
    ) -> np.ndarray:
        """Miss the pose as gently as the physical grasp would.

        Reached only when no exact solution fits the controller's padded box:
        the grasp pose is rigid within a step and comes from FK on the *measured*
        forearm, so when the person moves their arm somewhere the padded Gen3
        cannot follow rigidly, there is nothing exact to find. Physically the
        grasp stays clamped and the wrist is what compromises, so
        ``_IK_ORIENTATION_WEIGHT_M_PER_RAD`` makes orientation the cheap term of
        a bounded least-squares: position holds until it is itself infeasible,
        and the attitude carries the miss.

        The residual is built on ``gen3_ik``'s forward kinematics rather than
        pybullet's. Pybullet holds link state in single precision, so
        differencing it over the optimiser's ~1e-8 step returns a Jacobian that
        is mostly rounding — measured against the analytical one, a first column
        of ``[-0.375, 0, 0]`` where the truth is ``[-0.304, 0.022, 0]``. Every
        numerical solve here used to run on that.
        """
        target_position = target[:3, 3]
        target_rotation = Rotation.from_matrix(target[:3, :3])
        lower = np.where(self._continuous_joints, rest - np.pi, self._joint_lower)
        upper = np.where(self._continuous_joints, rest + np.pi, self._joint_upper)

        def pose_error(joints: np.ndarray) -> np.ndarray:
            pose = gen3_ik.fk(joints)
            return np.concatenate(
                (
                    pose[:3, 3] - target_position,
                    _IK_ORIENTATION_WEIGHT_M_PER_RAD
                    * (
                        Rotation.from_matrix(pose[:3, :3]) * target_rotation.inv()
                    ).as_rotvec(),
                )
            )

        best: OptimizeResult | None = None
        seeds = self._ik_solutions(target, rest, track=False)[:_IK_SEEDS]
        for seed in (rest, *seeds):
            result = least_squares(
                pose_error,
                np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=200,
            )
            if best is None or result.cost < best.cost:
                best = result
        assert best is not None
        return np.asarray(best.x, dtype=np.float64)

    def _drive(self, q_cmd: np.ndarray, *, mirror: bool = True) -> None:
        """Command the gripper pose this step's grasp puts on the commanded arm.

        With ``mirror=False`` the same solve runs on the IK robot alone and
        nothing is sent, which is what animates a :meth:`preview`: the robot then
        integrates on its own previous pose instead of the real arm's, since the
        real arm is standing still while the preview plays.

        :meth:`SimMannequinEnv._drive` cannot target a grasp pose absolutely, and
        neither could this method while the grasp was captured once: SMPL FK and
        the tracked limb disagree on segment lengths, so an absolute target
        re-anchors on that read-back bias every step and integrates it into
        unbounded drift, and only a target relative to the read-back makes
        executing the read-back a no-op.

        Re-measuring removes the bias rather than differencing it away.
        :meth:`_measure_grasp` has just set the transform from *this* step's
        ``q_meas`` and ee pose, so the gripper pose it implies for ``q_meas`` is
        exactly the current ee pose — the relative form reduces to this one
        identically, and the read-back is still a no-op.

        The gripper takes the forearm's whole rotation, not just its direction:
        the grasp is rigid within a step, and the forearm frame comes from the FK
        chain, so it carries its own roll and has no up-reference to flip near a
        vertical forearm. Where that rigid pose leaves the reachable set,
        :meth:`_solve_ik`'s position-priority weighting keeps the gripper on the
        forearm and lets the orientation carry the miss.
        """
        q_now = self._current_q() if mirror else self._sim_q()
        self._set_joints(q_now)
        target_pos, target_rot = self._grasp_pose_pb(q_cmd)
        delta = self._solve_ik(target_pos, target_rot.as_quat()) - q_now
        delta = np.clip(
            delta, -self._robot_max_joint_delta, self._robot_max_joint_delta
        )
        target = np.clip(q_now + delta, self._joint_lower, self._joint_upper)
        self._set_joints(target)
        if mirror and self._mirror is not None:
            self._mirror.send(target)

    def _sim_q(self) -> np.ndarray:
        """The IK robot's joint configuration."""
        return np.array(
            [
                p.getJointState(self._robot, j, physicsClientId=self._cid)[0]
                for j in self._movable_joints
            ]
        )

    def _current_q(self) -> np.ndarray:
        """The arm's actual configuration — the real arm's when mirroring."""
        if self._mirror is not None:
            return self._mirror.current_q()
        return self._sim_q()
