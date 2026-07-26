"""Real-world env: a mocap-measured human arm moved by the real Kinova Gen3.

The closed-loop counterpart to
:class:`~uncertain_feedback.envs.sim_mannequin.SimMannequinEnv`. There the robot
drags a simulated mannequin under physics and the real arm only shadows the sim;
here the human arm state is *measured* — OptiTrack rigid bodies on the collar,
shoulder, elbow, and wrist are streamed in over NatNet and converted to the
planner's ``(7,)`` configuration — so the MPC closes the loop on the actual
person. That includes the configuration the run *starts* from
(:meth:`RealEnv.initial_q`), so the person does not have to match the config's
start pose. The collar body also makes the registration yaw measurable rather
than assumed, and fixes *where* the person is: the torso anchor is the measured
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
from uncertain_feedback.envs.human_mesh import GOAL_COLOR, HumanMeshBody
from uncertain_feedback.envs.sim_mannequin import _ROBOT_SPECS, _SMPL_TO_PB
from uncertain_feedback.mocap.natnet import (
    NatNetReceiver,
    RigidBodyPose,
    require_fresh,
)
from uncertain_feedback.mocap.registration import ArmRegistration, arm_keypoints
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache

# Rigid-body roles the config must supply streaming ids for. The collar body is
# what makes the registration yaw measurable rather than assumed (see
# `mocap.registration`), so it is required, not optional.
_ARM_KEYS = ("collar", "shoulder", "elbow", "wrist")
_BASE_KEY = "robot_base"
# The person must be tracked before the scene can be built at all, so allow a
# generous window for markers to come into view.
_CALIBRATION_TIMEOUT_S = 20.0
# Re-reading tracking after the scene build should succeed on the first poll;
# anything longer means markers were lost while the scene came up.
_RESYNC_TIMEOUT_S = 2.0
# How far the measured gripper origin may sit from the forearm segment. The Gen3
# `tool_frame` is 12 cm past the flange, i.e. between the 2F-85's fingers, so a
# real grasp puts it within a few centimetres of the forearm axis; the slack is
# for SMPL bone lengths differing from the person's.
_GRASP_OFFSET_TOLERANCE_M = 0.15
# Live-view camera, framed on the person's collar (where the registration is
# anchored) with the robot in shot.
_LIVE_CAMERA_DISTANCE = 2.0
_LIVE_CAMERA_YAW = -30.0
_LIVE_CAMERA_PITCH = -12.0


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
    ) -> None:
        super().__init__()
        if robot not in _ROBOT_SPECS:
            raise ValueError(
                f"Unknown robot '{robot}'. Available: {sorted(_ROBOT_SPECS)}"
            )
        missing = {_BASE_KEY, *_ARM_KEYS} - set(mocap_rigid_bodies)
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
        self._live_mesh_period = 1.0 / live_view_fps if live_view_fps > 0.0 else 0.0
        self._last_live_mesh_s = 0.0
        self._cid: int = p.connect(p.GUI if self._live_view else p.DIRECT)
        if self._live_view:
            # The scene is the whole point of the window; pybullet's parameter
            # panes and preview tiles only take space away from it.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._cid)
        self._mesh_cache: SmplMeshCache | None = None
        self._human_mesh: HumanMeshBody | None = None
        self._goal_mesh: HumanMeshBody | None = None
        self._goal_q: np.ndarray | None = None
        self._registration: ArmRegistration | None = None
        self._robot: int = -1
        self._movable_joints: list[int] = []
        self._continuous_joints: np.ndarray = np.zeros(0, dtype=bool)
        self._joint_lower: np.ndarray = np.zeros(0, dtype=np.float64)
        self._joint_upper: np.ndarray = np.zeros(0, dtype=np.float64)
        self._ee_index: int = -1
        self._last_q: np.ndarray = np.zeros(0, dtype=np.float64)
        self._last_valid_s: float = 0.0
        self._grasp: MeasuredGrasp | None = None
        self._measured: list[np.ndarray] = []

    def initial_q(self, q_nominal: np.ndarray) -> np.ndarray:
        """Register against the person and report their *measured* arm config.

        Talks to mocap only — the grasp is measured on the first :meth:`execute`,
        once the planner (which may load a diffusion model) is built and about to
        command something.

        ``q_nominal`` is the run config's start pose. Only its clavicle slot
        survives — the registration yaw is solved by matching the measured
        collar->shoulder direction against the one that slot implies, so the
        clavicle and the yaw cannot both be measured (see
        :meth:`ArmRegistration.calibrate`). The elbow and wrist slots come
        from the person.
        """
        self._register(np.asarray(q_nominal, dtype=np.float64))
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

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        q = np.asarray(q_cmd, dtype=np.float64)
        if self._registration is None:
            self._register(q)
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

    def _register(self, q_nominal: np.ndarray) -> None:
        """Solve the mocap registration, build the scene, and measure the arm.

        Runs from :meth:`initial_q` before planning, because the robot base
        pose comes from mocap and the planner's start configuration is the
        measured one. Falls back to the first :meth:`execute` for callers that
        plan without asking for a start configuration.
        """
        assert self._fk is not None
        bodies = self._receiver.wait_for(
            [self._body_ids[key] for key in (_BASE_KEY, *_ARM_KEYS)],
            _CALIBRATION_TIMEOUT_S,
        )
        base = bodies[self._body_ids[_BASE_KEY]]
        keypoints = self._arm_keypoints(bodies)
        assert keypoints is not None
        self._registration = ArmRegistration.calibrate(
            fk=self._fk,
            q0=q_nominal,
            spine3_pos=self._spine3_pos,
            spine3_aa=self._spine3_aa,
            base_position=base.position,
            base_orientation=base.orientation,
            collar_mocap=keypoints[0],
            shoulder_mocap=keypoints[1],
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
        if self._mirror is None:
            self._pose_robot_at_nominal_grasp(q_meas)
        self._measure_grasp(q_meas)
        if self._mirror is not None:
            self._mirror.start_from_grasp()

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
        self._check_grasp_on_forearm(q_meas)

    def _pose_robot_at_nominal_grasp(self, q_meas: np.ndarray) -> None:
        """Dry run only: there is no real gripper pose to measure.

        Placing the IK robot at :func:`grasp_pose_fk`'s nominal grasp on the
        measured forearm gives the calibration something to read, so the
        mocap-only run exercises the same code path as the live one. The solution
        is clipped to the controller's joint limits and may miss the nominal
        point by centimetres; nothing moves, and a miss large enough to matter
        shows up as a grasp off the forearm in :meth:`_check_grasp_on_forearm`.
        """
        # Seed the IK from the arm's actual configuration. `calculateInverseKinematics`
        # seeds from the current joint state, and with 7 DOF the branch it returns
        # depends strongly on that seed: from `_RobotSpec.home` (tuned for
        # `sim_mannequin`'s hardcoded base offset) it can land on a branch that
        # needs clipping and so never reaches, while from the real arm's own
        # configuration it solves exactly.
        self._set_joints(self._current_q())
        target_pos, target_rot = self._nominal_grasp_pose_pb(q_meas)
        solution = self._solve_ik(
            target_pos,
            (target_rot * Rotation.from_quat(self._spec.tool_quat)).as_quat(),
        )
        self._set_joints(np.clip(solution, self._joint_lower, self._joint_upper))

    def _check_grasp_on_forearm(self, q_meas: np.ndarray) -> None:
        """Reject a measured grasp that did not land on the forearm.

        The measurement is only as good as the registration: a wrong yaw or a
        robot-base plate misaligned in Motive puts the person somewhere else in
        the scene, and the offset then bakes that error into a long lever arm
        that the MPC would swing the real arm through. A gripper actually holding
        the forearm sits within centimetres of the elbow→wrist segment.

        Running every step (with :meth:`_measure_grasp`) also makes this the slip
        guard: a grasp that creeps along the forearm or off it — or an arm the
        gripper is pushing past without carrying — walks the offset out of
        tolerance and halts the run instead of continuing on a lever arm that no
        longer describes anything physical.
        """
        assert self._registration is not None
        elbow, wrist = self._forearm_segment_pb(q_meas)
        gripper, _ = self._grasp_pose_pb(q_meas)
        bone = wrist - elbow
        along = float(
            np.clip(np.dot(gripper - elbow, bone) / float(np.dot(bone, bone)), 0.0, 1.0)
        )
        distance = float(np.linalg.norm(gripper - (elbow + along * bone)))
        if distance > _GRASP_OFFSET_TOLERANCE_M:
            raise RuntimeError(
                f"Measured grasp is {distance:.3f} m off the forearm (tolerance "
                f"{_GRASP_OFFSET_TOLERANCE_M} m), {along:.2f} of the way along it. "
                f"Gripper {np.round(gripper, 3)}, elbow {np.round(elbow, 3)}, wrist "
                f"{np.round(wrist, 3)}, robot base "
                f"{np.round(self._registration.base_pb, 3)}, solved yaw "
                f"{self._registration.robot_base_yaw:+.3f} rad, step "
                f"{len(self._measured)}. At step 0: the grasp was not taken before "
                "the run, or the mocap registration is wrong. Later: the grasp has "
                "slipped off the forearm."
            )

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
        self._set_joints(np.asarray(self._spec.home, dtype=np.float64))

    def _start_live_view(self) -> None:
        """Open the live scene: the measured person as a mesh, plus the robot.

        The robot's meshes come from the URDF already loaded for IK, so only the
        human needs adding. Both are drawn in the registered pybullet frame, so
        the window shows the geometry the MPC is actually solving against — a
        registration that put the person or the robot in the wrong place is
        visible here before the numbers say anything.
        """
        assert self._fk is not None and self._body_pos is not None
        assert self._registration is not None
        self._mesh_cache = SmplMeshCache(np.asarray(self._body_pos, dtype=np.float64))
        self._human_mesh = HumanMeshBody(self._cid, self._mesh_cache)
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
        """Re-pose the human mesh to the measured configuration, rate-limited.

        Replacing the 6890-vertex body in the GUI costs ~140 ms (measured) —
        pybullet has no vertex-update call, and it holds the GIL across the
        remove/create pair, which stalls the MPC step *and* delays the mocap
        receive thread toward ``mocap_hold_timeout``. So the person's mesh
        refreshes at ``live_view_fps``; the robot is re-posed every step
        regardless, since ``resetJointState`` is free by comparison.
        """
        assert self._fk is not None and self._human_mesh is not None
        now = time.monotonic()
        if now - self._last_live_mesh_s < self._live_mesh_period:
            return
        self._last_live_mesh_s = now
        self._human_mesh.update(
            self._fk.fk(
                q_to_arm_aa(q_meas, self._fk.elbow_hinge_axis),
                self._spine3_pos,
                self._spine3_aa,
            )
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

    def _forearm_segment_pb(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Elbow and wrist positions in pybullet coords."""
        assert self._fk is not None
        positions = self._fk.fk(
            q_to_arm_aa(q, self._fk.elbow_hinge_axis), self._spine3_pos, self._spine3_aa
        )
        return _SMPL_TO_PB @ positions[3], _SMPL_TO_PB @ positions[4]

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

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        solution = p.calculateInverseKinematics(
            self._robot,
            self._ee_index,
            tuple(target_pos),
            tuple(target_quat),
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self._cid,
        )
        return np.asarray(solution, dtype=np.float64)

    def _drive(self, q_cmd: np.ndarray) -> None:
        """Command the gripper pose this step's grasp puts on the commanded arm.

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
        vertical forearm.
        """
        q_now = self._current_q()
        self._set_joints(q_now)
        target_pos, target_rot = self._grasp_pose_pb(q_cmd)
        delta = self._solve_ik(target_pos, target_rot.as_quat()) - q_now
        # Continuous joints have no limits to anchor the IK solution, so it may
        # come back unwound by full turns; take the short way around.
        wrapped = self._continuous_joints
        delta[wrapped] = np.arctan2(np.sin(delta[wrapped]), np.cos(delta[wrapped]))
        delta = np.clip(
            delta, -self._robot_max_joint_delta, self._robot_max_joint_delta
        )
        target = np.clip(q_now + delta, self._joint_lower, self._joint_upper)
        self._set_joints(target)
        if self._mirror is not None:
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
