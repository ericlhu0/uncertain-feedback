"""Human-action Cartesian MPC gated on exact robot grasp reachability.

The plain human-action planners command arm configurations and leave the env's
IK to chase each implied gripper pose; nothing stops them picking samples the
robot cannot actually track, and the executed motion then bends away from the
plan while wrestling the grasp. Here the sample space stays the human arm, but
every rollout's *leading* frames (the ones about to be executed) are checked
against the robot: each frame's forearm pose implies a gripper pose through
the rigid measured grasp, the environment continues its current IK branch to
that pose from the rollout's current robot joints against the same padded
joint limits, and rollouts whose remaining pose error exceeds
``max_grasp_ik_residual`` are discarded outright. Every sample set includes a zero-motion hold, so the planner
never has to execute an infeasible fallback. Execution is the normal
human-action path (``env.execute``).

Two planners share that gate. :class:`ArmMPCCartesianNoMDMIKGated` is the pure
Cartesian wrist-goal MPC. :class:`LeftArmMPCCartesianIKGated` adds MDM/UQ
corrections: playback stays direct (no sampling), but the same continuation IK
screens every robot-facing step — frames the robot cannot reach are dropped
when the trajectory is pushed, a step whose implied gripper pose continuation
cannot place is held rather than handed to execution's enumerating fallback,
and a frame the arm stops progressing toward is skipped so an unreachable
stretch cannot stall the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.arm_mpc_cartesian import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.arm_mpc_cartesian_no_mdm import (
    ArmMPCCartesianNoMDM,
)
from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import (
    Q_CLAVICLE,
    Q_DIM,
    Q_ELBOW,
    Q_SHOULDER,
    SmplLeftArmFK,
)
from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer

if TYPE_CHECKING:
    from uncertain_feedback.envs.grasp import MeasuredGrasp


class _IKGateMixin:
    """Grasp-reachability IK gate shared by the gated human-action planners.

    Must be combined with a Cartesian ``SmplLeftArmMPC`` subclass, which
    supplies the attributes declared below (``_spine3_pos``/``_spine3_aa``
    via ``_init_cartesian``).
    """

    # Supplied by the host class; annotations only, so the mixin never
    # shadows the real attributes at runtime.
    _env: ExecutionEnv
    _fk: SmplLeftArmFK
    _spine3_pos: np.ndarray
    _spine3_aa: np.ndarray

    def _init_ik_gate(
        self, max_grasp_ik_residual: float, grasp_residual_frames: int
    ) -> None:
        self._max_grasp_ik_residual = float(max_grasp_ik_residual)
        self._grasp_ik_frames = int(grasp_residual_frames)

    def _cartesian_cost(self, q_trajs: np.ndarray) -> np.ndarray:
        residuals = self._grasp_ik_residuals(q_trajs)
        feasible = residuals <= self._max_grasp_ik_residual
        return np.where(feasible, super()._cartesian_cost(q_trajs), np.inf)

    def _sample_actions(self, mean: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
        actions = super()._sample_actions(mean, size)
        actions[0] = 0.0
        return actions

    def _grasp_ik_residuals(self, q_trajs: np.ndarray) -> np.ndarray:
        """Worst leading-frame IK pose error per rollout, metres + radians.

        Each frame's gripper target follows from the forearm frame through the
        rigid measured grasp; the environment tracks the frames sequentially
        from its current joints against its padded joint box, so the error is
        what continuation cannot remove — a limit binding, a singularity, or a
        pose outside the workspace.

        Continuation only (:meth:`ExecutionEnv.track_robot_ik_batch`), never
        execution's enumeration fallback. A pose reachable solely by changing
        branch is exact but tens of steps away at the execution rate cap, so
        gating it in would pass a rollout the arm cannot follow — and paying
        enumeration's serial cost to decide that dominates the solve as soon as
        the samples stop being reachable.
        """
        # Deferred: envs.grasp/envs.sim_mannequin import this package back —
        # a module-level import here closes that cycle.
        from uncertain_feedback.envs.grasp import (  # pylint: disable=import-outside-toplevel
            forearm_frame_fk,
        )
        from uncertain_feedback.envs.sim_mannequin import (  # pylint: disable=import-outside-toplevel
            _SMPL_TO_PB,
        )

        n_seqs = q_trajs.shape[0]
        n_frames = min(self._grasp_ik_frames, q_trajs.shape[1] - 1)
        # The grasp must exist before the robot state is read: establishing it
        # (first sim solve) moves the robot to the grasp configuration.
        grasp = self._env.current_grasp(q_trajs[0, 0])
        chain = self._env.robot_fk()
        robot_q = np.tile(self._env.current_robot_q(), (n_seqs, 1))
        to_pb = Rotation.from_matrix(_SMPL_TO_PB)
        residuals = np.zeros(n_seqs)
        for t in range(1, 1 + n_frames):
            target_pos = np.empty((n_seqs, 3))
            target_rot = np.empty((n_seqs, 3, 3))
            for i in range(n_seqs):
                forearm_pos, forearm_rot = forearm_frame_fk(
                    self._fk, q_trajs[i, t], self._spine3_pos, self._spine3_aa
                )
                grip_pos, grip_rot = grasp.gripper_pose(
                    _SMPL_TO_PB @ forearm_pos, to_pb * forearm_rot
                )
                target_pos[i] = grip_pos
                target_rot[i] = grip_rot.as_matrix()
            active = np.isfinite(residuals)
            target_quat = Rotation.from_matrix(target_rot).as_quat()
            solutions, feasible = self._env.track_robot_ik_batch(
                target_pos[active], target_quat[active], robot_q[active]
            )
            robot_q[active] = solutions
            pos, rot = chain.ee_pose(robot_q)
            frame_res = np.linalg.norm(target_pos - pos, axis=-1) + np.linalg.norm(
                _rotvecs(target_rot, rot), axis=-1
            )
            frame_res[~active] = np.inf
            frame_res[np.flatnonzero(active)[~feasible]] = np.inf
            residuals = np.maximum(residuals, frame_res)
        return residuals

    def _continuation_residual(
        self, q: np.ndarray, grasp: MeasuredGrasp, robot_q: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """One frame's continuation-IK pose error, target gap, and solved joints.

        The residual is ``inf`` when continuation cannot place the frame's
        implied gripper pose from ``robot_q`` (``(1, 7)``) — the same yes/no
        question the rollout gate asks, for a single configuration. The gap is
        the pose distance (metres + radians) between the target and the
        returned configuration's gripper — on failure that configuration is
        the seed, so the gap says how far the target sits from where the walk
        stands (a near-zero gap that still fails points at the joint box, not
        at distance).
        """
        from uncertain_feedback.envs.grasp import (  # pylint: disable=import-outside-toplevel
            forearm_frame_fk,
        )
        from uncertain_feedback.envs.sim_mannequin import (  # pylint: disable=import-outside-toplevel
            _SMPL_TO_PB,
        )

        forearm_pos, forearm_rot = forearm_frame_fk(
            self._fk, q, self._spine3_pos, self._spine3_aa
        )
        target_pos, target_rot = grasp.gripper_pose(
            _SMPL_TO_PB @ forearm_pos, Rotation.from_matrix(_SMPL_TO_PB) * forearm_rot
        )
        solutions, feasible = self._env.track_robot_ik_batch(
            target_pos[np.newaxis], target_rot.as_quat()[np.newaxis], robot_q
        )
        pos, rot = self._env.robot_fk().ee_pose(solutions)
        gap = float(
            np.linalg.norm(target_pos - pos[0])
            + np.linalg.norm(_rotvecs(target_rot.as_matrix()[np.newaxis], rot)[0])
        )
        return (gap if feasible[0] else np.inf), gap, solutions


class ArmMPCCartesianNoMDMIKGated(_IKGateMixin, ArmMPCCartesianNoMDM):
    """Pure Cartesian wrist-goal MPC that discards robot-infeasible samples.

    Args:
        max_grasp_ik_residual: Per-frame IK pose error (metres + radians)
            above which a rollout's leading frames count as breaking the
            grasp, discarding the rollout. The floor is IK convergence, not
            model mismatch, so it can sit near zero.
        grasp_residual_frames: How many leading frames that gate covers.
    """

    def __init__(
        self,
        cartesian_goals: list[np.ndarray],
        initial_q: np.ndarray,
        cartesian_threshold: float = 0.05,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        max_grasp_ik_residual: float = 0.001,
        grasp_residual_frames: int = 3,
        goal_threshold: float = 0.1,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
        seed: int | None = None,
        env: ExecutionEnv | None = None,
    ) -> None:
        if env is None:
            raise ValueError("ArmMPCCartesianNoMDMIKGated requires a robot env.")
        super().__init__(
            cartesian_goals=cartesian_goals,
            initial_q=initial_q,
            cartesian_threshold=cartesian_threshold,
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            goal_threshold=goal_threshold,
            visualize=visualize,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
            extra_costs=extra_costs,
            seed=seed,
            env=env,
        )
        self._init_ik_gate(max_grasp_ik_residual, grasp_residual_frames)


class LeftArmMPCCartesianIKGated(_IKGateMixin, LeftArmMPCCartesian):
    """MDM+UQ Cartesian MPC whose robot-facing steps are all IK-screened.

    MDM playback stays direct — no sampling, no cost — but never hands
    execution a pose whose gripper target continuation IK cannot place
    exactly, which is when :class:`~uncertain_feedback.envs.real.RealEnv`
    would otherwise fall back to a branch enumeration (an exact solution the
    arm chases for tens of steps with the grasp wrong) or a graceful miss
    (attitude twisted against the forearm). Three screens, all against the
    live measured grasp and robot state:

    - **push**: frames of a queued trajectory the robot cannot reach are
      dropped before playback starts;
    - **step**: each rate-limited playback step is checked and held when
      continuation cannot place it (the arm waits instead of missing);
    - **stall**: a frame the arm stops making progress toward is skipped
      after ``playback_stall_steps`` steps, so an unreachable stretch cannot
      hold the run forever — the playback follows the trajectory's reachable
      shadow.

    The post-playback Cartesian goal phase is the mixin's gated sampling,
    identical to :class:`ArmMPCCartesianNoMDMIKGated`.

    Args:
        max_grasp_ik_residual: As in :class:`ArmMPCCartesianNoMDMIKGated`;
            also the threshold for the push and step screens.
        grasp_residual_frames: Leading rollout frames the goal-phase gate
            covers.
        playback_stall_steps: Consecutive steps without closest-approach
            progress on the current playback frame before the cursor skips it.
    """

    def __init__(
        self,
        cartesian_goals: list[np.ndarray],
        initial_q: np.ndarray,
        cartesian_threshold: float = 0.05,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        advance_threshold: float = 0.1,
        max_playback_delta: float = 0.05,
        trajectory_fraction: float = LeftArmMPCMDM.TRAJECTORY_FRACTION,
        goal_threshold: float = 0.1,
        max_grasp_ik_residual: float = 0.001,
        grasp_residual_frames: int = 3,
        playback_stall_steps: int = 40,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        n_diffusion_samples: int = 512,
        n_clusters: int = 3,
        clusterer: TrajectoryClusterer | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
        seed: int | None = None,
        env: ExecutionEnv | None = None,
    ) -> None:
        if env is None:
            raise ValueError("LeftArmMPCCartesianIKGated requires a robot env.")
        super().__init__(
            cartesian_goals=cartesian_goals,
            initial_q=initial_q,
            cartesian_threshold=cartesian_threshold,
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            advance_threshold=advance_threshold,
            max_playback_delta=max_playback_delta,
            trajectory_fraction=trajectory_fraction,
            goal_threshold=goal_threshold,
            visualize=visualize,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
            n_diffusion_samples=n_diffusion_samples,
            n_clusters=n_clusters,
            clusterer=clusterer,
            extra_costs=extra_costs,
            seed=seed,
            env=env,
        )
        self._init_ik_gate(max_grasp_ik_residual, grasp_residual_frames)
        self._playback_stall_steps = int(playback_stall_steps)
        self._stall_best_dist = np.inf
        self._stall_steps = 0

    def push_trajectory(
        self, frames: np.ndarray, current_q: np.ndarray | None = None
    ) -> None:
        """Drop robot-unreachable frames, then queue the rest for playback.

        With ``current_q`` (the live measured configuration) the frames are
        walked sequentially through continuation IK from the current robot
        state — the seed advances only through kept frames, since those are
        what playback will visit. Advisory in tone but load-bearing in effect:
        dropped frames are simply never targeted, and the step screen still
        guards the interpolation between the survivors. Without ``current_q``
        the screen is skipped (no live robot state to screen against).
        """
        frames = np.asarray(frames, dtype=np.float64)
        q_frames = (
            frames
            if frames.shape[-1:] == (Q_DIM,)
            else self._fk.arm_aa_to_q_batch(frames, self._mdm_spine3_aa)
        )
        if current_q is not None:
            residuals, gaps = self._frame_ik_residuals(
                q_frames, np.asarray(current_q, dtype=np.float64)
            )
            keep = residuals <= self._max_grasp_ik_residual
            if not keep.any():
                print(
                    "[push] every MDM frame is robot-unreachable; nothing "
                    f"queued for playback. Frame 0's target sits {gaps[0]:.4f} "
                    "m+rad from the current gripper pose — near zero means the "
                    "solve fails on the joint box, not distance (see any "
                    "[real] joint-box warning at startup)."
                )
                return
            if not keep.all():
                dropped = np.flatnonzero(~keep)
                print(
                    f"[push] dropped {dropped.size}/{len(q_frames)} "
                    f"robot-unreachable MDM frames (first: frame {dropped[0]}, "
                    f"worst gap {gaps[~keep].max():.4f} m+rad); playback "
                    "follows the reachable frames."
                )
                q_frames = q_frames[keep]
            clav, shoulder, elbow = _frame_block_distances(
                np.asarray(current_q, dtype=np.float64), q_frames[0]
            )
            print(
                f"[push] first kept frame sits {clav:.3f}/{shoulder:.3f}/"
                f"{elbow:.3f} rad (clavicle/shoulder/elbow) from the measured "
                "arm — a correction generated from the live pose should start "
                "near zero on every block."
            )
        self._reset_stall()
        super().push_trajectory(q_frames)
        # The correction path set the goal marker to the unscreened last frame.
        if current_q is not None:
            self.set_mdm_goal(q_frames[-1])

    def _frame_ik_residuals(
        self, q_frames: np.ndarray, current_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sequential continuation-IK pose error and target gap per frame."""
        grasp = self._env.current_grasp(current_q)
        robot_q = self._env.current_robot_q()[np.newaxis]
        residuals = np.empty(len(q_frames))
        gaps = np.empty(len(q_frames))
        for i, q in enumerate(q_frames):
            residual, gap, solutions = self._continuation_residual(q, grasp, robot_q)
            residuals[i] = residual
            gaps[i] = gap
            if residual <= self._max_grasp_ik_residual:
                robot_q = solutions
        return residuals, gaps

    def _playback_step(self, current_q: np.ndarray) -> np.ndarray:
        """One rate-limited playback step, held when the robot cannot track it.

        The check guarantees execution's own solve then succeeds by
        continuation, so neither the enumerating fallback nor the
        position-priority miss ever fires during playback.
        """
        current_q = np.asarray(current_q, dtype=np.float64)
        q_next = self._advance_playback(current_q)
        if not self._step_reachable(current_q, q_next):
            q_next = current_q
        next_q = self._env.execute(q_next)
        dist = (
            float(np.linalg.norm(next_q - self._preview_q))
            if self._preview_q is not None
            else 0.0
        )
        self._update_playback_vis(next_q, dist)
        return next_q

    def _step_reachable(self, current_q: np.ndarray, q_cmd: np.ndarray) -> bool:
        grasp = self._env.current_grasp(current_q)
        robot_q = self._env.current_robot_q()[np.newaxis]
        residual, _, _ = self._continuation_residual(q_cmd, grasp, robot_q)
        return residual <= self._max_grasp_ik_residual

    def _advance_playback(self, current_q: np.ndarray) -> np.ndarray:
        """Rate-limited cursor with a stall-skip for unreachable frames.

        The cursor normally advances only once the measured arm reaches the
        frame; a frame the robot cannot reach (or settle on, under compliant
        control) would hold it forever. Closest approach to the frame is
        tracked instead — monotone, so measurement jitter cannot reset the
        counter — and after ``playback_stall_steps`` steps without progress
        the frame is skipped. A person actively resisting reads as a stall
        too, which is the desired yielding behavior.
        """
        assert self._playback_frames is not None
        idx = self._playback_idx
        target_q = self._playback_frames[idx]
        next_q = super()._advance_playback(current_q)
        if self._playback_idx != idx:
            self._reset_stall()
            return next_q
        dist = _frame_distance(current_q, target_q)
        if dist < self._stall_best_dist - 0.1 * self._max_playback_delta:
            self._stall_best_dist = dist
            self._stall_steps = 0
            return next_q
        self._stall_steps += 1
        if self._stall_steps >= self._playback_stall_steps:
            print(
                f"[playback] frame {idx} stalled {dist:.4f} rad away after "
                f"{self._stall_steps} steps without progress; skipping."
            )
            self._playback_idx += 1
            self._reset_stall()
            if not self._in_playback():
                self.reset_warmstart()
        return next_q

    def _reset_stall(self) -> None:
        self._stall_best_dist = np.inf
        self._stall_steps = 0


def _frame_block_distances(
    q: np.ndarray, target_q: np.ndarray
) -> tuple[float, float, float]:
    """Per-block geodesic angles (clavicle, shoulder, elbow) to ``target_q``."""
    blocks = tuple(
        float(
            np.linalg.norm(
                (
                    Rotation.from_rotvec(target_q[block])
                    * Rotation.from_rotvec(q[block]).inv()
                ).as_rotvec()
            )
        )
        for block in (Q_CLAVICLE, Q_SHOULDER)
    )
    return blocks[0], blocks[1], abs(float(target_q[Q_ELBOW] - q[Q_ELBOW]))


def _frame_distance(q: np.ndarray, target_q: np.ndarray) -> float:
    """Worst per-block geodesic angle to ``target_q``.

    The same three angles ``_rate_limited_step_q`` caps, so
    ``_frame_distance(q, t) <= max_playback_delta`` iff the cursor's reach
    test passes from ``q``.
    """
    return max(_frame_block_distances(q, target_q))


def _rotvecs(target_rot: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``(N, 3)`` world-frame rotation errors taking ``rot`` to ``target_rot``."""
    return Rotation.from_matrix(target_rot @ np.swapaxes(rot, -1, -2)).as_rotvec()
