"""Grasp-reachability feasibility constraint via exact continuation IK.

The plain human-action solve commands arm configurations and leaves the env's
IK to chase each implied gripper pose; nothing stops it picking samples the
robot cannot actually track, and the executed motion then bends away from the
plan while wrestling the grasp. This constraint checks every rollout's
*leading* frames (the ones about to be executed) against the robot: each
frame's forearm pose implies a gripper pose through the rigid measured grasp,
the environment continues its current IK branch to that pose from the
rollout's current robot joints against the same padded joint limits, and
rollouts whose remaining pose error exceeds ``max_residual`` are discarded
outright.

Feedback playback is screened by the same continuation IK: frames the robot
cannot reach are dropped when the trajectory is pushed, a step whose implied
gripper pose continuation cannot place is held rather than handed to
execution's enumerating fallback, and a frame the arm stops progressing
toward is skipped (``playback_stall_steps``) so an unreachable stretch cannot
stall the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.action_spaces.base import RolloutBatch
from uncertain_feedback.planners.mpc.constraints.base import FeasibilityConstraint
from uncertain_feedback.planners.mpc.kinematics import (
    SmplLeftArmFK,
    _frame_block_distances,
)

if TYPE_CHECKING:
    from uncertain_feedback.envs.grasp import MeasuredGrasp


@dataclass(frozen=True)
class RobotIkConfig:
    """Robot-IK feasibility parameters (the ``constraints: robot_ik:`` entry).

    Args:
        max_residual: Per-frame IK pose error (metres + radians) above which a
            frame counts as breaking the grasp. The floor is IK convergence,
            not model mismatch, so it can sit near zero.
        grasp_residual_frames: How many leading rollout frames the gate covers.
        playback_stall_steps: Consecutive steps without closest-approach
            progress on the current playback frame before the cursor skips it.
    """

    max_residual: float = 0.001
    grasp_residual_frames: int = 3
    playback_stall_steps: int = 40


class RobotIkConstraint(FeasibilityConstraint):
    """Discards motions whose implied gripper poses continuation IK cannot place."""

    def __init__(
        self,
        cfg: RobotIkConfig,
        *,
        env: ExecutionEnv,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
    ) -> None:
        self._env = env
        self._fk = fk
        self._spine3_pos = spine3_pos
        self._spine3_aa = spine3_aa
        self._max_residual = float(cfg.max_residual)
        self._grasp_ik_frames = int(cfg.grasp_residual_frames)
        self.playback_stall_steps = int(cfg.playback_stall_steps)

    def rollout_feasible(self, batch: RolloutBatch) -> np.ndarray:
        assert batch.q_trajs is not None
        return self._grasp_ik_residuals(batch.q_trajs) <= self._max_residual

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

    def screen_frames(
        self, q_frames: np.ndarray, current_q: np.ndarray
    ) -> np.ndarray:
        """Drop robot-unreachable frames of a feedback trajectory.

        The frames are walked sequentially through continuation IK from the
        current robot state — the seed advances only through kept frames,
        since those are what playback will visit. Advisory in tone but
        load-bearing in effect: dropped frames are simply never targeted, and
        the step screen still guards the interpolation between the survivors.
        """
        residuals, gaps = self._frame_ik_residuals(q_frames, current_q)
        keep = residuals <= self._max_residual
        if not keep.any():
            print(
                "[push] every MDM frame is robot-unreachable; nothing "
                f"queued for playback. Frame 0's target sits {gaps[0]:.4f} "
                "m+rad from the current gripper pose — near zero means the "
                "solve fails on the joint box, not distance (see any "
                "[real] joint-box warning at startup)."
            )
            return q_frames[:0]
        if not keep.all():
            dropped = np.flatnonzero(~keep)
            print(
                f"[push] dropped {dropped.size}/{len(q_frames)} "
                f"robot-unreachable MDM frames (first: frame {dropped[0]}, "
                f"worst gap {gaps[~keep].max():.4f} m+rad); playback "
                "follows the reachable frames."
            )
            q_frames = q_frames[keep]
        clav, shoulder, elbow = _frame_block_distances(current_q, q_frames[0])
        print(
            f"[push] first kept frame sits {clav:.3f}/{shoulder:.3f}/"
            f"{elbow:.3f} rad (clavicle/shoulder/elbow) from the measured "
            "arm — a correction generated from the live pose should start "
            "near zero on every block."
        )
        return q_frames

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
            if residual <= self._max_residual:
                robot_q = solutions
        return residuals, gaps

    def step_reachable(self, current_q: np.ndarray, q_cmd: np.ndarray) -> bool:
        """Whether execution's own solve would succeed by continuation.

        Guarantees neither the enumerating fallback nor the position-priority
        miss ever fires during playback.
        """
        grasp = self._env.current_grasp(current_q)
        robot_q = self._env.current_robot_q()[np.newaxis]
        residual, _, _ = self._continuation_residual(q_cmd, grasp, robot_q)
        return residual <= self._max_residual


def _rotvecs(target_rot: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``(N, 3)`` world-frame rotation errors taking ``rot`` to ``target_rot``."""
    return Rotation.from_matrix(target_rot @ np.swapaxes(rot, -1, -2)).as_rotvec()
