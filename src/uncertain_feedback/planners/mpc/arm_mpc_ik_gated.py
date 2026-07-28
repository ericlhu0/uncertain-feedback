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
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.arm_mpc_cartesian_no_mdm import (
    ArmMPCCartesianNoMDM,
)
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


class ArmMPCCartesianNoMDMIKGated(ArmMPCCartesianNoMDM):
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


def _rotvecs(target_rot: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``(N, 3)`` world-frame rotation errors taking ``rot`` to ``target_rot``."""
    return Rotation.from_matrix(target_rot @ np.swapaxes(rot, -1, -2)).as_rotvec()
