"""MDM+UQ Cartesian MPC acting in robot joint space.

The robot-action analog of :class:`LeftArmMPCCartesian`: corrections, UQ, and
LLM-generated costs are inherited untouched (they operate on human-q
trajectories), but both execution phases sample robot joint deltas and command
the robot directly. MDM playback keeps its rate-limited cursor; each playback
frame becomes the target of a one-step robot-action solve instead of a
human-q command routed through IK — so playback, too, can only ask for
motions the grasp can transmit.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.arm_mpc_cartesian import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.arm_mpc_robot import _RobotActionsMixin
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import (
    Q_ELBOW,
    Q_SHOULDER,
    SmplLeftArmFK,
)
from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer


class LeftArmMPCCartesianRobot(_RobotActionsMixin, LeftArmMPCCartesian):
    """MDM+UQ MPC with Cartesian wrist goals, sampling robot joint deltas.

    Args:
        max_robot_joint_delta: Per-step inf-norm cap on sampled robot joint
            deltas (radians) — the same cap execution enforces.
        robot_joint_delta_std: Std of the joint-delta sampling noise around
            the warm-started mean (radians). None means a third of the cap.
        robot_infeasibility_weight: Weight on the grasp-transmission residual.
        max_grasp_residual: Per-frame residual above which a rollout's leading
            frames count as breaking the grasp, discarding the rollout.
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
        max_robot_joint_delta: float = 0.005,
        robot_joint_delta_std: float | None = None,
        robot_infeasibility_weight: float = 1.0,
        max_grasp_residual: float = 0.02,
        grasp_residual_frames: int = 3,
        advance_threshold: float = 0.1,
        max_playback_delta: float = 0.05,
        trajectory_fraction: float = LeftArmMPCMDM.TRAJECTORY_FRACTION,
        goal_threshold: float = 0.1,
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
            raise ValueError("LeftArmMPCCartesianRobot requires a robot env.")
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
        self._init_robot_actions(
            max_robot_joint_delta,
            robot_joint_delta_std,
            robot_infeasibility_weight,
            max_grasp_residual,
            grasp_residual_frames,
        )

    def _robot_tracking_cost(self, q_target: np.ndarray):
        """Terminal distance to a playback frame, on the actuatable DOFs.

        The clavicle block is excluded: the robot cannot actuate the shoulder
        girdle, so any clavicle error in the playback frame is unreachable
        either way. The shoulder error is geodesic, not rotvec L2 — measured
        shoulder rotations sit near the ±pi rotvec boundary (the anatomical
        decode of a bent arm carries a ~pi twist), where the rotvec sign flips
        between steps and a plain L2 would see a phantom ~2pi error. Extra
        costs are not applied — playback follows a trajectory the user already
        validated, matching the direct-playback semantics of the human-action
        planner.
        """
        q_target = np.asarray(q_target, dtype=np.float64)
        hinge = self._fk.elbow_hinge_axis
        target_rot_inv = Rotation.from_rotvec(q_target[Q_SHOULDER]).inv()

        def cost(aa_trajs: np.ndarray, wrist_pos: np.ndarray) -> np.ndarray:
            _ = wrist_pos
            # Every horizon frame is costed, not just the terminal one: the
            # target is a (near-)stationary frame, and a terminal-only cost
            # leaves the executed first step free to wander around it.
            shoulder = aa_trajs[:, 1:, 1]
            relative = (
                target_rot_inv * Rotation.from_rotvec(shoulder.reshape(-1, 3))
            ).as_rotvec().reshape(shoulder.shape)
            shoulder_err = (relative**2).sum(axis=-1).mean(axis=-1)
            elbow_err = (
                (aa_trajs[:, 1:, 2] @ hinge - q_target[Q_ELBOW]) ** 2
            ).mean(axis=-1)
            return shoulder_err + elbow_err

        return cost

    def _playback_step(self, current_q: np.ndarray) -> np.ndarray:
        """Track one rate-limited playback frame with the robot-action solve."""
        current_q = np.asarray(current_q, dtype=np.float64)
        q_target = self._advance_playback(current_q)
        target = self._robot_solve(current_q, self._robot_tracking_cost(q_target))
        next_q = self._env.execute_robot(target)
        dist = (
            float(np.linalg.norm(next_q - self._preview_q))
            if self._preview_q is not None
            else 0.0
        )
        self._update_playback_vis(next_q, dist)
        return next_q

    def _cartesian_step(self, current_q: np.ndarray) -> np.ndarray:
        current_q = np.asarray(current_q, dtype=np.float64)
        if not self._cartesian_goals:
            self._env.current_grasp(current_q)
            return self._env.execute_robot(self._env.current_robot_q())

        target = self._robot_solve(current_q, self._robot_cartesian_cost)
        next_q = self._env.execute_robot(target)
        goal, dist = self._cartesian_progress(next_q)
        self._update_cartesian_vis(next_q, dist, goal)
        return next_q
