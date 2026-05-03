"""MDM+UQ MPC for the SMPL left arm with a Cartesian terminal goal.

Extends :class:`LeftArmMPCMDMUQ` so that the terminal goal — reached after
the MDM-generated trajectory is exhausted — is a **Cartesian wrist position
relative to the spine3 joint** rather than a joint-angle configuration.
Rotation is unconstrained; only end-effector position is optimised.
"""

from __future__ import annotations

import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc import (
    _N_JOINTS,
    _compose_rotvec,
)
from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.visualizer import ArmVisualizer, _MDM_COLOR, _TARGET_COLOR
from uncertain_feedback.uncertainty.base import TrajectoryClusterer


class LeftArmMPCCartesian(LeftArmMPCMDMUQ):
    """MDM+UQ MPC with a Cartesian terminal goal.

    MDM-generated joint-angle waypoints are tracked with the standard
    joint-angle cost (inherited).  Once the waypoint queue is empty the
    controller switches to a Cartesian cost: it minimises the L2 distance
    between the left wrist and ``cartesian_goal``, expressed relative to
    spine3.  Joint orientations are left unconstrained.

    Args:
        cartesian_goal:  ``(3,)`` target wrist position in spine3-relative
                         coordinates (Y-up, metres).
        initial_arm_aa:  ``(4, 3)`` arm axis-angles at run start.  Used as a
                         ghost-arm placeholder in the visualiser while the
                         goal queue is empty.
        horizon:         Look-ahead steps.
        n_mpc_samples:   CEM sample count.
        max_angle_delta: Sampling std dev (radians).
        advance_threshold: Joint-space advance threshold for MDM waypoints.
        trajectory_fraction: Fraction of MDM frames to enqueue.
        goal_threshold:  Unused; kept for API compatibility.
        visualize:       Open live matplotlib window.
        fk:              :class:`SmplLeftArmFK` instance (required).
        spine3_pos:      ``(3,)`` world position of spine3.
        spine3_aa:       ``(3,)`` world axis-angle of spine3.
        body_pos:        ``(22, 3)`` background skeleton joint positions.
        n_diffusion_samples: Number of MDM diffusion samples for UQ.
        n_clusters:      Number of KMeans clusters.
        clusterer:       Custom :class:`TrajectoryClusterer`.
    """

    def __init__(
        self,
        cartesian_goal: np.ndarray,
        initial_arm_aa: np.ndarray,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        advance_threshold: float = 0.1,
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
    ) -> None:
        if fk is None:
            raise ValueError("fk is required for LeftArmMPCCartesian.")
        super().__init__(
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            advance_threshold=advance_threshold,
            trajectory_fraction=trajectory_fraction,
            goals=[],
            goal_threshold=goal_threshold,
            visualize=visualize,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
            n_diffusion_samples=n_diffusion_samples,
            n_clusters=n_clusters,
            clusterer=clusterer,
        )
        self._cartesian_goal = np.asarray(cartesian_goal, dtype=np.float64)
        self._initial_arm_aa = np.asarray(initial_arm_aa, dtype=np.float64)
        self._fk_inst = fk
        self._spine3_pos = (
            np.asarray(spine3_pos, dtype=np.float64)
            if spine3_pos is not None
            else fk.tpose_spine3_pos
        )
        self._spine3_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3)
        )

    # ------------------------------------------------------------------
    # Cartesian cost
    # ------------------------------------------------------------------

    def _cartesian_cost(self, q_trajs: np.ndarray) -> np.ndarray:
        """L2 Cartesian cost: spine3-relative wrist distance to target.

        Args:
            q_trajs: ``(N, H+1, 4, 3)`` state trajectories.

        Returns:
            ``(N,)`` cost per trajectory.
        """
        terminal_q = q_trajs[:, -1]  # (N, 4, 3)
        positions = self._fk_inst.fk_batch(
            terminal_q, self._spine3_pos, self._spine3_aa
        )  # (N, 5, 3)
        wrist_rel = positions[:, -1] - self._spine3_pos  # (N, 3)
        return ((wrist_rel - self._cartesian_goal) ** 2).sum(axis=-1)  # (N,)

    # ------------------------------------------------------------------
    # solve override
    # ------------------------------------------------------------------

    def solve(
        self, current_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the best first action and full plan.

        Delegates to the parent (joint-angle cost) while MDM waypoints are
        queued.  Switches to Cartesian cost once the queue is empty.
        """
        if self._goals:
            return super().solve(current_q)

        current_q = np.asarray(current_q, dtype=np.float64)

        if self._prev_best is not None:
            mean = np.concatenate(
                [self._prev_best[1:], np.zeros((1, _N_JOINTS, 3))], axis=0
            )
        else:
            mean = np.zeros(
                (self._config.horizon, _N_JOINTS, 3), dtype=np.float64
            )

        actions = np.random.normal(
            loc=mean,
            scale=self._config.max_angle_delta,
            size=(
                self._config.n_mpc_samples,
                self._config.horizon,
                _N_JOINTS,
                3,
            ),
        )
        q_trajs = self._rollout(current_q, actions)
        costs = self._cartesian_cost(q_trajs)

        best_idx = np.argmin(costs)
        best_plan = actions[best_idx]
        self._prev_best = best_plan
        return best_plan[0], best_plan

    # ------------------------------------------------------------------
    # step override
    # ------------------------------------------------------------------

    def step(
        self,
        current_q: np.ndarray,
        advance_threshold: float | None = None,
    ) -> np.ndarray:
        """Perform one MPC step.

        While MDM waypoints are queued, tracks them with joint-angle cost.
        Unlike the parent, the last waypoint CAN be popped (no ``len > 1``
        guard), allowing the queue to empty and Cartesian mode to engage.

        Once the queue is empty, optimises toward the Cartesian wrist goal.
        """
        if self._goals:
            return self._mdm_tracking_step(current_q, advance_threshold)
        return self._cartesian_step(current_q)

    def _mdm_tracking_step(
        self,
        current_q: np.ndarray,
        advance_threshold: float | None = None,
    ) -> np.ndarray:
        target_q = self._goals[0]
        first_action, _ = self.solve(current_q)
        next_q = _compose_rotvec(
            np.asarray(current_q, dtype=np.float64), first_action
        )

        threshold = (
            advance_threshold
            if advance_threshold is not None
            else self.advance_threshold
        )
        dist = float(np.linalg.norm(next_q - target_q))
        # No len > 1 guard — allow queue to empty so Cartesian mode engages.
        if dist < threshold:
            self._goals.popleft()
            self.reset_warmstart()
            if self._goals:
                target_q = self._goals[0]
                dist = float(np.linalg.norm(next_q - target_q))

        if self._vis_config is not None:
            if self._vis is None:
                vis_goal = self._goals[-1] if self._goals else self._initial_arm_aa
                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    vis_goal,
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
                if self._mdm_goal is not None:
                    self._vis.update_mdm_goal(self._mdm_goal)
                if self._preview_q is not None:
                    self._vis.update_trajectory_preview(self._preview_q)
                self._vis.update_cartesian_target(
                    self._spine3_pos + self._cartesian_goal
                )
            color = _MDM_COLOR if self._goals else _TARGET_COLOR
            self._vis.update_step(next_q, dist=dist, color=color)

        return next_q

    def _cartesian_step(self, current_q: np.ndarray) -> np.ndarray:
        first_action, _ = self.solve(current_q)
        next_q = _compose_rotvec(
            np.asarray(current_q, dtype=np.float64), first_action
        )

        arm_pos = self._fk_inst.fk(next_q, self._spine3_pos, self._spine3_aa)
        wrist_rel = arm_pos[-1] - self._spine3_pos
        dist = float(np.linalg.norm(wrist_rel - self._cartesian_goal))

        if self._vis_config is not None:
            if self._vis is None:
                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    self._initial_arm_aa,
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
                if self._mdm_goal is not None:
                    self._vis.update_mdm_goal(self._mdm_goal)
                self._vis.update_cartesian_target(
                    self._spine3_pos + self._cartesian_goal
                )
            self._vis.update_step(next_q, dist=dist, color=_TARGET_COLOR)

        return next_q
