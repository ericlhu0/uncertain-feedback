"""MDM+UQ MPC for the SMPL left arm with a Cartesian goal queue.

Extends :class:`LeftArmMPCMDMUQ` so that after the MDM-generated trajectory
is exhausted the controller works through a queue of **Cartesian wrist
positions relative to the spine3 joint**.  Rotation is unconstrained; only
end-effector position is optimised.
"""

from __future__ import annotations

import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc_cartesian_base import _CartesianGoalsMixin
from uncertain_feedback.planners.mpc.arm_mpc_mdm import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, _compose_rotvec
from uncertain_feedback.planners.mpc.visualizer import (
    ArmVisualizer,
    _MDM_COLOR,
    _TARGET_COLOR,
)
from uncertain_feedback.uncertainty.base import TrajectoryClusterer


class LeftArmMPCCartesian(_CartesianGoalsMixin, LeftArmMPCMDMUQ):
    """MDM+UQ MPC with a queue of Cartesian wrist goals.

    MDM-generated joint-angle waypoints are tracked with the standard
    joint-angle cost (inherited).  Once that queue empties the controller
    works through ``cartesian_goals`` in order, minimising the L2 distance
    between the left wrist and each spine3-relative target position.
    Joint orientations are left unconstrained.

    Args:
        cartesian_goals: List of ``(3,)`` target wrist positions in
                         spine3-relative coordinates (Y-up, metres).
        initial_arm_aa:  ``(3, 3)`` controlled arm axis-angles at run start.  Used as a
                         ghost-arm placeholder in the visualiser while the
                         MDM waypoint queue is empty.
        cartesian_threshold: Cartesian L2 distance (metres) below which the
                         front Cartesian goal is considered reached and the
                         next one becomes active.  Default 0.05 m.
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
        cartesian_goals: list[np.ndarray],
        initial_arm_aa: np.ndarray,
        cartesian_threshold: float = 0.05,
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
        fixed_collar_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        n_diffusion_samples: int = 512,
        n_clusters: int = 3,
        clusterer: TrajectoryClusterer | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
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
            fixed_collar_aa=fixed_collar_aa,
            body_pos=body_pos,
            n_diffusion_samples=n_diffusion_samples,
            n_clusters=n_clusters,
            clusterer=clusterer,
            extra_costs=extra_costs,
        )
        self._init_cartesian(
            cartesian_goals, initial_arm_aa, cartesian_threshold,
            fk, spine3_pos, spine3_aa, fixed_collar_aa,
        )

    # ------------------------------------------------------------------
    # solve override
    # ------------------------------------------------------------------

    def solve(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the best first action and full plan.

        Delegates to the parent (joint-angle cost) while MDM waypoints are
        queued.  Switches to Cartesian cost once the MDM queue is empty.
        """
        if self._goals:
            return super().solve(current_q)
        return self._cartesian_solve(current_q)

    # ------------------------------------------------------------------
    # step override
    # ------------------------------------------------------------------

    def step(
        self,
        current_q: np.ndarray,
        advance_threshold: float | None = None,
    ) -> np.ndarray:
        """Perform one MPC step.

        Tracks MDM waypoints while that queue is non-empty (the last frame can
        be popped, unlike the parent, so the queue can reach empty).  Then
        works through the Cartesian goal queue.
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
        next_q = _compose_rotvec(np.asarray(current_q, dtype=np.float64), first_action)

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
                    collar_aa=self._vis_config.collar_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                    elbow_height_range=self._elbow_height_world_range(),
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
                if self._mdm_goal is not None:
                    self._vis.update_mdm_goal(self._mdm_goal)
                if self._preview_q is not None:
                    self._vis.update_trajectory_preview(self._preview_q)
                if self.current_cartesian_goal is not None:
                    self._vis.update_cartesian_target(
                        self._spine3_pos + self.current_cartesian_goal
                    )
            color = _MDM_COLOR if self._goals else _TARGET_COLOR
            self._vis.update_step(next_q, dist=dist, color=color)

        return next_q

    def _cartesian_step(self, current_q: np.ndarray) -> np.ndarray:
        if not self._cartesian_goals:
            # Nothing left to do — hold position.
            return np.asarray(current_q, dtype=np.float64)

        first_action, _ = self.solve(current_q)
        next_q = _compose_rotvec(np.asarray(current_q, dtype=np.float64), first_action)

        arm_pos = self._fk_inst.fk_controlled(
            next_q, self._fixed_collar_aa, self._spine3_pos, self._spine3_aa
        )
        wrist_rel = arm_pos[-1] - self._spine3_pos
        dist = float(np.linalg.norm(wrist_rel - self.current_cartesian_goal))

        # Advance to next Cartesian goal when current one is reached.
        if dist < self._cartesian_threshold and len(self._cartesian_goals) > 1:
            self._cartesian_goals.popleft()
            self.reset_warmstart()
            dist = float(np.linalg.norm(wrist_rel - self.current_cartesian_goal))

        if self._vis_config is not None:
            if self._vis is None:
                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    self._initial_arm_aa,
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                    collar_aa=self._vis_config.collar_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                    elbow_height_range=self._elbow_height_world_range(),
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
                if self._mdm_goal is not None:
                    self._vis.update_mdm_goal(self._mdm_goal)
            self._vis.update_cartesian_target(
                self._spine3_pos + self.current_cartesian_goal
            )
            self._vis.update_step(next_q, dist=dist, color=_TARGET_COLOR)

        return next_q
