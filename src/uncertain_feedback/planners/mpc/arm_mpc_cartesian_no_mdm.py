"""Pure Cartesian MPC for the controlled SMPL arm with no MDM or UQ."""

from __future__ import annotations

import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC
from uncertain_feedback.planners.mpc.arm_mpc_cartesian_base import _CartesianGoalsMixin
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, _compose_rotvec
from uncertain_feedback.planners.mpc.visualizer import ArmVisualizer, _TARGET_COLOR


class ArmMPCCartesianNoMDM(_CartesianGoalsMixin, SmplLeftArmMPC):
    """Pure Cartesian wrist-goal MPC with no motion generation or UQ."""

    def __init__(
        self,
        cartesian_goals: list[np.ndarray],
        initial_arm_aa: np.ndarray,
        cartesian_threshold: float = 0.05,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        goal_threshold: float = 0.1,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        fixed_collar_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
    ) -> None:
        if fk is None:
            raise ValueError("fk is required for ArmMPCCartesianNoMDM.")
        super().__init__(
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            goals=[],
            goal_threshold=goal_threshold,
            visualize=visualize,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            fixed_collar_aa=fixed_collar_aa,
            body_pos=body_pos,
            extra_costs=extra_costs,
        )
        self._init_cartesian(
            cartesian_goals, initial_arm_aa, cartesian_threshold,
            fk, spine3_pos, spine3_aa, fixed_collar_aa,
        )

    def solve(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the best first action and full Cartesian plan."""
        return self._cartesian_solve(current_q)

    def step(self, current_q: np.ndarray) -> np.ndarray:
        """Perform one pure Cartesian MPC step."""
        if not self._cartesian_goals:
            return np.asarray(current_q, dtype=np.float64)

        first_action, _ = self.solve(current_q)
        next_q = _compose_rotvec(np.asarray(current_q, dtype=np.float64), first_action)

        arm_pos = self._fk_inst.fk_controlled(
            next_q, self._fixed_collar_aa, self._spine3_pos, self._spine3_aa
        )
        wrist_rel = arm_pos[-1] - self._spine3_pos
        dist = float(np.linalg.norm(wrist_rel - self.current_cartesian_goal))

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
            self._vis.update_cartesian_target(
                self._spine3_pos + self.current_cartesian_goal
            )
            self._vis.update_step(next_q, dist=dist, color=_TARGET_COLOR)

        return next_q
