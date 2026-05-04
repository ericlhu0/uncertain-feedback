"""Pure Cartesian MPC for the SMPL left arm with no MDM or UQ."""

from __future__ import annotations

from collections import deque

import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc import (
    _N_JOINTS,
    _compose_rotvec,
    SmplLeftArmMPC,
)
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.visualizer import ArmVisualizer, _TARGET_COLOR


class LeftArmCartesianMPCNoMDM(SmplLeftArmMPC):
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
            raise ValueError("fk is required for LeftArmCartesianMPCNoMDM.")
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
            extra_costs=extra_costs,
        )
        self._cartesian_goals: deque[np.ndarray] = deque(
            np.asarray(g, dtype=np.float64) for g in cartesian_goals
        )
        self._cartesian_threshold = cartesian_threshold
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
        self._fixed_collar_aa = (
            np.asarray(fixed_collar_aa, dtype=np.float64)
            if fixed_collar_aa is not None
            else np.zeros(3)
        )
        if self._vis_config is not None:
            self._vis_config.body_pos = body_pos

    @property
    def current_cartesian_goal(self) -> np.ndarray | None:
        """The active Cartesian goal, or ``None`` if the queue is empty."""
        return self._cartesian_goals[0] if self._cartesian_goals else None

    def append_cartesian_goal(self, goal: np.ndarray) -> None:
        """Add a Cartesian goal to the back of the queue."""
        self._cartesian_goals.append(np.asarray(goal, dtype=np.float64))

    def _cartesian_cost(self, q_trajs: np.ndarray) -> np.ndarray:
        """L2 Cartesian cost from terminal wrist to current spine3-relative goal."""
        target = self.current_cartesian_goal
        if target is None:
            return np.zeros(q_trajs.shape[0])
        terminal_q = q_trajs[:, -1]
        positions = self._fk_inst.fk_controlled_batch(
            terminal_q, self._fixed_collar_aa, self._spine3_pos, self._spine3_aa
        )
        wrist_rel = positions[:, -1] - self._spine3_pos
        wrist_cost = ((wrist_rel - target) ** 2).sum(axis=-1)
        return wrist_cost + self._extra_costs(q_trajs)

    def solve(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the best first action and full Cartesian plan."""
        current_q = np.asarray(current_q, dtype=np.float64)

        if self._prev_best is not None:
            mean = np.concatenate(
                [self._prev_best[1:], np.zeros((1, _N_JOINTS, 3))], axis=0
            )
        else:
            mean = np.zeros((self._config.horizon, _N_JOINTS, 3), dtype=np.float64)

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
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
            self._vis.update_cartesian_target(
                self._spine3_pos + self.current_cartesian_goal
            )
            self._vis.update_step(next_q, dist=dist, color=_TARGET_COLOR)

        return next_q
