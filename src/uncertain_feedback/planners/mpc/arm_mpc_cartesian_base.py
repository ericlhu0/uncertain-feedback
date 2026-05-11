"""Shared Cartesian wrist-goal state and logic for Cartesian MPC variants."""

from __future__ import annotations

from collections import deque

import numpy as np

from uncertain_feedback.planners.mpc.kinematics import (
    SmplLeftArmFK,
    _N_JOINTS,
)


class _CartesianGoalsMixin:
    """Mixin providing shared Cartesian wrist-goal logic.

    Must be combined with a :class:`~uncertain_feedback.planners.mpc.arm_mpc.SmplLeftArmMPC`
    subclass so that ``_prev_best``, ``_horizon``, ``_n_mpc_samples``,
    ``_max_angle_delta``, ``_extra_costs``, and ``_rollout()`` are available.
    """

    def _init_cartesian(
        self,
        cartesian_goals: list[np.ndarray],
        initial_arm_aa: np.ndarray,
        cartesian_threshold: float,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        fixed_collar_aa: np.ndarray | None,
    ) -> None:
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

    @property
    def current_cartesian_goal(self) -> np.ndarray | None:
        """The active Cartesian goal, or ``None`` if the queue is empty."""
        return self._cartesian_goals[0] if self._cartesian_goals else None

    def append_cartesian_goal(self, goal: np.ndarray) -> None:
        """Add a Cartesian goal to the back of the queue."""
        self._cartesian_goals.append(np.asarray(goal, dtype=np.float64))

    def _cartesian_cost(self, q_trajs: np.ndarray) -> np.ndarray:
        """L2 Cartesian cost: spine3-relative wrist distance to current target.

        Args:
            q_trajs: ``(N, H+1, 4, 3)`` state trajectories.

        Returns:
            ``(N,)`` cost per trajectory.
        """
        target = self.current_cartesian_goal
        if target is None:
            return np.zeros(q_trajs.shape[0])
        terminal_q = q_trajs[:, -1]
        positions = self._fk_inst.fk_controlled_batch(
            terminal_q, self._fixed_collar_aa, self._spine3_pos, self._spine3_aa
        )
        wrist_rel = positions[:, -1] - self._spine3_pos
        wrist_cost = ((wrist_rel - target) ** 2).sum(axis=-1)
        return wrist_cost + self._extra_costs(q_trajs)  # type: ignore[attr-defined]

    def _cartesian_solve(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample and return the best Cartesian action.

        Args:
            current_q: ``(3, 3)`` current axis-angle joint angles.

        Returns:
            Tuple of first action ``(3, 3)`` and full plan ``(H, 3, 3)``.
        """
        current_q = np.asarray(current_q, dtype=np.float64)

        prev_best = self._prev_best  # type: ignore[attr-defined]
        if prev_best is not None:
            mean = np.concatenate(
                [prev_best[1:], np.zeros((1, _N_JOINTS, 3))], axis=0
            )
        else:
            mean = np.zeros(
                (self._horizon, _N_JOINTS, 3), dtype=np.float64  # type: ignore[attr-defined]
            )

        actions = np.random.normal(
            loc=mean,
            scale=self._max_angle_delta,  # type: ignore[attr-defined]
            size=(self._n_mpc_samples, self._horizon, _N_JOINTS, 3),  # type: ignore[attr-defined]
        )
        q_trajs = self._rollout(current_q, actions)  # type: ignore[attr-defined]
        costs = self._cartesian_cost(q_trajs)
        best_idx = np.argmin(costs)
        best_plan = actions[best_idx]
        self._prev_best = best_plan  # type: ignore[attr-defined]
        return best_plan[0], best_plan
