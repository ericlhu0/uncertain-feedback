"""Shared Cartesian wrist-goal state and logic for Cartesian MPC variants."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    q_to_arm_aa,
)


class _CartesianGoalsMixin:
    """Mixin providing shared Cartesian wrist-goal logic.

    Must be combined with a :class:`~uncertain_feedback.planners.mpc.arm_mpc.SmplLeftArmMPC`
    subclass, which supplies the state and methods declared below.
    """

    # Supplied by the SmplLeftArmMPC subclass; annotations only, so the mixin
    # never shadows the real attributes at runtime.
    _prev_best: np.ndarray | None
    _horizon: int
    _n_mpc_samples: int
    _max_angle_delta: float
    _extra_costs: CompositeTrajectoryCost
    _sample_actions: Callable[[np.ndarray, tuple[int, ...]], np.ndarray]
    _rollout: Callable[[np.ndarray, np.ndarray], np.ndarray]
    reset_warmstart: Callable[[], None]

    def _init_cartesian(
        self,
        cartesian_goals: list[np.ndarray],
        initial_q: np.ndarray,
        cartesian_threshold: float,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
    ) -> None:
        self._cartesian_goals: deque[np.ndarray] = deque(
            np.asarray(g, dtype=np.float64) for g in cartesian_goals
        )
        self._cartesian_threshold = cartesian_threshold
        self._initial_q = np.asarray(initial_q, dtype=np.float64)
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

    @property
    def current_cartesian_goal(self) -> np.ndarray | None:
        """The active Cartesian goal, or ``None`` if the queue is empty."""
        return self._cartesian_goals[0] if self._cartesian_goals else None

    def append_cartesian_goal(self, goal: np.ndarray) -> None:
        """Add a Cartesian goal to the back of the queue."""
        self._cartesian_goals.append(np.asarray(goal, dtype=np.float64))

    def goal_reached(self, q: np.ndarray) -> bool:
        """Whether the wrist has reached the final Cartesian goal.

        Overrides the joint-space check on
        :class:`~uncertain_feedback.planners.mpc.arm_mpc.SmplLeftArmMPC`: the
        wrist (spine3-relative) must be within ``cartesian_threshold`` of the
        last remaining goal. While earlier goals are still queued the rollout
        has not finished, so this returns ``False``.
        """
        goal = self.current_cartesian_goal
        if goal is None or len(self._cartesian_goals) > 1:
            return False
        arm_pos = self._fk_inst.fk(
            q_to_arm_aa(q, self._fk_inst.elbow_hinge_axis),
            self._spine3_pos,
            self._spine3_aa,
        )
        wrist_rel = arm_pos[-1] - self._spine3_pos
        return float(np.linalg.norm(wrist_rel - goal)) < self._cartesian_threshold

    def _cartesian_progress(self, next_q: np.ndarray) -> tuple[np.ndarray, float]:
        """Distance to the front goal, popping it when reached.

        Returns the active goal and the wrist distance to it, after advancing
        the queue (and resetting the warm start) if ``next_q`` reached the
        front goal and more goals remain.
        """
        arm_pos = self._fk_inst.fk(
            q_to_arm_aa(next_q, self._fk_inst.elbow_hinge_axis),
            self._spine3_pos,
            self._spine3_aa,
        )
        wrist_rel = arm_pos[-1] - self._spine3_pos
        goal = self._cartesian_goals[0]
        dist = float(np.linalg.norm(wrist_rel - goal))
        if dist < self._cartesian_threshold and len(self._cartesian_goals) > 1:
            self._cartesian_goals.popleft()
            self.reset_warmstart()
            goal = self._cartesian_goals[0]
            dist = float(np.linalg.norm(wrist_rel - goal))
        return goal, dist

    def _cartesian_cost(self, q_trajs: np.ndarray) -> np.ndarray:
        """L2 Cartesian cost: spine3-relative wrist distance to current target.

        Args:
            q_trajs: ``(N, H+1, 7)`` state trajectories.

        Returns:
            ``(N,)`` cost per trajectory.
        """
        target = self.current_cartesian_goal
        if target is None:
            return np.zeros(q_trajs.shape[0])
        aa_trajs = q_to_arm_aa(q_trajs, self._fk_inst.elbow_hinge_axis)
        terminal_aa = aa_trajs[:, -1]
        positions = self._fk_inst.fk_batch(
            terminal_aa, self._spine3_pos, self._spine3_aa
        )
        wrist_rel = positions[:, -1] - self._spine3_pos
        wrist_cost = ((wrist_rel - target) ** 2).sum(axis=-1)
        return wrist_cost + self._extra_costs(aa_trajs)

    def _cartesian_solve(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample and return the best Cartesian action.

        Args:
            current_q: ``(7,)`` current planner state.

        Returns:
            Tuple of first action ``(7,)`` and full plan ``(H, 7)``.
        """
        current_q = np.asarray(current_q, dtype=np.float64)

        prev_best = self._prev_best
        if prev_best is not None:
            mean = np.concatenate([prev_best[1:], np.zeros((1, Q_DIM))], axis=0)
        else:
            mean = np.zeros((self._horizon, Q_DIM), dtype=np.float64)

        actions = self._sample_actions(
            mean,
            (self._n_mpc_samples, self._horizon, Q_DIM),
        )
        q_trajs = self._rollout(current_q, actions)
        costs = self._cartesian_cost(q_trajs)
        best_idx = np.argmin(costs)
        best_plan = actions[best_idx]
        self._prev_best = best_plan
        return best_plan[0], best_plan
