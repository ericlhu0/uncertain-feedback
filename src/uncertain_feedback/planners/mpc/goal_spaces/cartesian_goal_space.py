"""Cartesian goal space: spine3-relative wrist positions, rotation-free."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from uncertain_feedback.planners.mpc.action_spaces.base import RolloutBatch, StageCost
from uncertain_feedback.planners.mpc.costs.base import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.goal_spaces.base import GoalSpace
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa


@dataclass(frozen=True)
class CartesianConfig:
    """Cartesian wrist goals and the distance that counts as reaching one."""

    goals: list[list[float]] = field(default_factory=list)
    threshold: float = 0.05


class CartesianGoalSpace(GoalSpace):
    """A queue of Cartesian wrist goals relative to the spine3 joint.

    Only end-effector position is optimised; rotation is unconstrained. The
    front goal is popped (within ``threshold``, L2 metres) as the queue is
    worked through.
    """

    def __init__(
        self,
        goals: list[np.ndarray],
        threshold: float,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
    ) -> None:
        self._goals: deque[np.ndarray] = deque(
            np.asarray(g, dtype=np.float64) for g in goals
        )
        self._threshold = threshold
        self._fk = fk
        self._spine3_pos = spine3_pos
        self._spine3_aa = spine3_aa

    @property
    def has_goals(self) -> bool:
        return bool(self._goals)

    @property
    def current_goal(self) -> np.ndarray | None:
        """The active Cartesian goal, or ``None`` if the queue is empty."""
        return self._goals[0] if self._goals else None

    def append(self, goal: np.ndarray) -> None:
        """Add a Cartesian goal to the back of the queue."""
        self._goals.append(np.asarray(goal, dtype=np.float64))

    def _wrist_rel(self, q: np.ndarray) -> np.ndarray:
        arm_pos = self._fk.fk(
            q_to_arm_aa(q, self._fk.elbow_hinge_axis),
            self._spine3_pos,
            self._spine3_aa,
        )
        return arm_pos[-1] - self._spine3_pos

    def reached(self, q: np.ndarray) -> bool:
        """Whether the wrist has reached the final Cartesian goal.

        The wrist (spine3-relative) must be within ``threshold`` of the last
        remaining goal. While earlier goals are still queued the rollout has
        not finished, so this returns ``False``.
        """
        goal = self.current_goal
        if goal is None or len(self._goals) > 1:
            return False
        return float(np.linalg.norm(self._wrist_rel(q) - goal)) < self._threshold

    def progress(
        self, next_q: np.ndarray, on_pop: Callable[[], None]
    ) -> tuple[np.ndarray, float]:
        """Distance to the front goal, popping it when reached.

        Returns the active goal and the wrist distance to it, after advancing
        the queue (and calling ``on_pop``, which resets the planner's warm
        start) if ``next_q`` reached the front goal and more goals remain.
        """
        wrist_rel = self._wrist_rel(next_q)
        goal = self._goals[0]
        dist = float(np.linalg.norm(wrist_rel - goal))
        if dist < self._threshold and len(self._goals) > 1:
            self._goals.popleft()
            on_pop()
            goal = self._goals[0]
            dist = float(np.linalg.norm(wrist_rel - goal))
        return goal, dist

    def stage_cost(self, extra_costs: CompositeTrajectoryCost) -> StageCost:
        """L2 wrist distance to the current target, plus the extra cost terms.

        Human rollouts locate the wrist by terminal-frame FK; robot rollouts
        already carry projected world wrist positions and use those directly.
        """

        def cost(batch: RolloutBatch) -> np.ndarray:
            target = self.current_goal
            if target is None:
                return np.zeros(batch.aa_trajs.shape[0])
            if batch.wrist_pos is not None:
                wrist_rel = batch.wrist_pos[:, -1] - self._spine3_pos
            else:
                positions = self._fk.fk_batch(
                    batch.aa_trajs[:, -1], self._spine3_pos, self._spine3_aa
                )
                wrist_rel = positions[:, -1] - self._spine3_pos
            wrist_cost = ((wrist_rel - target) ** 2).sum(axis=-1)
            return wrist_cost + extra_costs(batch.aa_trajs)

        return cost
