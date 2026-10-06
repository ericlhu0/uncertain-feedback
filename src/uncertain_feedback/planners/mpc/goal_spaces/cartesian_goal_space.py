"""Cartesian goal space: a queue of goal regions in spine3-relative coordinates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from uncertain_feedback.planners.mpc.action_spaces.base import RolloutBatch, StageCost
from uncertain_feedback.planners.mpc.costs.base import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.goal_spaces.base import GoalSpace
from uncertain_feedback.planners.mpc.goal_spaces.regions import (
    GoalRegion,
    as_goal_region,
)
from uncertain_feedback.planners.mpc.human import Human


@dataclass(frozen=True)
class GoalStallConfig:
    """When the goal phase counts as stuck.

    Stuck: still outside the goal, and over the last ``window`` goal steps the
    best distance to the front goal improved by less than ``min_gain`` (goal
    units) while the arm's net joint change stayed under ``max_travel``
    radians. The travel test keeps a detour that moves away from the goal from
    reading as stuck.
    """

    window: int = 40
    min_gain: float = 0.01
    max_travel: float = 0.15


@dataclass(frozen=True)
class CartesianConfig:
    """Goal regions and the slack past a region's boundary that counts as reaching it.

    Each goal is a raw ``[x, y, z]`` spine3-relative wrist point or a
    :class:`GoalRegion`. ``threshold`` is in the region's own units: metres for
    point/box/sphere regions, radians for feature regions. ``stall`` enables
    stall detection (:class:`GoalStallConfig`).
    """

    goals: Sequence[Sequence[float] | np.ndarray | GoalRegion] = ()
    threshold: float = 0.01
    stall: GoalStallConfig | None = None


class CartesianGoalSpace(GoalSpace):
    """A queue of goal regions relative to the spine3 joint.

    Only the terminal state is scored; rotation is unconstrained. The front
    region is popped (distance within ``threshold``) as the queue is worked
    through. With ``stall`` set, :meth:`progress` also tracks the best
    distance so far and the arm state over the last ``stall.window`` steps.
    """

    def __init__(
        self,
        goals: Sequence[Sequence[float] | np.ndarray | GoalRegion],
        threshold: float,
        human: Human,
        stall: GoalStallConfig | None = None,
    ) -> None:
        self._goals: deque[GoalRegion] = deque(as_goal_region(g) for g in goals)
        self._threshold = threshold
        self._human = human
        self._fk = human.fk
        self._spine3_pos = human.spine3_pos
        self._spine3_aa = human.spine3_aa
        self._stall = stall
        self._stall_window: deque[tuple[float, np.ndarray]] = deque(
            maxlen=stall.window + 1 if stall is not None else 0
        )
        self._stall_best = np.inf
        self._last_dist = np.inf

    @property
    def has_goals(self) -> bool:
        return bool(self._goals)

    @property
    def stalled(self) -> bool:
        """Whether the goal phase has stopped progressing (:class:`GoalStallConfig`)."""
        stall = self._stall
        if stall is None or len(self._stall_window) <= stall.window:
            return False
        best_then, q_then = self._stall_window[0]
        best_now, q_now = self._stall_window[-1]
        return (
            self._last_dist >= self._threshold
            and best_then - best_now < stall.min_gain
            and float(np.linalg.norm(q_now - q_then)) < stall.max_travel
        )

    def reset_stall(self) -> None:
        """Start a fresh stall window (new goal, costs or trajectory)."""
        self._stall_window.clear()
        self._stall_best = np.inf

    @property
    def current_goal(self) -> GoalRegion | None:
        """The active goal region, or ``None`` if the queue is empty."""
        return self._goals[0] if self._goals else None

    def append(self, goal: Sequence[float] | np.ndarray | GoalRegion) -> None:
        """Add a goal to the back of the queue."""
        self._goals.append(as_goal_region(goal))

    def _distance(self, q: np.ndarray, region: GoalRegion) -> float:
        arm_aa = self._human.arm_aa_from_q(q)
        wrist_rel = self._human.wrist_from_q(q)
        return float(region.distance(wrist_rel[None], arm_aa[None], self._human)[0])

    def reached(self, q: np.ndarray) -> bool:
        """Whether ``q`` has reached the final goal region.

        The state must be within ``threshold`` of the last remaining region.
        While earlier goals are still queued the rollout has not finished, so
        this returns ``False``.
        """
        goal = self.current_goal
        if goal is None or len(self._goals) > 1:
            return False
        return self._distance(q, goal) < self._threshold

    def progress(
        self, next_q: np.ndarray, on_pop: Callable[[], None]
    ) -> tuple[GoalRegion, float]:
        """Distance to the front goal, popping it when reached.

        Returns the active region and the distance to it, after advancing
        the queue (and calling ``on_pop``, which resets the planner's warm
        start) if ``next_q`` reached the front goal and more goals remain.
        """
        goal = self._goals[0]
        dist = self._distance(next_q, goal)
        if dist < self._threshold and len(self._goals) > 1:
            self._goals.popleft()
            on_pop()
            self.reset_stall()
            goal = self._goals[0]
            dist = self._distance(next_q, goal)
        self._stall_best = min(self._stall_best, dist)
        self._stall_window.append((self._stall_best, np.array(next_q, dtype=float)))
        self._last_dist = dist
        return goal, dist

    def stage_cost(self, extra_costs: CompositeTrajectoryCost) -> StageCost:
        """Squared distance to the current region, plus the extra cost terms.

        Human rollouts locate the wrist by terminal-frame FK; robot rollouts
        already carry projected world wrist positions and use those directly.
        """

        def cost(batch: RolloutBatch) -> np.ndarray:
            target = self.current_goal
            if target is None:
                return np.zeros(batch.aa_trajs.shape[0])
            terminal_aa = batch.aa_trajs[:, -1]
            if batch.wrist_pos is not None:
                wrist_rel = batch.wrist_pos[:, -1] - self._spine3_pos
            else:
                positions = self._fk.fk_batch(
                    terminal_aa, self._spine3_pos, self._spine3_aa
                )
                wrist_rel = positions[:, -1] - self._spine3_pos
            dist = target.distance(wrist_rel, terminal_aa, self._human)
            return dist**2 + extra_costs(batch.aa_trajs)

        return cost
