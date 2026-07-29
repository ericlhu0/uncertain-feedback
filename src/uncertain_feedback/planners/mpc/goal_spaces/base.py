"""Goal-space contract: what the MPC steers toward once feedback is done."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from uncertain_feedback.planners.mpc.action_spaces.base import StageCost
from uncertain_feedback.planners.mpc.costs.base import CompositeTrajectoryCost


class GoalSpace(ABC):
    """A queue of goals plus the stage cost that steers rollouts toward them."""

    @property
    @abstractmethod
    def has_goals(self) -> bool:
        """Whether any goal remains in the queue."""

    @abstractmethod
    def stage_cost(self, extra_costs: CompositeTrajectoryCost) -> StageCost:
        """Cost toward the front goal, plus the configured extra cost terms."""

    @abstractmethod
    def reached(self, q: np.ndarray) -> bool:
        """Whether ``q`` has reached the final goal so the rollout can end."""

    @abstractmethod
    def progress(
        self, next_q: np.ndarray, on_pop: Callable[[], None]
    ) -> tuple[np.ndarray, float]:
        """Distance to the front goal, popping it (and calling ``on_pop``)
        when reached."""
