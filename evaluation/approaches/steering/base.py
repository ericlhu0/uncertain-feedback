"""Steering ABC: how MDM sampling is steered toward preference bounds."""

from __future__ import annotations

import abc

from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import SteeringConfig, SteeringSpec
from uncertain_feedback.simulated_users import SimulatedUser


class Steering(abc.ABC):
    """One steering method; only the mdm grounder consumes it."""

    # Written into UqConfig.steering.mode by the mdm grounder ("off" or "cg").
    mode: str

    @abc.abstractmethod
    def spec(
        self,
        gen: MotionGenerator,
        user: SimulatedUser,
        config: SteeringConfig,
        seed: int,
    ) -> SteeringSpec | None:
        """The compiled steering spec for this episode, or None for unsteered."""
