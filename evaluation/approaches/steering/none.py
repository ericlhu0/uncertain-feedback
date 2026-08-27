"""No steering: MDM samples are drawn unsteered."""

from __future__ import annotations

from evaluation.approaches.steering.base import Steering
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import SteeringConfig, SteeringSpec
from uncertain_feedback.simulated_users import SimulatedUser


class NoSteering(Steering):
    mode = "off"

    def spec(
        self,
        gen: MotionGenerator,
        user: SimulatedUser,
        config: SteeringConfig,
        seed: int,
    ) -> SteeringSpec | None:
        del gen, user, config, seed
        return None
