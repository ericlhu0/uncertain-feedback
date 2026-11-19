"""Classifier-guidance steering of MDM sampling with the learned preference."""

from __future__ import annotations

from typing import Callable

from evaluation.approaches.steering.base import Steering
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import (
    SteeringConfig,
    SteeringSpec,
    build_learned_steering_spec,
)
from uncertain_feedback.planners.mpc.costs import GeneratedPythonCost
from uncertain_feedback.simulated_users import SimulatedUser


class LearnedSteering(Steering):
    """Guide diffusion with the learned costs the planner is using.

    :class:`Approach` binds ``learned_costs`` to its cost learner. It is read
    at every generation, so steering follows each round's learning and a
    resumed run steers from its first round. The persona's hidden limits are
    never read: with nothing learned, sampling is unsteered.
    """

    mode = "cg"

    def __init__(self) -> None:
        self.learned_costs: Callable[[], list[GeneratedPythonCost]] = list

    def spec(
        self,
        gen: MotionGenerator,
        user: SimulatedUser,
        config: SteeringConfig,
        seed: int,
    ) -> SteeringSpec | None:
        del user
        return build_learned_steering_spec(gen, self.learned_costs(), config, seed)
