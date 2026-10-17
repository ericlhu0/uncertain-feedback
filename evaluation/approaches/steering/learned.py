"""Classifier-guidance steering of MDM sampling with the learned preference."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from evaluation.approaches.steering.base import Steering
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import (
    SteeringConfig,
    SteeringSpec,
    build_steering_spec,
)
from uncertain_feedback.simulated_users import SimulatedUser
from uncertain_feedback.simulated_users.base import Bound


class LearnedSteering(Steering):
    """Guide diffusion with the bounds the approach has learned so far.

    :class:`Approach` binds ``learned_bounds`` to its cost learner. It is read
    at every generation, so steering follows each round's learning and a
    resumed run steers from its first round. The persona's hidden limits are
    never read: with nothing learned, sampling is unsteered.
    """

    mode = "cg"

    def __init__(self) -> None:
        self.learned_bounds: Callable[[], tuple[Bound, ...]] = lambda: ()

    def spec(
        self,
        gen: MotionGenerator,
        user: SimulatedUser,
        config: SteeringConfig,
        seed: int,
    ) -> SteeringSpec | None:
        bounds = self.learned_bounds()
        if not bounds:
            return None
        return build_steering_spec(gen, user, replace(config, bounds=bounds), seed=seed)
