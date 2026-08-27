"""Classifier-guidance steering of MDM sampling."""

from __future__ import annotations

from evaluation.approaches.steering.base import Steering
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import (
    SteeringConfig,
    SteeringSpec,
    build_steering_spec,
)
from uncertain_feedback.simulated_users import SimulatedUser


class ClassifierGuidanceSteering(Steering):
    """Guide diffusion with the persona's hidden bounds.

    The persona's bounds are the only steering-cost source wired today, so a
    steered run is the known-preference upper bound; steering from the learned
    cost plugs in here once that wiring exists.
    """

    mode = "cg"

    def spec(
        self,
        gen: MotionGenerator,
        user: SimulatedUser,
        config: SteeringConfig,
        seed: int,
    ) -> SteeringSpec | None:
        spec = build_steering_spec(gen, user, config, seed=seed)
        if spec is None:
            print(
                f"[evaluation] steering unsupported for persona {user.name}; "
                "sampling unsteered.",
                flush=True,
            )
        return spec
