"""Classifier-guidance steering of MDM sampling from the persona's hidden bounds."""

from __future__ import annotations

from dataclasses import replace

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

    The known-limits ceiling (``full_hidden_limit_steering``,
    ``no_learning_cg``), never the method: the only steering that reads ground
    truth, and it hands the hidden bounds to the compiler explicitly.
    """

    mode = "cg"

    def spec(
        self,
        gen: MotionGenerator,
        user: SimulatedUser,
        config: SteeringConfig,
        seed: int,
    ) -> SteeringSpec | None:
        spec = build_steering_spec(gen, replace(config, bounds=user.bounds), seed=seed)
        if spec is None:
            print(
                f"[evaluation] steering unsupported for persona {user.name}; "
                "sampling unsteered.",
                flush=True,
            )
        return spec
