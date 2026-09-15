"""How far a generated correction pushes the body past the hidden bound.

The grounding stage is scored against the persona's hidden restrictions
directly: every frame of the generated correction is read in the simulated
user's feature space and charged the radians by which it exceeds the bounds,
and those per-frame amounts are averaged over the trajectory. Averaging rather
than summing lets corrections of different lengths compare — MDM clips, LLM
waypoint paths and the naive continuation do not share a clock.

Only the persona's hidden bounds count. The anatomical joint-box limits every
persona carries are a shared plausibility check rather than the preference
being grounded, so they are left out.
"""

from __future__ import annotations

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users import SimulatedUser, feature_series


def correction_violation(
    user: SimulatedUser,
    generated_correction: np.ndarray,
    context: MpcCostContext,
) -> float:
    """Mean per-frame hidden-bound violation over a correction, in radians.

    ``generated_correction`` may be canonical q ``(T, 7)`` or FK-boundary arm
    axis-angles ``(T, 3, 3)``. ``0`` means every frame of the correction stayed
    inside every one of the persona's hidden bounds.
    """
    violations = user.violation_series(feature_series(context, generated_correction))
    return float(np.mean(violations))
