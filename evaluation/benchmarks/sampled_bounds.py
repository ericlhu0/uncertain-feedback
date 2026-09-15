"""Procedural preferences anchored to a nominal reach and a feasible goal pose."""

from __future__ import annotations

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import arm_feature_series
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.simulated_users.base import (
    CoupledBound,
    HiddenBound,
    SimulatedUser,
)
from uncertain_feedback.simulated_users.personas import DEFAULT_ARM_JOINT_LIMITS

_FEATURES = (
    "elbow_flexion",
    "shoulder_elevation",
    "shoulder_abduction_adduction",
    "shoulder_flexion_extension",
)


def sample_bound(
    rng: np.random.Generator,
    naive: np.ndarray,
    goal_pose: np.ndarray,
    context: MpcCostContext,
    name: str,
    family: str,
    min_history: int,
    window: int,
    min_violation: float,
) -> SimulatedUser | None:
    """Place an upper/lower boundary on f or f - slope*g at a first crossing.

    The prefix and goal witness must lie on the comfortable side, while the
    following window must open a substantial gap. Oracle replanning downstream
    still decides whether a comfortable connecting path exists.
    """
    if len(naive) <= min_history + window + 1:
        return None
    features = arm_feature_series(naive, context)
    goal_features = arm_feature_series(goal_pose[None], context)
    for _ in range(128):
        feature, conditioning = rng.choice(_FEATURES, size=2, replace=False)
        feature, conditioning = str(feature), str(conditioning)
        slope = (
            float(rng.choice([-1, 1]) * rng.uniform(0.5, 2.0))
            if family == "coupled"
            else 0.0
        )
        residual = features[feature] - slope * features[conditioning]
        goal_value = float(
            goal_features[feature][0] - slope * goal_features[conditioning][0]
        )
        step = int(rng.integers(min_history + 1, len(naive) - window))
        if (
            family == "coupled"
            and abs(slope) * np.ptp(features[conditioning][step : step + window])
            < min_violation
        ):
            continue
        direction = int(rng.choice([-1, 1]))
        signed = direction * residual
        floor = max(float(signed[:step].max()), direction * goal_value)
        if signed[step] <= floor:
            continue
        boundary = float(rng.uniform(floor, signed[step]))
        if signed[step : step + window].max() - boundary < min_violation:
            continue
        threshold = direction * boundary
        bound_type = "upper_bound" if direction == 1 else "lower_bound"
        bound = (
            CoupledBound(
                feature=feature,
                bound_type=bound_type,
                cond_feature=conditioning,
                intercept=threshold,
                slope=slope,
            )
            if family == "coupled"
            else HiddenBound(
                feature=feature,
                bound_type=bound_type,
                high=threshold if direction == 1 else None,
                low=threshold if direction == -1 else None,
            )
        )
        return SimulatedUser(
            name=name,
            description=f"Synthetic {family} preference; not a clinical persona.",
            feedback_text="",
            bounds=(bound,),
            joint_limits=DEFAULT_ARM_JOINT_LIMITS,
        )
    return None
