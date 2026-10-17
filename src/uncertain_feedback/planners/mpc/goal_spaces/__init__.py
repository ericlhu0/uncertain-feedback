"""Goal spaces for the sampling MPC: what the solve loop steers toward."""

from uncertain_feedback.planners.mpc.goal_spaces.base import GoalSpace
from uncertain_feedback.planners.mpc.goal_spaces.cartesian_goal_space import (
    CartesianConfig,
    CartesianGoalSpace,
    GoalStallConfig,
)
from uncertain_feedback.planners.mpc.goal_spaces.regions import (
    BoxRegion,
    DisplacementRegion,
    FeatureRegion,
    ForearmBoxRegion,
    GoalRegion,
    PointRegion,
    SphereRegion,
    as_goal_region,
    goal_point,
)

__all__ = [
    "GoalSpace",
    "CartesianConfig",
    "CartesianGoalSpace",
    "GoalStallConfig",
    "GoalRegion",
    "PointRegion",
    "BoxRegion",
    "ForearmBoxRegion",
    "SphereRegion",
    "FeatureRegion",
    "DisplacementRegion",
    "as_goal_region",
    "goal_point",
]
