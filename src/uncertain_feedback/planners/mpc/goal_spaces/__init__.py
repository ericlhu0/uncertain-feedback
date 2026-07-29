"""Goal spaces for the sampling MPC: what the solve loop steers toward."""

from uncertain_feedback.planners.mpc.goal_spaces.base import GoalSpace
from uncertain_feedback.planners.mpc.goal_spaces.cartesian_goal_space import (
    CartesianConfig,
    CartesianGoalSpace,
)

__all__ = ["GoalSpace", "CartesianConfig", "CartesianGoalSpace"]
