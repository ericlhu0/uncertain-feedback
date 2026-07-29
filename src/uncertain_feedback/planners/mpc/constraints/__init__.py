"""Feasibility constraints for the sampling MPC.

Mirrors the ``COST_BUILDERS`` pattern: the YAML ``constraints:`` section maps
constraint names to parameter mappings; :data:`CONSTRAINT_BUILDERS` maps each
name to its ``(config dataclass, constraint class)`` pair. The config is
parsed at load time and the constraint is built by the planner, which supplies
the runtime objects (env, fk, spine3 frame).
"""

from typing import Callable

from uncertain_feedback.planners.mpc.constraints.base import FeasibilityConstraint
from uncertain_feedback.planners.mpc.constraints.robot_ik_feasibility_constraint import (
    RobotIkConfig,
    RobotIkConstraint,
)

ConstraintBuilder = Callable[..., FeasibilityConstraint]

CONSTRAINT_BUILDERS: dict[str, tuple[type, ConstraintBuilder]] = {
    "robot_ik": (RobotIkConfig, RobotIkConstraint),
}

__all__ = [
    "FeasibilityConstraint",
    "RobotIkConfig",
    "RobotIkConstraint",
    "CONSTRAINT_BUILDERS",
    "ConstraintBuilder",
]
