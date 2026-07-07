"""Simulated care recipients with hidden comfort costs for headless experiments."""

from uncertain_feedback.simulated_users.base import (
    BOUND_TYPES,
    FEATURE_NAMES,
    FeatureCondition,
    HiddenBound,
    HiddenCostTerm,
    SimulatedUser,
    choose_cluster,
    compute_violations,
    feature_series,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.simulated_users.personas import PERSONAS, get_persona

__all__ = [
    "BOUND_TYPES",
    "FEATURE_NAMES",
    "FeatureCondition",
    "HiddenBound",
    "HiddenCostTerm",
    "SimulatedUser",
    "choose_cluster",
    "compute_violations",
    "feature_series",
    "first_violation_step",
    "violation_metrics",
    "PERSONAS",
    "get_persona",
]
