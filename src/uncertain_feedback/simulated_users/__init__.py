"""Simulated care recipients with hidden comfort costs for headless experiments."""

from uncertain_feedback.simulated_users.attribution import (
    ATTRIBUTED_FEATURES,
    CorrectionIntent,
    assert_axis_conventions,
    attribute_correction,
    has_feedback_content,
)
from uncertain_feedback.simulated_users.base import (
    BOUND_TYPES,
    FEATURE_NAMES,
    Bound,
    CoupledBound,
    FeatureCondition,
    HiddenBound,
    HiddenCostTerm,
    JointBoxLimit,
    SimulatedUser,
    choose_cluster,
    compute_violations,
    feature_series,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.simulated_users.chooser import ChoiceResult, choose_correction
from uncertain_feedback.simulated_users.personas import PERSONAS, get_persona
from uncertain_feedback.simulated_users.verbalizers import (
    VERBALIZERS,
    Utterance,
    verbalize_everyday,
    verbalize_joint_resolved,
    verbalize_vague,
)
from uncertain_feedback.simulated_users.viz import render_hidden_bounds

__all__ = [
    "ATTRIBUTED_FEATURES",
    "CorrectionIntent",
    "assert_axis_conventions",
    "attribute_correction",
    "has_feedback_content",
    "VERBALIZERS",
    "Utterance",
    "verbalize_everyday",
    "verbalize_joint_resolved",
    "verbalize_vague",
    "BOUND_TYPES",
    "FEATURE_NAMES",
    "Bound",
    "CoupledBound",
    "FeatureCondition",
    "HiddenBound",
    "HiddenCostTerm",
    "JointBoxLimit",
    "SimulatedUser",
    "ChoiceResult",
    "choose_cluster",
    "choose_correction",
    "compute_violations",
    "feature_series",
    "first_violation_step",
    "violation_metrics",
    "PERSONAS",
    "get_persona",
    "render_hidden_bounds",
]
