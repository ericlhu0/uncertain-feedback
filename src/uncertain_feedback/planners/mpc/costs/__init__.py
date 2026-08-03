"""MPC cost terms and the compiled-cost runtime.

The planner's own cost surface: import from
``uncertain_feedback.planners.mpc.costs`` rather than the submodules.

- ``base`` — hand-authored cost terms, the cost registry, and preference learning.
- ``generated`` — the runtime for LLM-authored Python costs (context, compilation,
  response parsing).

Authoring a cost lives in :mod:`uncertain_feedback.cost_generation`; scoring one
lives in :mod:`uncertain_feedback.evaluation_mechanism`. Neither is imported here —
the planner must stay usable without them (the ``agent`` backend's sandbox stages
this package but not those).
"""

from uncertain_feedback.planners.mpc.costs.base import (
    CompositeTrajectoryCost,
    ElbowFlexionAngleCost,
    ElbowHeightCost,
    JointLimitCost,
    LearnablePreferenceCost,
    MpcCostContext,
    ShoulderAbductionAngleCost,
    TrajectoryCost,
    available_cost_names,
    build_extra_costs,
    compute_elbow_flexion_angles,
    compute_elbow_heights,
    compute_shoulder_abduction_angles,
    replace_cost_in_composite,
    update_elbow_cost,
    update_preference_cost,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedCostValidationError,
    GeneratedPythonCost,
    LlmCostResponse,
    build_generated_cost_context,
    compile_generated_cost,
    extract_json_object,
    generated_cost_feature_dependencies,
    parse_llm_cost_response,
    replace_generated_costs,
)

__all__ = [
    # base
    "CompositeTrajectoryCost",
    "ElbowFlexionAngleCost",
    "ElbowHeightCost",
    "JointLimitCost",
    "LearnablePreferenceCost",
    "MpcCostContext",
    "ShoulderAbductionAngleCost",
    "TrajectoryCost",
    "available_cost_names",
    "build_extra_costs",
    "compute_elbow_flexion_angles",
    "compute_elbow_heights",
    "compute_shoulder_abduction_angles",
    "replace_cost_in_composite",
    "update_elbow_cost",
    "update_preference_cost",
    # generated-cost runtime
    "GeneratedCostContext",
    "GeneratedCostValidationError",
    "GeneratedPythonCost",
    "LlmCostResponse",
    "build_generated_cost_context",
    "compile_generated_cost",
    "extract_json_object",
    "generated_cost_feature_dependencies",
    "parse_llm_cost_response",
    "replace_generated_costs",
]
