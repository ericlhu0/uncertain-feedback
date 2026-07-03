"""MPC cost terms, the LLM-generated cost pipeline, and prompt templates.

This package is the single public surface for everything cost-related: import from
``uncertain_feedback.planners.mpc.costs`` rather than the submodules.

- ``base`` — hand-authored cost terms, the cost registry, and preference learning.
- ``llm_costs`` — the LLM-generated Python cost pipeline (context, summaries,
  compilation, overlay rendering).
- ``prompts`` — prompt templates loaded from ``.txt`` files.
"""

from uncertain_feedback.planners.mpc.costs.base import (
    CompositeTrajectoryCost,
    ElbowFlexionAngleCost,
    ElbowHeightCost,
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
    build_motion_summaries,
    compile_generated_cost,
    parse_llm_cost_response,
    render_prompt_images,
)
from uncertain_feedback.planners.mpc.costs.cost_generator import (
    CostGenerator,
    artifact_run_dir,
    create_cost_generator,
    evaluate_and_render,
    evaluate_candidate_cost,
)
from uncertain_feedback.planners.mpc.costs.cost_feedback import EvalState
from uncertain_feedback.planners.mpc.costs.llm_costs import LlmCostGenerator
from uncertain_feedback.planners.mpc.costs.turns_costs import TurnsCostGenerator
from uncertain_feedback.planners.mpc.costs.agent_costs import AgentCostGenerator
from uncertain_feedback.planners.mpc.costs.prompts import (
    IMAGE_PLACEHOLDERS,
    build_author_prompt,
    build_ground_prompt,
    build_interpret_prompt,
    build_refine_prompt,
    build_staged_task_body,
    compact_summaries,
)

__all__ = [
    # base
    "CompositeTrajectoryCost",
    "ElbowFlexionAngleCost",
    "ElbowHeightCost",
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
    # generated primitives
    "GeneratedCostContext",
    "GeneratedCostValidationError",
    "GeneratedPythonCost",
    "LlmCostResponse",
    "build_generated_cost_context",
    "build_motion_summaries",
    "compile_generated_cost",
    "parse_llm_cost_response",
    "render_prompt_images",
    # cost generators
    "CostGenerator",
    "LlmCostGenerator",
    "TurnsCostGenerator",
    "AgentCostGenerator",
    "create_cost_generator",
    "evaluate_candidate_cost",
    "evaluate_and_render",
    "EvalState",
    "artifact_run_dir",
    # prompts
    "IMAGE_PLACEHOLDERS",
    "build_interpret_prompt",
    "build_ground_prompt",
    "build_author_prompt",
    "build_refine_prompt",
    "build_staged_task_body",
    "compact_summaries",
]
