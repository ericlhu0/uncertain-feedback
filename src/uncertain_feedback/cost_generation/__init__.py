"""The cost-generation stage: a user correction becomes an executable MPC cost.

- ``generate`` — the stage entry point (:func:`generate_cost_for_cluster`).
- ``base`` — the shared :class:`CostGenerator` and the backend selector.
- ``llm_costs`` / ``turns_costs`` / ``agent_costs`` — the three backends.
- ``combine_costs`` — unify several rounds' costs into one replacement cost.
- ``summaries`` — the JSON summaries and overlay images that ground a prompt.
- ``corpus`` — the on-disk log of executed trajectories used as accepted-pose evidence.
- ``prompts`` — prompt templates loaded from ``.txt`` files.

Depends on ``planners`` and ``evaluation_mechanism``; neither depends on this.
"""

from uncertain_feedback.cost_generation.agent_costs import AgentCostGenerator
from uncertain_feedback.cost_generation.base import (
    CostGenerator,
    artifact_run_dir,
    create_cost_generator,
    parse_goal_conflict,
)
from uncertain_feedback.cost_generation.combine_costs import (
    CombineCostGenerator,
    CostRound,
)
from uncertain_feedback.cost_generation.corpus import TrajectoryCorpus
from uncertain_feedback.cost_generation.generate import (
    CostGenerationResult,
    generate_cost_for_cluster,
)
from uncertain_feedback.cost_generation.llm_costs import LlmCostGenerator
from uncertain_feedback.cost_generation.summaries import (
    build_motion_summaries,
    build_rollout_joint_comparison,
    render_prompt_images,
)
from uncertain_feedback.cost_generation.turns_costs import TurnsCostGenerator

__all__ = [
    "AgentCostGenerator",
    "CombineCostGenerator",
    "CostGenerationResult",
    "CostGenerator",
    "CostRound",
    "LlmCostGenerator",
    "TrajectoryCorpus",
    "TurnsCostGenerator",
    "artifact_run_dir",
    "build_motion_summaries",
    "build_rollout_joint_comparison",
    "create_cost_generator",
    "generate_cost_for_cluster",
    "parse_goal_conflict",
    "render_prompt_images",
]
