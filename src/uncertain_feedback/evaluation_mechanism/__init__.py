"""How the method evaluates its own generated cost functions.

Part of the method, not of the researcher's tooling: the cost generators call this
to score and visually compare the costs they author, and the ``agent`` backend's
codex sandbox imports it to run :mod:`render_cost_comparison` on its own candidates.

- ``eval_state`` — the picklable rollout/context bundle handed to an off-process agent.
- ``scoring`` — ranking, L2 rollout scoring, goal-reach checks, overlay rendering.
- ``render_cost_comparison`` — the CLI the ``agent`` backend runs on each candidate.

Imports the planner layer only; ``cost_generation`` imports this, never the reverse.
"""

from uncertain_feedback.evaluation_mechanism.eval_state import EvalMpcConfig, EvalState
from uncertain_feedback.evaluation_mechanism.scoring import (
    CostRanking,
    evaluate_and_render,
    evaluate_candidate_cost,
    goal_reach_report,
    rank_candidate_cost,
    reference_with_correction_traj,
)

__all__ = [
    "CostRanking",
    "EvalMpcConfig",
    "EvalState",
    "evaluate_and_render",
    "evaluate_candidate_cost",
    "goal_reach_report",
    "rank_candidate_cost",
    "reference_with_correction_traj",
]
