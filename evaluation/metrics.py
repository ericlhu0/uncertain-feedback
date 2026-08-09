"""Flatten per-round evaluation records into analysis-ready rows."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.structs import GroundingResult, InteractionTask, LearnOutcome
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    SimulatedUser,
    oracle_cluster_scores,
    violation_metrics,
)


def round_row(
    *,
    task: InteractionTask,
    goal_index: int,
    round_index: int,
    event_index: int,
    user: SimulatedUser,
    context: MpcCostContext,
    utterance_text: str,
    utterance_form: str,
    grounding: GroundingResult,
    choice: ChoiceResult,
    outcome: LearnOutcome,
    continuation: np.ndarray,
    retrigger_step: int | None,
    ground_seconds: float,
    learn_seconds: float,
) -> dict[str, Any]:
    """One flat record per feedback round; the unit the analysis aggregates."""
    hidden_scores = oracle_cluster_scores(
        user, context, grounding.candidates, grounding.magnitude
    )
    continuation_metrics = violation_metrics(user, context, continuation)
    n_acceptable = sum(1 for ok in choice.acceptable.values() if ok)
    return {
        "persona": task.persona,
        "verbalizer": task.verbalizer,
        "goal_index": goal_index,
        "round_index": round_index,
        "event_index": event_index,
        "utterance_form": utterance_form,
        "utterance_text": utterance_text,
        "n_candidates": len(grounding.candidates),
        "n_acceptable": n_acceptable,
        "any_acceptable": n_acceptable > 0,
        "no_acceptable_cluster": choice.no_acceptable_cluster,
        "chosen_label": grounding.chosen_label,
        "magnitude": grounding.magnitude,
        "correction_alignment": float(
            choice.alignment.get(grounding.chosen_label, np.nan)
        ),
        "best_alignment": (
            float(max(choice.alignment.values())) if choice.alignment else np.nan
        ),
        "candidate_hidden_mean": float(np.mean(list(hidden_scores.values()))),
        "candidate_hidden_min": float(np.min(list(hidden_scores.values()))),
        "chosen_hidden": float(hidden_scores[grounding.chosen_label]),
        "cost_accepted": outcome.cost_accepted,
        "unified_installed": outcome.unified_installed,
        "continuation_mean_violation": float(continuation_metrics["mean_violation"]),
        "continuation_max_violation": float(continuation_metrics["max_violation"]),
        "continuation_frac_violated": float(
            continuation_metrics["frac_frames_violated"]
        ),
        "retrigger_step": np.nan if retrigger_step is None else int(retrigger_step),
        "resolved": retrigger_step is None,
        "ground_seconds": ground_seconds,
        "learn_seconds": learn_seconds,
    }
