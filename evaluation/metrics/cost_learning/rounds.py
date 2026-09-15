"""Per-round grounding quality inside the loop: one flat record per feedback round."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.benchmarks.structs import Interaction
from uncertain_feedback.simulated_users import oracle_cluster_scores, violation_metrics


def round_rows(interaction: Interaction) -> list[dict[str, Any]]:
    """One flat record per feedback round; the unit the grounding analysis aggregates."""
    task = interaction.task
    user = interaction.user
    context = interaction.context
    rows: list[dict[str, Any]] = []
    for rnd in interaction.rounds:
        grounding = rnd.grounding
        choice = rnd.choice
        hidden_scores = oracle_cluster_scores(
            user, context, grounding.candidates, grounding.magnitude
        )
        continuation_metrics = violation_metrics(user, context, rnd.continuation)
        n_acceptable = sum(1 for ok in choice.acceptable.values() if ok)
        rows.append(
            {
                "persona": task.persona,
                "verbalizer": task.verbalizer,
                "seed": task.seed,
                "approach": interaction.approach,
                "goal_index": interaction.goal_index,
                "round_index": rnd.round_index,
                "event_index": rnd.event_index,
                "utterance_form": rnd.utterance.form,
                "utterance_text": rnd.utterance.text,
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
                    float(max(choice.alignment.values()))
                    if choice.alignment
                    else np.nan
                ),
                "candidate_hidden_mean": float(np.mean(list(hidden_scores.values()))),
                "candidate_hidden_min": float(np.min(list(hidden_scores.values()))),
                "chosen_hidden": float(hidden_scores[grounding.chosen_label]),
                "cost_accepted": rnd.outcome.cost_accepted,
                "unified_installed": rnd.outcome.unified_installed,
                "continuation_mean_violation": float(
                    continuation_metrics["mean_violation"]
                ),
                "continuation_max_violation": float(
                    continuation_metrics["max_violation"]
                ),
                "continuation_frac_violated": float(
                    continuation_metrics["frac_frames_violated"]
                ),
                "retrigger_step": (
                    np.nan if rnd.retrigger_step is None else int(rnd.retrigger_step)
                ),
                "resolved": rnd.retrigger_step is None,
                "ground_seconds": rnd.ground_seconds,
                "learn_seconds": rnd.learn_seconds,
            }
        )
    return rows

