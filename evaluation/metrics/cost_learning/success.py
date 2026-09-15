"""Aggregate interactions: success within k feedback rounds."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from evaluation.benchmarks.structs import Interaction
from uncertain_feedback.simulated_users import violation_metrics


def goal_row(interaction: Interaction) -> dict[str, Any]:
    """One flat record per goal; the unit success-at-k and rounds aggregate over."""
    task = interaction.task
    executed_metrics = violation_metrics(
        interaction.user, interaction.context, interaction.executed
    )
    return {
        "persona": task.persona,
        "verbalizer": task.verbalizer,
        "seed": task.seed,
        "approach": interaction.approach,
        "goal_index": interaction.goal_index,
        "result": interaction.result,
        "resolved": interaction.resolved,
        "reached": interaction.reached,
        "rounds_used": interaction.rounds_used,
        "executed_mean_violation": float(executed_metrics["mean_violation"]),
        "executed_max_violation": float(executed_metrics["max_violation"]),
    }


def goal_table(interactions: Sequence[Interaction]) -> pd.DataFrame:
    """One row per interaction, via :func:`goal_row`."""
    return pd.DataFrame([goal_row(item) for item in interactions])


def success_at_k(
    interactions: Sequence[Interaction],
    max_k: int,
    by: Sequence[str] = ("approach",),
) -> pd.DataFrame:
    """Fraction of goals resolved within ``k`` feedback rounds, for k in 0..max_k.

    A goal that hit the round cap or stalled short of the goal never counts at
    any ``k``, so the curve needs no failure value. ``k = 0`` is the goal
    completed with no feedback at all.
    """
    goals = goal_table(interactions)
    keys = [goals[column] for column in by]
    frames = []
    for k in range(max_k + 1):
        hit = goals["resolved"] & (goals["rounds_used"] <= k)
        frame = hit.groupby(keys).mean().rename("success").reset_index()
        frame["k"] = k
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

