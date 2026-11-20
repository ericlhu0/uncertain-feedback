"""Aggregate interactions: success within k user turns (feedback rounds + approvals)."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from evaluation.benchmarks.structs import Interaction
from evaluation.metrics.speed import felt_segments, limit_violation
from uncertain_feedback.simulated_users import violation_metrics


def goal_row(interaction: Interaction) -> dict[str, Any]:
    """One flat record per goal; the unit success-at-k and rounds aggregate over.

    ``felt_*`` read every motion the persona went through, rewound steps
    included (:func:`felt_segments`), against its hidden limits: speed limits
    for a speed persona, path bounds otherwise (:func:`limit_violation`).
    ``duration_ratio`` is the executed motion's length against the oracle's.
    """
    task = interaction.task
    executed_metrics = violation_metrics(
        interaction.user, interaction.human, interaction.executed
    )
    felt = limit_violation(
        interaction.user, interaction.human, felt_segments(interaction)
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
        "approvals": interaction.approvals,
        "user_turns": interaction.user_turns,
        "proposal_approved": interaction.proposal_approved,
        "violation_count": interaction.violation_count,
        "speed_complaints": interaction.speed_complaints,
        "executed_mean_violation": float(executed_metrics["mean_violation"]),
        "executed_max_violation": float(executed_metrics["max_violation"]),
        "felt_over_limit_share": felt["over_limit_share"],
        "felt_peak_excess": felt["peak_excess"],
        "motion_steps": interaction.motion_steps,
        "duration_ratio": interaction.motion_steps
        / max(len(interaction.oracle_path) - 1, 1),
        "robot_seconds": interaction.robot_seconds,
    }


def goal_table(interactions: Sequence[Interaction]) -> pd.DataFrame:
    """One row per interaction, via :func:`goal_row`."""
    return pd.DataFrame([goal_row(item) for item in interactions])


def success_at_k(
    interactions: Sequence[Interaction],
    max_k: int,
    by: Sequence[str] = ("approach",),
) -> pd.DataFrame:
    """Fraction of goals resolved within ``k`` user turns, for k in 0..max_k.

    A user turn is a feedback round or an approved stall proposal
    (:attr:`Interaction.user_turns`), so a goal that needed the persona's
    approval to get there costs that answer too. A goal that hit the round cap
    or stalled short of the goal never counts at any ``k``, so the curve needs
    no failure value. ``k = 0`` is the goal completed with no feedback at all;
    filter on ``goal_table``'s ``proposal_approved`` to see how much of the
    curve relied on an approval.
    """
    goals = goal_table(interactions)
    keys = [goals[column] for column in by]
    frames = []
    for k in range(max_k + 1):
        hit = goals["resolved"] & (goals["user_turns"] <= k)
        frame = hit.groupby(keys).mean().rename("success").reset_index()
        frame["k"] = k
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
