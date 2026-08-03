"""Oracle-path cluster-and-magnitude chooser for the simulated user.

Replaces min-mean-violation cluster choice in the episode loop: candidates are
every cluster mean at every magnitude, filtered by the hidden pain threshold,
then scored by how much of the oracle path would remain after taking them.
The chooser never sees language, so it is identical across verbalizer levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users.base import (
    HiddenCostTerm,
    SimulatedUser,
    compute_violations,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory


@dataclass(frozen=True)
class ChoiceResult:
    """Outcome of one cluster-and-magnitude choice."""

    label: int
    magnitude: float
    acceptable: dict[int, bool]
    scores: dict[int, float]
    no_acceptable_cluster: bool


def choose_correction(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
    oracle_path: np.ndarray,
    min_join: int = 0,
    threshold: float = 0.02,
    magnitudes: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
) -> ChoiceResult:
    """Pick the acceptable candidate closest to finishing the oracle path.

    Candidates are scaled in the cluster means' native representation (the
    same scaling the UQ planner applies) and canonicalized to 7-DOF q before
    scoring against the q-space oracle path: score is
    ``min_j(||end - oracle[j]|| + remaining_arc(j))`` over ``j >= min_join``.
    When no candidate stays under the pain ``threshold``, the lowest
    mean-violation candidate is returned with ``no_acceptable_cluster=True``.
    """
    oracle_q = canonical_arm_q(oracle_path, context).reshape(-1, 7)
    tail = oracle_q[min_join:]
    step_lengths = np.linalg.norm(np.diff(tail, axis=0), axis=-1)
    remaining_arc = np.concatenate([np.cumsum(step_lengths[::-1])[::-1], [0.0]])

    best_score = np.inf
    best: tuple[int, float] | None = None
    fallback_violation = np.inf
    fallback: tuple[int, float] | None = None
    acceptable = {label: False for label in cluster_means}
    scores: dict[int, float] = {}

    for label in sorted(cluster_means):
        for magnitude in magnitudes:
            candidate = scale_trajectory(cluster_means[label], magnitude)
            end_q = canonical_arm_q(candidate, context).reshape(-1, 7)[-1]
            score = float(np.min(np.linalg.norm(tail - end_q, axis=-1) + remaining_arc))
            scores[label] = min(score, scores.get(label, np.inf))
            violations = compute_violations(user, context, candidate)
            mean_violation = float(np.mean(violations))
            if mean_violation < fallback_violation:
                fallback_violation = mean_violation
                fallback = (label, magnitude)
            if float(np.max(violations)) <= threshold:
                acceptable[label] = True
                if score < best_score:
                    best_score = score
                    best = (label, magnitude)

    if best is not None:
        label, magnitude = best
        return ChoiceResult(label, magnitude, acceptable, scores, False)
    assert fallback is not None
    label, magnitude = fallback
    return ChoiceResult(label, magnitude, acceptable, scores, True)


def oracle_cluster_scores(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
    scale: float,
) -> dict[int, float]:
    """Hidden-cost score for each cluster mean at the given magnitude."""
    oracle_cost = HiddenCostTerm(user=user, context=context)
    return {
        label: float(
            oracle_cost(
                np.expand_dims(
                    scale_trajectory(np.asarray(traj, dtype=np.float64), scale),
                    axis=0,
                )
            )[0]
        )
        for label, traj in cluster_means.items()
    }
