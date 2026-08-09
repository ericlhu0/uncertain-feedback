"""Cluster-and-magnitude chooser for the simulated user.

Modes model different users picking among candidate corrections:

- ``intent_aligned`` (default): the comfortable candidate whose motion best
  aligns with the user's private correction intent (the same
  :class:`CorrectionIntent` the verbalizers phrase) — a person who knows what
  they want even when their words were vague, with no knowledge of the oracle
  path geometry.
- ``progress``: the legacy oracle-path chooser — comfortable candidates scored
  by how much of the internal oracle path would remain after taking them.
  Superhuman path knowledge; kept for ablation.
- ``random``: uniform choice among comfortable candidates.

No mode sees language; ``intent_aligned`` sees the intent behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    arm_aa_from_state,
    canonical_arm_q,
)
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users.attribution import (
    ATTRIBUTED_FEATURES,
    CorrectionIntent,
)
from uncertain_feedback.simulated_users.base import (
    HiddenCostTerm,
    SimulatedUser,
    compute_violations,
    feature_series,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

CHOOSER_MODES = ("intent_aligned", "progress", "random")

_FEATURE_DEAD_BAND = 0.15
_OFFSET_DEAD_BAND = 0.05
# Offsets are meters, features radians; weight offsets so the dead-bands match.
_OFFSET_WEIGHT = _FEATURE_DEAD_BAND / _OFFSET_DEAD_BAND

_ELBOW_CHAIN_IDX = 3
_WRIST_CHAIN_IDX = 4


@dataclass(frozen=True)
class ChoiceResult:
    """Outcome of one cluster-and-magnitude choice."""

    label: int
    magnitude: float
    acceptable: dict[int, bool]
    scores: dict[int, float]
    no_acceptable_cluster: bool
    alignment: dict[int, float] = field(default_factory=dict)


def _desired_vector(intent: CorrectionIntent) -> np.ndarray:
    """Concatenated desired change (features then offsets), dead-banded."""
    parts = []
    for name in ATTRIBUTED_FEATURES:
        delta = intent.feature_deltas[name]
        parts.append(-delta if abs(delta) > _FEATURE_DEAD_BAND else 0.0)
    for offset in (intent.wrist_offset, intent.elbow_offset):
        offset = np.asarray(offset, dtype=np.float64)
        active = float(np.linalg.norm(offset)) > _OFFSET_DEAD_BAND
        parts.extend((-offset * _OFFSET_WEIGHT) if active else np.zeros(3))
    return np.asarray(parts, dtype=np.float64)


def _candidate_change(context: MpcCostContext, candidate: np.ndarray) -> np.ndarray:
    """Candidate motion in the same coordinates as :func:`_desired_vector`."""
    features = feature_series(context, candidate)
    parts = [
        float(np.mean(features[name][1:]) - features[name][0])
        for name in ATTRIBUTED_FEATURES
    ]
    arm_aa = arm_aa_from_state(candidate, context).reshape(-1, 3, 3)
    positions = context.fk.fk_batch(arm_aa, context.spine3_pos, context.spine3_aa)
    for chain in (_WRIST_CHAIN_IDX, _ELBOW_CHAIN_IDX):
        change = positions[1:, chain].mean(axis=0) - positions[0, chain]
        parts.extend(change * _OFFSET_WEIGHT)
    return np.asarray(parts, dtype=np.float64)


def choose_correction(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
    oracle_path: np.ndarray,
    min_join: int = 0,
    threshold: float = 0.02,
    magnitudes: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    *,
    mode: str = "intent_aligned",
    intent: CorrectionIntent | None = None,
    rng: np.random.Generator | None = None,
) -> ChoiceResult:
    """Pick a (cluster, magnitude) the user would accept, per ``mode``.

    All modes reject candidates whose playback exceeds the pain ``threshold``;
    when none survives, the lowest mean-violation candidate is returned with
    ``no_acceptable_cluster=True``.
    """
    if mode not in CHOOSER_MODES:
        raise ValueError(f"Unknown chooser mode {mode!r}; expected {CHOOSER_MODES}.")
    if mode == "intent_aligned" and intent is None:
        raise ValueError("intent_aligned mode requires the correction intent.")

    desired = _desired_vector(intent) if intent is not None else None
    desired_norm = float(np.linalg.norm(desired)) if desired is not None else 0.0

    oracle_q = canonical_arm_q(oracle_path, context).reshape(-1, 7)
    tail = oracle_q[min_join:]
    step_lengths = np.linalg.norm(np.diff(tail, axis=0), axis=-1)
    remaining_arc = np.concatenate([np.cumsum(step_lengths[::-1])[::-1], [0.0]])

    acceptable = {label: False for label in cluster_means}
    scores: dict[int, float] = {}
    alignment: dict[int, float] = {}
    acceptable_pairs: list[tuple[int, float]] = []
    pair_keys: dict[tuple[int, float], tuple[float, float]] = {}
    fallback_violation = np.inf
    fallback: tuple[int, float] | None = None

    for label in sorted(cluster_means):
        if desired is not None:
            change = _candidate_change(context, cluster_means[label])
            change_norm = float(np.linalg.norm(change))
            cosine = (
                float(np.dot(change, desired)) / (change_norm * desired_norm)
                if change_norm > 1e-9 and desired_norm > 1e-9
                else 0.0
            )
            alignment[label] = cosine
        for magnitude in magnitudes:
            candidate = scale_trajectory(cluster_means[label], magnitude)
            end_q = canonical_arm_q(candidate, context).reshape(-1, 7)[-1]
            progress = float(
                np.min(np.linalg.norm(tail - end_q, axis=-1) + remaining_arc)
            )
            scores[label] = min(progress, scores.get(label, np.inf))
            violations = compute_violations(user, context, candidate)
            mean_violation = float(np.mean(violations))
            if mean_violation < fallback_violation:
                fallback_violation = mean_violation
                fallback = (label, magnitude)
            # A demo is acceptable when it is not painful on average — a brief
            # graze of the limit does not make a person reject the motion. The
            # execution-time trigger (first_violation_step) stays max-based.
            if mean_violation > threshold:
                continue
            acceptable[label] = True
            acceptable_pairs.append((label, magnitude))
            if mode == "progress":
                pair_keys[(label, magnitude)] = (-progress, 0.0)
            elif mode == "intent_aligned":
                assert desired is not None
                projection = float(
                    np.dot(_candidate_change(context, candidate), desired)
                ) / max(desired_norm, 1e-9)
                pair_keys[(label, magnitude)] = (
                    alignment[label],
                    -abs(projection - desired_norm),
                )

    if not acceptable_pairs:
        assert fallback is not None
        label, magnitude = fallback
        return ChoiceResult(label, magnitude, acceptable, scores, True, alignment)

    if mode == "random":
        assert rng is not None, "random mode requires an rng."
        label, magnitude = acceptable_pairs[int(rng.integers(len(acceptable_pairs)))]
    else:
        label, magnitude = max(acceptable_pairs, key=lambda pair: pair_keys[pair])
    return ChoiceResult(label, magnitude, acceptable, scores, False, alignment)


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
