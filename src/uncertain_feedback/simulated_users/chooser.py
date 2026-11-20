"""Cluster-and-magnitude chooser for the simulated user.

Modes model different users picking among candidate corrections:

- ``oracle_progress``: among candidates that never violate a hidden
  bound at any frame (zero tolerance, not the pain threshold), the one whose
  endpoint lands closest to the oracle correction's own end along its
  feature-space path (:func:`~.progress.correction_progress`; ``arc_progress``
  nearest 1), ties broken by ``alignment``. ``oracle_path`` must be the oracle
  *correction* the candidates are meant to reproduce, from ``min_join`` on.
- ``intent_aligned``: the comfortable candidate whose motion best aligns with
  the user's private correction intent (the same :class:`CorrectionIntent` the
  verbalizers phrase) — a person who knows what they want even when their
  words were vague, with no knowledge of the oracle path geometry.
- ``progress``: the legacy oracle-path chooser — comfortable candidates scored
  by how much of the internal oracle path would remain after taking them.
  Superhuman path knowledge; kept for ablation.
- ``random``: uniform choice among comfortable candidates.
- ``robot_top``: no choice by the user; the robot runs its own top pick as
  proposed, the first candidate in the grounder's order (the largest cluster
  for MDM) at magnitude 1, comfortable or not. ``acceptable`` is still judged
  as ``oracle_progress`` would, so the metrics read the same.
- ``cost_to_go`` (the planner config's default): among the candidates ``oracle_progress`` accepts, those
  whose largest motion is what the user's own sentence names, going the way
  it says (failing any, those that merely move the named way; failing any,
  all — :func:`_says_so`), the one after which the goal is fewest planner
  steps away under the user's own limits (``steps_to_goal``; ``inf`` when
  unreached), ties broken by the ``oracle_progress`` key. A person who knows
  where they are going and what hurts, but not the oracle's route. Acceptability, and so the rejected
  candidates a learner sees, is identical to ``oracle_progress``.

In every mode a comfortable candidate also never moves the user over a speed
limit long enough to notice, as it would play (:func:`~.base.first_complaint`).
Only ``cost_to_go`` reads the sentence; ``intent_aligned`` sees the intent
behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    arm_aa_from_state,
    canonical_arm_q,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users.attribution import (
    ATTRIBUTED_FEATURES,
    CorrectionIntent,
)
from uncertain_feedback.simulated_users.base import (
    HiddenCostTerm,
    SimulatedUser,
    compute_violations,
    feature_series,
    first_complaint,
    speed_margins,
)
from uncertain_feedback.simulated_users.progress import correction_progress
from uncertain_feedback.simulated_users.verbalizers import (
    JOINT_PHRASES,
    Utterance,
    named_axes,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

CHOOSER_MODES = (
    "oracle_progress",
    "intent_aligned",
    "progress",
    "random",
    "robot_top",
    "cost_to_go",
)

# Offsets are meters, features radians; 5 cm of offset weighs as much as 0.15 rad.
_OFFSET_WEIGHT = 0.15 / 0.05

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
    """Concatenated desired change (features then offsets)."""
    parts = [-intent.feature_deltas[name] for name in ATTRIBUTED_FEATURES]
    for offset in (intent.wrist_offset, intent.elbow_offset):
        parts.extend(-np.asarray(offset, dtype=np.float64) * _OFFSET_WEIGHT)
    return np.asarray(parts, dtype=np.float64)


def _candidate_change(human: Human, candidate: np.ndarray) -> np.ndarray:
    """Candidate motion in the same coordinates as :func:`_desired_vector`."""
    features = feature_series(human, candidate)
    parts = [
        float(np.mean(features[name][1:]) - features[name][0])
        for name in ATTRIBUTED_FEATURES
    ]
    arm_aa = arm_aa_from_state(candidate, human).reshape(-1, 3, 3)
    positions = human.fk.fk_batch(arm_aa, human.spine3_pos, human.spine3_aa)
    for chain in (_WRIST_CHAIN_IDX, _ELBOW_CHAIN_IDX):
        change = positions[1:, chain].mean(axis=0) - positions[0, chain]
        parts.extend(change * _OFFSET_WEIGHT)
    return np.asarray(parts, dtype=np.float64)


def _stated(
    intent: CorrectionIntent, utterance: Utterance
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`_desired_vector`, the components the sentence names, and the
    group they are compared within.

    A joint phrase names its feature among the joint features; an arm or elbow
    phrase names its referent's direction-word axes among that referent's
    three. Free-form and vague sentences name nothing.
    """
    desired = _desired_vector(intent)
    named = np.zeros(desired.shape, dtype=bool)
    n_features = len(ATTRIBUTED_FEATURES)
    group = np.arange(0)
    if utterance.form == "joint_resolved":
        group = np.arange(n_features)
        for index, name in enumerate(ATTRIBUTED_FEATURES):
            sign = 1 if intent.feature_deltas[name] > 0 else -1
            named[index] = JOINT_PHRASES[(name, sign)] in utterance.text
    elif utterance.form in ("arm", "elbow", "motion_directive"):
        elbow = "elbow" in utterance.text
        offset = intent.elbow_offset if elbow else intent.wrist_offset
        start = n_features + (3 if elbow else 0)
        group = np.arange(start, start + 3)
        for axis in named_axes(offset):
            named[start + axis] = True
    return desired, named, group


def _says_so(
    change: np.ndarray, stated: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[bool, bool]:
    """Whether ``change`` does what the sentence says: (every named component
    goes the right way and the largest motion in their group is a named one,
    the named components move the right way overall). With nothing named,
    both are agreement with the whole intent."""
    desired, named, group = stated
    if not named.any():
        agrees = float(np.dot(change, desired)) > 0.0
        return agrees, agrees
    signs = bool(np.all(change[named] * desired[named] > 0.0))
    largest = bool(named[group[np.argmax(np.abs(change[group]))]])
    return signs and largest, float(np.dot(change[named], desired[named])) > 0.0


def choose_correction(
    user: SimulatedUser,
    human: Human,
    cluster_means: dict[int, np.ndarray],
    oracle_path: np.ndarray,
    min_join: int = 0,
    threshold: float = 0.02,
    magnitudes: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    *,
    mode: str = "oracle_progress",
    intent: CorrectionIntent | None = None,
    rng: np.random.Generator | None = None,
    history: np.ndarray | None = None,
    steps_to_goal: Callable[[np.ndarray], float] | None = None,
    utterance: Utterance | None = None,
) -> ChoiceResult:
    """Pick a (cluster, magnitude) the user would accept, per ``mode``.

    A candidate is rejected when the user would complain while it plays, by
    the execution-time trigger (:func:`first_complaint`): past the pain
    ``threshold`` at any frame (``oracle_progress`` rejects any violation at
    all), or moved over a speed limit long enough to notice, at the candidate's
    size and with ``history``, the motion executed up to its first frame,
    filling the window. When none survives, the candidate least past the path
    limits, then least over a speed limit at its fastest step, is returned with
    ``no_acceptable_cluster=True``. ``robot_top`` returns the first candidate
    at magnitude 1 regardless, flagging the same.
    """
    if mode not in CHOOSER_MODES:
        raise ValueError(f"Unknown chooser mode {mode!r}; expected {CHOOSER_MODES}.")
    if mode == "intent_aligned" and intent is None:
        raise ValueError("intent_aligned mode requires the correction intent.")
    if mode == "cost_to_go" and (
        steps_to_goal is None or intent is None or utterance is None
    ):
        raise ValueError("cost_to_go mode requires steps_to_goal, intent, utterance.")

    desired = _desired_vector(intent) if intent is not None else None
    desired_norm = float(np.linalg.norm(desired)) if desired is not None else 0.0
    stated = (
        _stated(intent, utterance)
        if intent is not None and utterance is not None
        else None
    )

    oracle_q = canonical_arm_q(oracle_path, human).reshape(-1, 7)
    tail = oracle_q[min_join:]
    limit = 0.0 if mode in ("oracle_progress", "robot_top", "cost_to_go") else threshold
    step_lengths = np.linalg.norm(np.diff(tail, axis=0), axis=-1)
    remaining_arc = np.concatenate([np.cumsum(step_lengths[::-1])[::-1], [0.0]])

    acceptable = {label: False for label in cluster_means}
    scores: dict[int, float] = {}
    alignment: dict[int, float] = {}
    acceptable_pairs: list[tuple[int, float]] = []
    pair_keys: dict[tuple[int, float], tuple[float, ...]] = {}
    # cost_to_go: (_says_so strictly, loosely, candidate) per pair.
    stated_pairs: dict[tuple[int, float], tuple[bool, bool, np.ndarray]] = {}
    fallback_violation = (np.inf, np.inf)
    fallback: tuple[int, float] | None = None

    for label in sorted(cluster_means):
        if desired is not None:
            change = _candidate_change(human, cluster_means[label])
            change_norm = float(np.linalg.norm(change))
            cosine = (
                float(np.dot(change, desired)) / (change_norm * desired_norm)
                if change_norm > 1e-9 and desired_norm > 1e-9
                else 0.0
            )
            alignment[label] = cosine
        for magnitude in magnitudes:
            candidate = scale_trajectory(cluster_means[label], magnitude)
            end_q = canonical_arm_q(candidate, human).reshape(-1, 7)[-1]
            progress = float(
                np.min(np.linalg.norm(tail - end_q, axis=-1) + remaining_arc)
            )
            scores[label] = min(progress, scores.get(label, np.inf))
            peak_violation = (
                float(np.max(compute_violations(user, human, candidate))),
                (
                    float(np.max(speed_margins(user, human, candidate), initial=0.0))
                    if user.speed_limits
                    else 0.0
                ),
            )
            if peak_violation < fallback_violation:
                fallback_violation = peak_violation
                fallback = (label, magnitude)
            if first_complaint(user, human, candidate, limit, history) is not None:
                continue
            acceptable[label] = True
            acceptable_pairs.append((label, magnitude))
            if mode in ("oracle_progress", "cost_to_go"):
                result = correction_progress(tail, candidate, human)
                pair_keys[(label, magnitude)] = (
                    -abs(result.arc_progress - 1.0),
                    result.alignment,
                )
                if stated is not None and mode == "cost_to_go":
                    stated_pairs[(label, magnitude)] = (
                        *_says_so(_candidate_change(human, candidate), stated),
                        candidate,
                    )
            elif mode == "progress":
                pair_keys[(label, magnitude)] = (-progress, 0.0)
            elif mode == "intent_aligned":
                assert desired is not None
                projection = float(
                    np.dot(_candidate_change(human, candidate), desired)
                ) / max(desired_norm, 1e-9)
                pair_keys[(label, magnitude)] = (
                    alignment[label],
                    -abs(projection - desired_norm),
                )

    if mode == "robot_top":
        return ChoiceResult(
            next(iter(cluster_means)),
            1.0,
            acceptable,
            scores,
            not acceptable_pairs,
            alignment,
        )
    if not acceptable_pairs:
        assert fallback is not None
        label, magnitude = fallback
        return ChoiceResult(label, magnitude, acceptable, scores, True, alignment)

    if mode == "random":
        assert rng is not None, "random mode requires an rng."
        label, magnitude = acceptable_pairs[int(rng.integers(len(acceptable_pairs)))]
    elif mode == "cost_to_go":
        assert steps_to_goal is not None
        pool = (
            [pair for pair in acceptable_pairs if stated_pairs[pair][0]]
            or [pair for pair in acceptable_pairs if stated_pairs[pair][1]]
            or acceptable_pairs
        )
        label, magnitude = max(
            pool,
            key=lambda pair: (-steps_to_goal(stated_pairs[pair][2]), *pair_keys[pair]),
        )
    else:
        label, magnitude = max(acceptable_pairs, key=lambda pair: pair_keys[pair])
    return ChoiceResult(label, magnitude, acceptable, scores, False, alignment)


def oracle_cluster_scores(
    user: SimulatedUser,
    human: Human,
    cluster_means: dict[int, np.ndarray],
    scale: float,
) -> dict[int, float]:
    """Hidden-cost score for each cluster mean at the given magnitude."""
    oracle_cost = HiddenCostTerm(user=user, human=human)
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
