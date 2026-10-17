"""The learned preference as structured bounds, which steering compiles to torch.

Cost generation states each preference twice: as Python cost code the planner
runs, and as bound terms over the anatomical joint features (stage two's
grounding of a round; the ``terms`` a combination writes next to its code). The
terms become the same :class:`HiddenBound` / :class:`CoupledBound` structures
the personas' hidden limits use, so ``SteeringConfig.bounds`` compiles them into
a differentiable steering cost without further translation.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from uncertain_feedback.evaluation_mechanism import EvalState
from uncertain_feedback.simulated_users.base import (
    Bound,
    CoupledBound,
    HiddenBound,
    feature_series,
)

# A chosen correction may sit this far (radians) past a learned bound and still
# count as inside it.
_INSIDE_TOLERANCE = 1e-6


def grounding_terms(grounding: str) -> list[dict[str, Any]]:
    """The ``terms`` of a round's stage-two grounding JSON; empty when absent."""
    if not grounding:
        return []
    terms = json.loads(grounding).get("terms")
    return terms if isinstance(terms, list) else []


def _term_bounds(term: dict[str, Any]) -> list[Bound]:
    feature = term["feature"]
    bound_type = term["bound_type"]
    values = term["values"]
    if bound_type == "band":
        return [
            HiddenBound(feature, "lower_bound", low=float(values["low"])),
            HiddenBound(feature, "upper_bound", high=float(values["high"])),
        ]
    if "points" in values:
        # CoupledBound is linear: the line through the anchors, extended past
        # them where the generated cost holds the end anchors' thresholds.
        points = np.asarray(values["points"], dtype=np.float64)
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        return [
            CoupledBound(
                feature,
                bound_type,
                values["cond_feature"],
                float(intercept),
                float(slope),
            )
        ]
    threshold = float(values["threshold"])
    if bound_type == "lower_bound":
        return [HiddenBound(feature, "lower_bound", low=threshold)]
    return [HiddenBound(feature, "upper_bound", high=threshold)]


def bounds_from_terms(terms: Sequence[dict[str, Any]]) -> tuple[Bound, ...]:
    """Convert stage-two style terms to bounds, skipping malformed ones."""
    bounds: list[Bound] = []
    for term in terms:
        try:
            bounds.extend(_term_bounds(term))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[evaluation] learned bound term skipped ({exc!r}): {term}")
    return tuple(bounds)


def _peak(bound: Bound, state: EvalState, trajectory: np.ndarray) -> float:
    return float(np.max(bound.violation(feature_series(state.human, trajectory))))


def admissible_bounds(
    bounds: Sequence[Bound], states: Sequence[EvalState]
) -> tuple[Bound, ...]:
    """The bounds the revealed preferences support.

    Each round's chosen correction must end inside every kept bound (stage two
    requires that of each bound it grounds), and some round's interrupted plan
    or rejected candidate must cross it, or it expresses no revealed preference.
    """
    kept = []
    for bound in bounds:
        inside = all(
            _peak(bound, state, state.correction_traj[-1:]) <= _INSIDE_TOLERANCE
            for state in states
        )
        crossed = any(
            _peak(bound, state, trajectory) > _INSIDE_TOLERANCE
            for state in states
            for trajectory in (
                *(() if state.reference_traj is None else (state.reference_traj,)),
                *state.rejected_trajs,
            )
        )
        if inside and crossed:
            kept.append(bound)
        else:
            print(
                f"[evaluation] learned bound {bound} dropped from steering: "
                f"{'chosen correction ends outside it' if not inside else 'nothing revealed crosses it'}"
            )
    return tuple(kept)
