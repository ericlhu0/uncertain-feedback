"""How varied the candidate corrections proposed for one utterance are.

Every candidate is read in the five-dimensional anatomical feature space (all
radians, so the space needs no weighting) relative to its own first frame, and
resampled to ``N_WAYPOINTS`` points equidistant in arclength. Resampling
rather than comparing frames removes timing: candidates from one generator
share a clock but not a speed profile, and generators differ in both clip
length and how long they dwell, so frame-wise sums would weight endpoint
spread against route spread by each method's pacing and credit stochastic
timing as diversity. After resampling two candidates that trace one path at
different speeds are the same vector, and every method is weighted uniformly
along its route.

:func:`candidate_position_diversity` is the same measure in the space the
person watching the menu perceives — elbow and wrist positions in metres — since
candidates that fan out in hand position can share joint angles and vice versa.

``diversity`` is the spread of those anchored paths around their mean over
their pooled magnitude::

    sqrt( mean_i ||p_i - mean(p)||^2 / mean_i ||p_i||^2 )

which equals ``sqrt(1 - ||mean(p)||^2 / mean(||p||^2))`` and so lives in
``[0, 1]``: ``0`` when every candidate proposes the same correction, ``1``
when they cancel out. Dividing by magnitude rather than within-cluster
scatter is what makes a set of near-identical large moves read as low
diversity, and lets a large candidate weigh more than a small one pointing
the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.metrics.grounding.progress import feature_path
from uncertain_feedback.planners.mpc.arm_features import (
    arm_aa_from_state,
    resample_equidistant,
)
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import ELBOW_CHAIN_IDX, WRIST_CHAIN_IDX

N_WAYPOINTS = 15
_MIN_MOTION = 1e-6


@dataclass(frozen=True)
class DiversityResult:
    """Spread of one utterance's candidates, normalized by how far they move."""

    diversity: float
    rms_displacement: float


def _normalized_dispersion(paths: np.ndarray) -> DiversityResult:
    """``paths`` is ``(N, n, D)`` anchored, arclength-resampled feature paths."""
    magnitude = float(np.mean(np.sum(paths**2, axis=-1)))
    if paths.shape[0] < 2 or magnitude < _MIN_MOTION**2:
        return DiversityResult(
            diversity=float("nan"), rms_displacement=float(np.sqrt(magnitude))
        )
    spread = float(np.mean(np.sum((paths - paths.mean(axis=0)) ** 2, axis=-1)))
    return DiversityResult(
        diversity=float(np.sqrt(spread / magnitude)),
        rms_displacement=float(np.sqrt(magnitude)),
    )


def candidate_diversity(
    candidates: dict[int, np.ndarray], context: MpcCostContext
) -> DiversityResult:
    """Score how varied a grounder's candidate corrections are.

    Candidates may be canonical q ``(T, 7)`` or FK-boundary arm axis-angles
    ``(T, 3, 3)`` and need not share a length. ``rms_displacement`` is the
    normalizer: the root-mean-square feature-space displacement from the start
    pose over every waypoint of every candidate, in radians. ``diversity`` is
    ``nan`` with fewer than two candidates or when none of them move.
    """
    paths = np.stack(
        [
            resample_equidistant(path - path[0], N_WAYPOINTS)
            for path in (
                feature_path(c, context) for _, c in sorted(candidates.items())
            )
        ]
    )
    return _normalized_dispersion(paths)


def _position_path(trajectory: np.ndarray, context: MpcCostContext) -> np.ndarray:
    """``(T, 6)`` elbow and wrist world positions in metres."""
    positions = context.fk.fk_batch(
        arm_aa_from_state(trajectory, context).reshape(-1, 3, 3),
        context.spine3_pos,
        context.spine3_aa,
    )
    return positions[:, [ELBOW_CHAIN_IDX, WRIST_CHAIN_IDX]].reshape(len(positions), 6)


def candidate_position_diversity(
    candidates: dict[int, np.ndarray], context: MpcCostContext
) -> DiversityResult:
    """:func:`candidate_diversity` over elbow and wrist positions instead of features.

    ``rms_displacement`` is then in metres.
    """
    paths = np.stack(
        [
            resample_equidistant(path - path[0], N_WAYPOINTS)
            for path in (
                _position_path(c, context) for _, c in sorted(candidates.items())
            )
        ]
    )
    return _normalized_dispersion(paths)
