"""How a generated correction compares to the oracle correction it should match.

Both corrections are read in the five-dimensional anatomical feature space (all
radians, so the space needs no weighting), relative to their own first frame:
anchoring on frame 0 rather than comparing poses makes the metrics indifferent
to the small start offset between an oracle window joined onto the oracle path
and a generated correction re-anchored on the live pose.

One projection answers both questions. The generated correction's endpoint is
located against the oracle's feature-space polyline — the point on that
polyline nearest it — and then:

``arc_progress``
    the arclength fraction of that nearest point. ``1`` is the oracle's own
    end, ``0`` its start. Arclength rather than projection onto the chord
    because the oracle is an MPC replan under a hidden bound and its path
    through feature space curves, so a correction sitting on the curve should
    score the fraction of the *path* it covers. Unclamped at both ends:
    stopping short of the start reads negative, carrying past the end reads
    above ``1``.

``alignment``
    the cosine between the two movement directions there: the direction the
    generated correction is travelling as it arrives — its last segment — and
    the direction the oracle is moving at that nearest point, the tangent of
    the segment owning it. Both sides are local, so a correction that traces
    the oracle scores ``1`` wherever it stopped, and one that reached the same
    place by a different route does not.

Neither needs the two trajectories on a common clock, which they are not: MDM
frames and MPC steps run at different rates and MDM output is systematically
slower, so frame-wise timing is a pipeline artifact carrying no signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_feature_series,
)
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext

# Below this the oracle asked for nothing on that axis and a fraction of it is
# noise; matches the verbalizers' meaningful-change dead-band.
FEATURE_DEAD_BAND = 0.15
_MIN_MOTION = 1e-6


@dataclass(frozen=True)
class ProgressResult:
    """A generated correction scored against the oracle it should reproduce."""

    arc_progress: float
    alignment: float
    oracle_path_length: float
    oracle_displacement: float
    per_feature: dict[str, float]


def feature_path(trajectory: np.ndarray, context: MpcCostContext) -> np.ndarray:
    """``(T, 5)`` anatomical feature path in radians."""
    series = arm_feature_series(trajectory, context)
    return np.stack([series[name] for name in FEATURE_NAMES], axis=-1)


def _final_direction(path: np.ndarray) -> np.ndarray:
    """Unit direction a path is travelling as it ends; zero if it never moves.

    Zero-length segments are dropped so a dwell frame at the end cannot decide
    the heading, matching how :func:`_arc_projection` treats a stalled frame.
    """
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=-1)
    moving = lengths > _MIN_MOTION
    if not moving.any():
        return np.zeros(path.shape[-1], dtype=np.float64)
    return segments[moving][-1] / lengths[moving][-1]


def _arc_projection(
    path: np.ndarray, point: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Locate ``point`` against a polyline: ``(arclength, local direction, length)``.

    Zero-length segments are dropped so a stalled frame cannot own the
    projection, then the nearest point is taken over the remaining segments.
    The parameter is extended past the first and last segments so a correction
    that undershoots the polyline's start or overshoots its end keeps a signed
    arclength instead of saturating at either end. The returned direction is
    the unit tangent of the segment owning the nearest point.
    """
    starts = path[:-1]
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=-1)
    entry_arc = np.concatenate([[0.0], np.cumsum(lengths)])[:-1]

    moving = lengths > _MIN_MOTION
    starts, segments = starts[moving], segments[moving]
    lengths, entry_arc = lengths[moving], entry_arc[moving]

    parameters = np.einsum("ij,ij->i", point - starts, segments) / lengths**2
    clamped = np.clip(parameters, 0.0, 1.0)
    index = int(
        np.argmin(np.linalg.norm(starts + clamped[:, None] * segments - point, axis=-1))
    )
    parameter = clamped[index]
    if index == 0:
        parameter = min(parameters[0], parameter)
    if index == len(lengths) - 1:
        parameter = max(parameters[-1], parameter)

    arc = float(entry_arc[index] + parameter * lengths[index])
    return arc, segments[index] / lengths[index], float(np.sum(lengths))


def correction_progress(
    oracle_correction: np.ndarray,
    generated_correction: np.ndarray,
    context: MpcCostContext,
) -> ProgressResult:
    """Score a generated correction against the oracle correction.

    Both trajectories may be canonical q ``(T, 7)`` or FK-boundary arm
    axis-angles ``(T, 3, 3)``. ``arc_progress`` says how far along the oracle's
    feature-space path the correction ended up; ``alignment`` whether it is
    moving the way the oracle moves at that point. ``oracle_path_length`` over
    ``oracle_displacement`` is how much the oracle's path curves: at ``1`` the
    arclength and chord formulations agree. ``per_feature`` is the signed
    fraction of the oracle's net change achieved per anatomical feature,
    ``nan`` where the oracle barely moved it.
    """
    oracle = feature_path(oracle_correction, context)
    generated = feature_path(generated_correction, context)
    oracle_relative = oracle - oracle[0]
    chord = oracle_relative[-1]
    endpoint = generated[-1] - generated[0]
    chord_norm = float(np.linalg.norm(chord))

    if len(oracle) < 2 or float(np.abs(oracle_relative).max()) < _MIN_MOTION:
        return ProgressResult(
            arc_progress=float("nan"),
            alignment=float("nan"),
            oracle_path_length=0.0,
            oracle_displacement=chord_norm,
            per_feature={name: float("nan") for name in FEATURE_NAMES},
        )

    arc, direction, path_length = _arc_projection(oracle_relative, endpoint)
    return ProgressResult(
        arc_progress=arc / path_length,
        alignment=float(_final_direction(generated) @ direction),
        oracle_path_length=path_length,
        oracle_displacement=chord_norm,
        per_feature={
            name: (
                float(endpoint[index] / chord[index])
                if abs(chord[index]) > FEATURE_DEAD_BAND
                else float("nan")
            )
            for index, name in enumerate(FEATURE_NAMES)
        },
    )
