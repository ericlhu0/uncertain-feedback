"""How far an executed motion goes over the persona's hidden limits.

A speed persona is judged by its speed limits, any other by its path bounds.
Speed only exists between frames on one clock, so the motion is passed as
segments, each on a single clock; no step is taken across a segment seam.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import STEP_SECONDS
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import (
    SimulatedUser,
    feature_series,
    speed_margins,
)


def step_excess(
    user: SimulatedUser, human: Human, segment: np.ndarray, dt: float = STEP_SECONDS
) -> np.ndarray:
    """Excess over the hidden limits at each step of a ``(T, 7)`` segment, ``(T - 1,)``.

    The step's speed over its cap (m/s or rad/s) for a speed persona, else the
    path-bound violation (rad) of the frame the step ends on.
    """
    if user.speed_limits:
        return np.maximum(speed_margins(user, human, segment, dt).max(axis=0), 0.0)
    return user.violation_series(feature_series(human, segment))[1:]


def limit_violation(
    user: SimulatedUser,
    human: Human,
    segments: Sequence[np.ndarray],
    dt: float = STEP_SECONDS,
) -> dict[str, float]:
    """Share of executed steps over a hidden limit and the peak excess.

    ``segments`` are ``(T, 7)`` stretches of the executed motion, ``dt`` seconds
    between frames. ``peak_excess`` is in the limits' own units (rad, m/s or
    rad/s). Both are NaN when the segments hold no step.
    """
    excess = np.concatenate(
        [step_excess(user, human, segment, dt) for segment in segments]
    )
    if excess.size == 0:
        return {"over_limit_share": float("nan"), "peak_excess": float("nan")}
    return {
        "over_limit_share": float(np.mean(excess > 0.0)),
        "peak_excess": float(excess.max()),
    }
