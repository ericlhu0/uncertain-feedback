"""How far an executed motion goes over the persona's hidden limits.

A speed persona is judged by its speed limits, any other by its path bounds.
Speed only exists between frames on one clock, so the motion is passed as
segments, each on a single clock; no step is taken across a segment seam. An
episode's rewind after a complaint is such a seam (:func:`felt_segments`).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from evaluation.benchmarks.structs import Interaction
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    SPEED_FEATURE_NAMES,
    STEP_SECONDS,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    compile_generated_cost,
    generated_cost_feature_dependencies,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import (
    SimulatedUser,
    SpeedLimit,
    feature_series,
    speed_margins,
)

# How far past its cap a probe step is moved (see learned_limit_row).
_PROBE_OVER = 4.0


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


def felt_segments(interaction: Interaction) -> list[np.ndarray]:
    """Every motion the persona went through on one goal, ``(T, 7)`` each.

    The arm moves until each complaint, then rewinds ``history_cut`` steps in an
    instant and goes on from there, so each stretch is its own segment: the
    motion before feedback up to the trigger, then per round the correction
    from the feedback pose and its continuation up to the next complaint.
    Unlike ``Interaction.executed`` this keeps the steps the rewinds discarded,
    which the persona felt.
    """
    first = interaction.initial_rollout
    if interaction.trigger_step is not None:
        first = first[: interaction.trigger_step + 1]
    segments = [first]
    for rnd in interaction.rounds:
        end = rnd.retrigger_step
        continuation = (
            rnd.continuation[1:] if end is None else rnd.continuation[1 : end + 1]
        )
        segments.append(
            np.concatenate([rnd.q_feedback[None], rnd.correction_q[1:], continuation])
        )
    return segments


def _over(code: str, params: dict[str, Any], limit: SpeedLimit, sign: int) -> bool:
    """Whether ``code``'s speed margins are over for one probe step.

    The step moves ``limit``'s feature at ``_PROBE_OVER`` times its cap, the way
    it caps (``sign`` 1) or the other way (-1), from the pose where the cap is
    tightest; every other feature is still.
    """
    import torch  # pylint: disable=import-outside-toplevel

    _, speed = compile_generated_cost(code)
    if speed is None:
        return False
    features = {
        name: torch.zeros(1, 1, dtype=torch.float64)
        for name in (*FEATURE_NAMES, *SPEED_FEATURE_NAMES)
    }
    if limit.cond_feature is not None:
        features[limit.cond_feature] += limit.cond_start
    features[limit.feature] += sign * limit.direction * _PROBE_OVER * limit.cap
    with torch.no_grad():
        margins = speed(features, params)
    return any(float(margin.max()) > 0.0 for margin in margins.values())


def learned_limit_row(
    code: str, params: dict[str, Any], user: SimulatedUser
) -> dict[str, Any]:
    """How a learned cost relates to a speed persona's hidden limits.

    ``speed_cap``: it states speed margins. ``right_feature``: it reads a
    feature the persona caps. ``catches``: a step well over a hidden cap, the
    way it is capped, is over its margins. ``spares_reverse``: the same step
    the other way, for a signed rate, is not (``None`` for a speed, which has
    no other way).
    """
    _, speed = compile_generated_cost(code)
    read = set(generated_cost_feature_dependencies(code))
    signed = [limit for limit in user.speed_limits if limit.feature.endswith("_rate")]
    return {
        "speed_cap": speed is not None,
        "right_feature": any(limit.feature in read for limit in user.speed_limits),
        "catches": any(_over(code, params, limit, 1) for limit in user.speed_limits),
        "spares_reverse": (
            not any(_over(code, params, limit, -1) for limit in signed)
            if signed
            else None
        ),
    }
