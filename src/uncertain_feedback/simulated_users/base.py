"""Simulated care recipients with hidden comfort costs.

A :class:`SimulatedUser` holds range-of-motion restrictions the planner does not
know about. It plays the care recipient in headless experiments: it decides when
the robot's motion becomes uncomfortable (:func:`first_violation_step`), what it
says (``feedback_text``), which UQ cluster it prefers (:func:`choose_cluster`),
and scores any trajectory by how much the hidden restrictions are violated
(:func:`violation_metrics`). The hidden cost is never shown to the cost
generator — it is the ground truth the generated cost is evaluated against.

Restrictions are expressed over the same four anatomical joint features the cost
generator uses (see ``GeneratedCostContext``), so a hidden bound and a generated
bound are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext, TrajectoryCost
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    build_joint_angle_series,
)

FEATURE_NAMES = (
    "elbow_flexion",
    "shoulder_flexion_extension",
    "shoulder_abduction_adduction",
    "shoulder_internal_external_rotation",
)

BOUND_TYPES = ("upper_bound", "lower_bound", "avoid_band")


def _feature_context(context: MpcCostContext) -> GeneratedCostContext:
    """Wrap an MPC cost context so the shared joint-feature helpers apply."""
    empty = np.empty((0, 3, 3), dtype=np.float64)
    return GeneratedCostContext(
        fk=context.fk,
        spine3_pos=np.asarray(context.spine3_pos, dtype=np.float64),
        spine3_aa=np.asarray(context.spine3_aa, dtype=np.float64),
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=empty,
        recent_q=empty,
    )


def feature_series(
    context: MpcCostContext, trajectory: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the four joint-feature series for any ``(..., 3, 3)`` trajectory."""
    return build_joint_angle_series(
        _feature_context(context), np.asarray(trajectory, dtype=np.float64)
    )


@dataclass(frozen=True)
class FeatureCondition:
    """Activate a bound only while another feature is inside ``[low, high]``."""

    feature: str
    low: float
    high: float


@dataclass(frozen=True)
class HiddenBound:
    """One hidden restriction over a joint feature, in radians.

    ``upper_bound`` penalizes exceeding ``high``; ``lower_bound`` penalizes
    falling below ``low``; ``avoid_band`` penalizes depth inside ``[low, high]``
    (a painful range the arm should pass around, not through).
    """

    feature: str
    bound_type: str
    low: float | None = None
    high: float | None = None
    condition: FeatureCondition | None = None

    def __post_init__(self) -> None:
        if self.feature not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature {self.feature!r}.")
        if self.bound_type not in BOUND_TYPES:
            raise ValueError(f"Unknown bound_type {self.bound_type!r}.")
        if self.bound_type in ("upper_bound", "avoid_band") and self.high is None:
            raise ValueError(f"{self.bound_type} requires high.")
        if self.bound_type in ("lower_bound", "avoid_band") and self.low is None:
            raise ValueError(f"{self.bound_type} requires low.")

    def violation(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Return per-frame violation magnitudes (radians), zero when satisfied."""
        values = np.asarray(features[self.feature], dtype=np.float64)
        if self.bound_type == "upper_bound":
            v = np.maximum(values - float(self.high), 0.0)  # type: ignore[arg-type]
        elif self.bound_type == "lower_bound":
            v = np.maximum(float(self.low) - values, 0.0)  # type: ignore[arg-type]
        else:
            v = np.maximum(
                np.minimum(values - float(self.low), float(self.high) - values),  # type: ignore[arg-type]
                0.0,
            )
        if self.condition is not None:
            cond = np.asarray(features[self.condition.feature], dtype=np.float64)
            active = (cond >= self.condition.low) & (cond <= self.condition.high)
            v = np.where(active, v, 0.0)
        return v


@dataclass(frozen=True)
class SimulatedUser:
    """A care recipient persona with hidden ROM restrictions."""

    name: str
    description: str
    feedback_text: str
    bounds: tuple[HiddenBound, ...]

    def violation_series(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Return summed per-frame violations across all hidden bounds."""
        total = np.zeros_like(
            np.asarray(features[self.bounds[0].feature], dtype=np.float64)
        )
        for bound in self.bounds:
            total = total + bound.violation(features)
        return total


def compute_violations(
    user: SimulatedUser, context: MpcCostContext, trajectory: np.ndarray
) -> np.ndarray:
    """Return the hidden-cost violation series for a ``(..., 3, 3)`` trajectory."""
    return user.violation_series(feature_series(context, trajectory))


def first_violation_step(
    user: SimulatedUser,
    context: MpcCostContext,
    trajectory: np.ndarray,
    threshold: float = 0.02,
) -> int | None:
    """Return the first frame whose violation exceeds ``threshold`` radians."""
    violations = compute_violations(user, context, trajectory)
    indices = np.nonzero(violations > threshold)[0]
    return int(indices[0]) if indices.size > 0 else None


def violation_metrics(
    user: SimulatedUser, context: MpcCostContext, trajectory: np.ndarray
) -> dict[str, float]:
    """Return summary violation statistics for a ``(T, 3, 3)`` trajectory."""
    violations = compute_violations(user, context, trajectory)
    return {
        "mean_violation": float(np.mean(violations)),
        "max_violation": float(np.max(violations)),
        "frac_frames_violated": float(np.mean(violations > 0.0)),
    }


def choose_cluster(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
) -> int:
    """Return the cluster label whose mean trajectory the user finds most comfortable."""
    scores = {
        label: float(np.mean(compute_violations(user, context, traj)))
        for label, traj in cluster_means.items()
    }
    return min(sorted(scores), key=lambda label: scores[label])


@dataclass(frozen=True)
class HiddenCostTerm(TrajectoryCost):
    """Oracle planner cost: the hidden restriction exposed as a rollout cost.

    Used only for the oracle condition in experiments (upper bound on what any
    generated cost could achieve); never given to the cost generator.
    """

    user: SimulatedUser
    context: MpcCostContext
    weight: float = 1.0

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        q_trajs = np.asarray(q_trajs, dtype=np.float64)
        future = q_trajs[:, 1:] if q_trajs.shape[1] > 1 else q_trajs
        violations = self.user.violation_series(feature_series(self.context, future))
        return self.weight * np.mean(violations**2, axis=1)
