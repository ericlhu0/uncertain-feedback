"""Simulated care recipients with hidden comfort costs.

A :class:`SimulatedUser` holds range-of-motion restrictions the planner does not
know about. It plays the care recipient in headless experiments: it decides when
the robot's motion becomes uncomfortable (:func:`first_violation_step`), what it
says (``feedback_text``), which UQ cluster it prefers (:func:`choose_cluster`),
and scores any trajectory by how much the hidden restrictions are violated
(:func:`violation_metrics`). The hidden cost is never shown to the cost
generator — it is the ground truth the generated cost is evaluated against.

Restrictions are expressed over the same five anatomical joint features the cost
generator uses (see ``GeneratedCostContext``), so a hidden bound and a generated
bound are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_aa_from_state,
    arm_feature_series,
)
from uncertain_feedback.planners.mpc.costs.base import (
    JointLimitCost,
    MpcCostContext,
    TrajectoryCost,
)

BOUND_TYPES = ("upper_bound", "lower_bound", "avoid_band")


def feature_series(
    context: MpcCostContext, trajectory: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the canonical anatomical features for q or axis-angle states."""
    return arm_feature_series(trajectory, context)


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
class CoupledBound:
    """Pose-dependent limit: the threshold on ``feature`` moves linearly with
    ``cond_feature``.

    ``threshold = intercept + slope * cond_value``. ``upper_bound`` penalizes the
    feature above the threshold, ``lower_bound`` below it. Where the line leaves
    the feature's physical range the bound is naturally inactive (e.g. a negative
    lower bound on a non-negative feature).
    """

    feature: str
    bound_type: str
    cond_feature: str
    intercept: float
    slope: float

    def __post_init__(self) -> None:
        if self.feature not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature {self.feature!r}.")
        if self.cond_feature not in FEATURE_NAMES:
            raise ValueError(f"Unknown cond_feature {self.cond_feature!r}.")
        if self.bound_type not in ("upper_bound", "lower_bound"):
            raise ValueError(
                f"CoupledBound bound_type must be upper_bound or lower_bound; "
                f"got {self.bound_type!r}."
            )

    def threshold(self, cond_values: np.ndarray) -> np.ndarray:
        """Return the pose-dependent threshold for conditioning-feature values."""
        return self.intercept + self.slope * np.asarray(cond_values, dtype=np.float64)

    def violation(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Return per-frame violation magnitudes (radians), zero when satisfied."""
        values = np.asarray(features[self.feature], dtype=np.float64)
        threshold = self.threshold(features[self.cond_feature])
        if self.bound_type == "upper_bound":
            return np.maximum(values - threshold, 0.0)
        return np.maximum(threshold - values, 0.0)


Bound = HiddenBound | CoupledBound

# Controlled arm_aa row for each joint slot. Under the repo FK convention
# (a joint's rotation transforms the bone arriving at it) the slots physically
# drive: left_shoulder -> clavicle, left_elbow -> upper arm, left_wrist -> forearm.
JOINT_SLOTS = {"left_shoulder": 0, "left_elbow": 1, "left_wrist": 2}


@dataclass(frozen=True)
class JointBoxLimit:
    """Per-axis box on one controlled joint slot's axis-angle (radians).

    Anatomical joint limits shared by every persona — unlike feature bounds,
    these act on the raw controlled rotation, so they also catch implausible
    configurations (e.g. large clavicle swings) that the anatomical joint
    features cannot see.
    """

    joint: str
    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.joint not in JOINT_SLOTS:
            raise ValueError(
                f"Unknown joint {self.joint!r}; expected {sorted(JOINT_SLOTS)}."
            )

    def violation(self, trajectory: np.ndarray) -> np.ndarray:
        """Return per-frame violation magnitudes for ``(..., 3, 3)`` trajectories."""
        q = np.asarray(trajectory, dtype=np.float64)[..., JOINT_SLOTS[self.joint], :]
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        return (np.maximum(q - high, 0.0) + np.maximum(low - q, 0.0)).sum(axis=-1)


@dataclass(frozen=True)
class SimulatedUser:
    """A care recipient persona with hidden ROM restrictions."""

    name: str
    description: str
    feedback_text: str
    bounds: tuple[Bound, ...]
    joint_limits: tuple[JointBoxLimit, ...] = ()

    def violation_series(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Return summed per-frame violations across all hidden bounds."""
        total = np.zeros_like(
            np.asarray(next(iter(features.values())), dtype=np.float64)
        )
        for bound in self.bounds:
            total = total + bound.violation(features)
        return total

    def limit_cost(self, weight: float = 50.0) -> JointLimitCost:
        """Return the joint-box-limit penalty over this persona's controlled slots.

        The same anatomical box used to *score* trajectories, turned into an
        always-on MPC cost so the planner also refuses to *generate* motions that
        leave the box (see JointLimitCost).
        """
        slots = tuple(JOINT_SLOTS[limit.joint] for limit in self.joint_limits)
        low = np.array([limit.low for limit in self.joint_limits], dtype=np.float64)
        high = np.array([limit.high for limit in self.joint_limits], dtype=np.float64)
        return JointLimitCost(slots=slots, low=low, high=high, weight=weight)

    def limit_violation_series(self, trajectory: np.ndarray) -> np.ndarray:
        """Return joint-box violations for FK-boundary ``(..., 3, 3)`` data."""
        trajectory = np.asarray(trajectory, dtype=np.float64)
        total = np.zeros(trajectory.shape[:-2], dtype=np.float64)
        for limit in self.joint_limits:
            total = total + limit.violation(trajectory)
        return total


def compute_violations(
    user: SimulatedUser, context: MpcCostContext, trajectory: np.ndarray
) -> np.ndarray:
    """Return the hidden-cost violation series for q or axis-angle states."""
    return user.violation_series(
        feature_series(context, trajectory)
    ) + user.limit_violation_series(arm_aa_from_state(trajectory, context))


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
    weight: float = 10.0

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        q_trajs = np.asarray(q_trajs, dtype=np.float64)
        future = q_trajs[:, 1:] if q_trajs.shape[1] > 1 else q_trajs
        violations = self.user.violation_series(
            feature_series(self.context, future)
        ) + self.user.limit_violation_series(arm_aa_from_state(future, self.context))
        return self.weight * np.mean(violations**2, axis=1)
