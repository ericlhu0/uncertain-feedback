"""Goal regions: sets of acceptable terminal arm states the MPC steers into.

A region's ``distance`` is zero everywhere inside it, so the goal term of the
stage cost vanishes once a rollout ends in the region and only the comfort
costs decide where the arm settles. A single wrist point is the degenerate
``PointRegion``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    ArmFeatureContext,
    arm_feature_series,
)

_CIRCLE_POINTS = 33


class GoalRegion(ABC):
    """A set of acceptable terminal states, in spine3-relative coordinates."""

    @abstractmethod
    def distance(
        self, wrist_rel: np.ndarray, arm_aa: np.ndarray, context: ArmFeatureContext
    ) -> np.ndarray:
        """``(N,)`` distance from each terminal state to the region; 0 inside.

        Args:
            wrist_rel: ``(N, 3)`` spine3-relative wrist positions.
            arm_aa: ``(N, 3, 3)`` terminal arm axis-angles.
            context: FK context for regions defined over anatomical features.
        """

    def marker(self) -> np.ndarray | None:
        """A representative spine3-relative wrist point, or ``None``."""
        return None

    def outline(self) -> list[np.ndarray]:
        """Spine3-relative ``(K, 3)`` polylines tracing the region's boundary."""
        return []


@dataclass(frozen=True)
class PointRegion(GoalRegion):
    """A single wrist position; the distance is the plain L2 norm."""

    point: tuple[float, float, float]

    def distance(
        self, wrist_rel: np.ndarray, arm_aa: np.ndarray, context: ArmFeatureContext
    ) -> np.ndarray:
        del arm_aa, context
        return np.linalg.norm(wrist_rel - np.asarray(self.point), axis=-1)

    def marker(self) -> np.ndarray:
        return np.asarray(self.point, dtype=np.float64)


@dataclass(frozen=True)
class BoxRegion(GoalRegion):
    """An axis-aligned box of wrist positions."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def distance(
        self, wrist_rel: np.ndarray, arm_aa: np.ndarray, context: ArmFeatureContext
    ) -> np.ndarray:
        del arm_aa, context
        low, high = np.asarray(self.low), np.asarray(self.high)
        outside = np.maximum(low - wrist_rel, 0.0) + np.maximum(wrist_rel - high, 0.0)
        return np.linalg.norm(outside, axis=-1)

    def marker(self) -> np.ndarray:
        return (np.asarray(self.low) + np.asarray(self.high)) / 2.0

    def outline(self) -> list[np.ndarray]:
        low, high = np.asarray(self.low), np.asarray(self.high)
        corners = np.array(
            [
                [[low[0], high[0]][i], [low[1], high[1]][j], [low[2], high[2]][k]]
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ]
        )
        edges = [
            (a, b)
            for a in range(8)
            for b in range(a + 1, 8)
            if bin(a ^ b).count("1") == 1
        ]
        return [corners[[a, b]] for a, b in edges]


@dataclass(frozen=True)
class SphereRegion(GoalRegion):
    """A ball of wrist positions around ``center``."""

    center: tuple[float, float, float]
    radius: float

    def distance(
        self, wrist_rel: np.ndarray, arm_aa: np.ndarray, context: ArmFeatureContext
    ) -> np.ndarray:
        del arm_aa, context
        radial = np.linalg.norm(wrist_rel - np.asarray(self.center), axis=-1)
        return np.maximum(radial - self.radius, 0.0)

    def marker(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    def outline(self) -> list[np.ndarray]:
        center = np.asarray(self.center, dtype=np.float64)
        theta = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_POINTS)
        cos, sin, zero = np.cos(theta), np.sin(theta), np.zeros_like(theta)
        circles = (
            np.stack([cos, sin, zero], axis=-1),
            np.stack([cos, zero, sin], axis=-1),
            np.stack([zero, cos, sin], axis=-1),
        )
        return [center + self.radius * circle for circle in circles]


@dataclass(frozen=True)
class FeatureRegion(GoalRegion):
    """A box in anatomical-feature space (radians), independent of wrist position.

    ``bounds`` maps a :data:`FEATURE_NAMES` entry to ``(low, high)``; either
    side may be ``None`` for one-sided bounds.
    """

    bounds: Mapping[str, tuple[float | None, float | None]]

    def distance(
        self, wrist_rel: np.ndarray, arm_aa: np.ndarray, context: ArmFeatureContext
    ) -> np.ndarray:
        del wrist_rel
        features = arm_feature_series(arm_aa, context)
        hinges = []
        for name, (low, high) in self.bounds.items():
            value = features[name]
            hinge = np.zeros_like(value)
            if low is not None:
                hinge = hinge + np.maximum(low - value, 0.0)
            if high is not None:
                hinge = hinge + np.maximum(value - high, 0.0)
            hinges.append(hinge)
        return np.linalg.norm(np.stack(hinges, axis=-1), axis=-1)


def as_goal_region(goal: Sequence[float] | np.ndarray | GoalRegion) -> GoalRegion:
    """Wrap a raw ``[x, y, z]`` goal as a :class:`PointRegion`; pass regions through."""
    if isinstance(goal, GoalRegion):
        return goal
    x, y, z = (float(v) for v in np.asarray(goal, dtype=np.float64))
    return PointRegion((x, y, z))


def goal_point(goal: Sequence[float] | np.ndarray | GoalRegion) -> np.ndarray | None:
    """The representative wrist point of a goal, for consumers that need one."""
    return as_goal_region(goal).marker()


__all__ = [
    "GoalRegion",
    "PointRegion",
    "BoxRegion",
    "SphereRegion",
    "FeatureRegion",
    "as_goal_region",
    "goal_point",
]
