"""Reusable extra cost terms for left-arm MPC planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


class TrajectoryCost(Protocol):
    """Cost term evaluated on a batch of candidate MPC rollouts."""

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        """Return one scalar cost per rollout."""


@dataclass(frozen=True)
class MpcCostContext:
    """Shared FK context needed by Cartesian feature costs."""

    fk: SmplLeftArmFK
    spine3_pos: np.ndarray
    spine3_aa: np.ndarray
    fixed_collar_aa: np.ndarray


class CompositeTrajectoryCost:
    """Sum a set of optional rollout cost terms."""

    def __init__(self, terms: list[TrajectoryCost] | None = None) -> None:
        self._terms = list(terms) if terms else []

    def __bool__(self) -> bool:
        return bool(self._terms)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        total = np.zeros(q_trajs.shape[0], dtype=np.float64)
        for term in self._terms:
            total = total + term(q_trajs)
        return total


@dataclass(frozen=True)
class ElbowHeightCost:
    """Keep rollout left-elbow height within a spine3-relative range."""

    min_height: float
    max_height: float
    weight: float
    progress_weight: float
    context: MpcCostContext

    def __post_init__(self) -> None:
        if self.min_height >= self.max_height:
            raise ValueError("elbow_height min must be less than max.")
        if self.weight < 0:
            raise ValueError("elbow_height weight must be non-negative.")
        if self.progress_weight < 0:
            raise ValueError("elbow_height progress_weight must be non-negative.")

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        q_trajs = np.asarray(q_trajs, dtype=np.float64)
        future_q = q_trajs[:, 1:] if q_trajs.shape[1] > 1 else q_trajs[:, -1:]
        future_heights = self._elbow_heights(future_q.reshape(-1, 3, 3)).reshape(
            future_q.shape[:2]
        )
        future_violation = self._range_violation(future_heights)
        range_cost = self.weight * np.mean(future_violation**2, axis=1)

        initial_height = self._elbow_heights(q_trajs[:, 0])
        initial_violation = self._range_violation(initial_height)
        worsened_violation = np.maximum(
            future_violation - initial_violation[:, np.newaxis], 0.0
        )
        started_outside = initial_violation > 0.0
        progress_cost = self.progress_weight * np.mean(worsened_violation**2, axis=1)
        progress_cost = np.where(started_outside, progress_cost, 0.0)
        return range_cost + progress_cost

    def _elbow_heights(self, arm_aa: np.ndarray) -> np.ndarray:
        positions = self.context.fk.fk_controlled_batch(
            arm_aa,
            self.context.fixed_collar_aa,
            self.context.spine3_pos,
            self.context.spine3_aa,
        )
        return positions[:, 3, 1] - self.context.spine3_pos[1]

    def _range_violation(self, elbow_height: np.ndarray) -> np.ndarray:
        low_violation = np.maximum(self.min_height - elbow_height, 0.0)
        high_violation = np.maximum(elbow_height - self.max_height, 0.0)
        return low_violation + high_violation


def build_extra_costs(
    cost_configs: dict[str, dict[str, Any]] | None,
    context: MpcCostContext,
) -> CompositeTrajectoryCost:
    """Build configured extra MPC costs from YAML-shaped dictionaries."""
    if not cost_configs:
        return CompositeTrajectoryCost()

    terms: list[TrajectoryCost] = []
    for name, params in cost_configs.items():
        builder = COST_BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"Unknown MPC cost '{name}'.")
        if not isinstance(params, dict):
            raise ValueError(f"{name} config must be a mapping.")
        terms.append(builder(params, context))
    return CompositeTrajectoryCost(terms)


CostBuilder = Callable[[dict[str, Any], MpcCostContext], TrajectoryCost]


def _build_elbow_height(
    params: dict[str, Any], context: MpcCostContext
) -> ElbowHeightCost:
    return ElbowHeightCost(
        min_height=float(params["min"]),
        max_height=float(params["max"]),
        weight=float(params.get("weight", 1.0)),
        progress_weight=float(params.get("progress_weight", params.get("weight", 1.0))),
        context=context,
    )


COST_BUILDERS: dict[str, CostBuilder] = {
    "elbow_height": _build_elbow_height,
}


def available_cost_names() -> set[str]:
    """Return registered YAML cost keys."""
    return set(COST_BUILDERS)
