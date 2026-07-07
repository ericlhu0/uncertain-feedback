"""Reusable extra cost terms for left-arm MPC planners."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_JOINT_BEFORE_WRIST_POS_IDX = -2
_SHOULDER_POS_IDX = 2
_ELBOW_POS_IDX = 3
_WRIST_POS_IDX = 4
_TORSO_DOWN = np.array([0.0, -1.0, 0.0], dtype=np.float64)


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


@runtime_checkable
class LearnablePreferenceCost(TrajectoryCost, Protocol):
    """Range cost whose bounds can be updated from preference demonstrations."""

    cost_name: str
    weight: float
    progress_weight: float
    context: MpcCostContext

    @property
    def min_value(self) -> float:
        """Lower scalar feature bound."""

    @property
    def max_value(self) -> float:
        """Upper scalar feature bound."""

    def feature_values(self, trajectory: np.ndarray) -> np.ndarray:
        """Return one scalar preference feature per trajectory frame."""

    def with_range(
        self, min_value: float, max_value: float
    ) -> "LearnablePreferenceCost":
        """Return a copy with updated scalar bounds."""


class CompositeTrajectoryCost:
    """Sum a set of optional rollout cost terms."""

    def __init__(self, terms: list[TrajectoryCost] | None = None) -> None:
        self._terms = list(terms) if terms else []

    def __bool__(self) -> bool:
        return bool(self._terms)

    def terms(self) -> tuple[TrajectoryCost, ...]:
        """Return configured cost terms."""
        return tuple(self._terms)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        total = np.zeros(q_trajs.shape[0], dtype=np.float64)
        for term in self._terms:
            total = total + term(q_trajs)
        return total


def _validate_range_cost(
    name: str,
    min_value: float,
    max_value: float,
    weight: float,
    progress_weight: float,
) -> None:
    if min_value >= max_value:
        raise ValueError(f"{name} min must be less than max.")
    if weight < 0:
        raise ValueError(f"{name} weight must be non-negative.")
    if progress_weight < 0:
        raise ValueError(f"{name} progress_weight must be non-negative.")


def _range_violation(
    values: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    low_violation = np.maximum(min_value - values, 0.0)
    high_violation = np.maximum(values - max_value, 0.0)
    return low_violation + high_violation


def _range_rollout_cost(
    q_trajs: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    min_value: float,
    max_value: float,
    weight: float,
    progress_weight: float,
) -> np.ndarray:
    q_trajs = np.asarray(q_trajs, dtype=np.float64)
    future_q = q_trajs[:, 1:] if q_trajs.shape[1] > 1 else q_trajs[:, -1:]
    future_values = feature_fn(future_q.reshape(-1, 3, 3)).reshape(future_q.shape[:2])
    future_violation = _range_violation(future_values, min_value, max_value)
    range_cost = weight * np.mean(future_violation**2, axis=1)

    initial_values = feature_fn(q_trajs[:, 0])
    initial_violation = _range_violation(initial_values, min_value, max_value)
    worsened_violation = np.maximum(
        future_violation - initial_violation[:, np.newaxis], 0.0
    )
    started_outside = initial_violation > 0.0
    progress_cost = progress_weight * np.mean(worsened_violation**2, axis=1)
    progress_cost = np.where(started_outside, progress_cost, 0.0)
    return range_cost + progress_cost


@dataclass(frozen=True)
class ElbowHeightCost:
    """Keep rollout left-elbow height within a spine3-relative range."""

    cost_name = "elbow_height"

    min_height: float
    max_height: float
    weight: float
    progress_weight: float
    context: MpcCostContext

    def __post_init__(self) -> None:
        _validate_range_cost(
            self.cost_name,
            self.min_height,
            self.max_height,
            self.weight,
            self.progress_weight,
        )

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        return _range_rollout_cost(
            q_trajs,
            self.feature_values,
            self.min_height,
            self.max_height,
            self.weight,
            self.progress_weight,
        )

    @property
    def min_value(self) -> float:
        """Lower elbow-height bound."""
        return self.min_height

    @property
    def max_value(self) -> float:
        """Upper elbow-height bound."""
        return self.max_height

    def feature_values(self, trajectory: np.ndarray) -> np.ndarray:
        """Return spine3-relative elbow heights."""
        return compute_elbow_heights(trajectory, self.context)

    def with_range(self, min_value: float, max_value: float) -> "ElbowHeightCost":
        """Return a copy with updated height bounds."""
        return dataclasses.replace(
            self,
            min_height=min_value,
            max_height=max_value,
        )


@dataclass(frozen=True)
class ElbowFlexionAngleCost:
    """Keep rollout left-elbow rotation magnitude within a range in radians."""

    cost_name = "elbow_flexion_angle"

    min_angle: float
    max_angle: float
    weight: float
    progress_weight: float
    context: MpcCostContext

    def __post_init__(self) -> None:
        _validate_range_cost(
            self.cost_name,
            self.min_angle,
            self.max_angle,
            self.weight,
            self.progress_weight,
        )

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        return _range_rollout_cost(
            q_trajs,
            self.feature_values,
            self.min_angle,
            self.max_angle,
            self.weight,
            self.progress_weight,
        )

    @property
    def min_value(self) -> float:
        """Lower elbow-flexion bound."""
        return self.min_angle

    @property
    def max_value(self) -> float:
        """Upper elbow-flexion bound."""
        return self.max_angle

    def feature_values(self, trajectory: np.ndarray) -> np.ndarray:
        """Return elbow flexion angles."""
        return compute_elbow_flexion_angles(trajectory, self.context)

    def with_range(self, min_value: float, max_value: float) -> "ElbowFlexionAngleCost":
        """Return a copy with updated angle bounds."""
        return dataclasses.replace(
            self,
            min_angle=min_value,
            max_angle=max_value,
        )


@dataclass(frozen=True)
class ShoulderAbductionAngleCost:
    """Keep rollout upper-arm abduction within a range in radians."""

    cost_name = "shoulder_abduction_angle"

    min_angle: float
    max_angle: float
    weight: float
    progress_weight: float
    context: MpcCostContext

    def __post_init__(self) -> None:
        _validate_range_cost(
            self.cost_name,
            self.min_angle,
            self.max_angle,
            self.weight,
            self.progress_weight,
        )

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        return _range_rollout_cost(
            q_trajs,
            self.feature_values,
            self.min_angle,
            self.max_angle,
            self.weight,
            self.progress_weight,
        )

    @property
    def min_value(self) -> float:
        """Lower shoulder-abduction bound."""
        return self.min_angle

    @property
    def max_value(self) -> float:
        """Upper shoulder-abduction bound."""
        return self.max_angle

    def feature_values(self, trajectory: np.ndarray) -> np.ndarray:
        """Return shoulder abduction angles."""
        return compute_shoulder_abduction_angles(trajectory, self.context)

    def with_range(
        self,
        min_value: float,
        max_value: float,
    ) -> "ShoulderAbductionAngleCost":
        """Return a copy with updated angle bounds."""
        return dataclasses.replace(
            self,
            min_angle=min_value,
            max_angle=max_value,
        )


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


def _build_elbow_flexion_angle(
    params: dict[str, Any], context: MpcCostContext
) -> ElbowFlexionAngleCost:
    return ElbowFlexionAngleCost(
        min_angle=float(params["min"]),
        max_angle=float(params["max"]),
        weight=float(params.get("weight", 1.0)),
        progress_weight=float(params.get("progress_weight", params.get("weight", 1.0))),
        context=context,
    )


def _build_shoulder_abduction_angle(
    params: dict[str, Any], context: MpcCostContext
) -> ShoulderAbductionAngleCost:
    return ShoulderAbductionAngleCost(
        min_angle=float(params["min"]),
        max_angle=float(params["max"]),
        weight=float(params.get("weight", 1.0)),
        progress_weight=float(params.get("progress_weight", params.get("weight", 1.0))),
        context=context,
    )


COST_BUILDERS: dict[str, CostBuilder] = {
    "elbow_height": _build_elbow_height,
    "elbow_flexion_angle": _build_elbow_flexion_angle,
    "shoulder_abduction_angle": _build_shoulder_abduction_angle,
}


def available_cost_names() -> set[str]:
    """Return registered YAML cost keys."""
    return set(COST_BUILDERS)


# ---------------------------------------------------------------------------
# Preference learning utilities
# ---------------------------------------------------------------------------


def compute_elbow_heights(
    trajectory: np.ndarray,
    context: MpcCostContext,
) -> np.ndarray:
    """Return spine3-relative elbow Y-heights for each frame in a trajectory.

    Args:
        trajectory: ``(N, 3, 3)`` axis-angle frames (shoulder/elbow/wrist).
        context:    Shared FK context with spine3 position and collar angle.

    Returns:
        ``(N,)`` elbow heights relative to spine3.
    """
    positions = context.fk.fk_batch(
        trajectory,
        context.spine3_pos,
        context.spine3_aa,
    )
    return positions[:, _JOINT_BEFORE_WRIST_POS_IDX, 1] - context.spine3_pos[1]


def compute_elbow_flexion_angles(
    trajectory: np.ndarray,
    context: MpcCostContext,
) -> np.ndarray:
    """Return elbow bend as the angle between upper arm and forearm, in radians.

    0 = fully extended; larger = more bent. Computed from FK positions because
    under this repo's FK convention (joint *j*'s rotation transforms the bone
    arriving at *j*) the anatomical bend is encoded in the wrist-slot rotvec,
    so the elbow-slot rotvec magnitude does not measure it.
    """
    positions = context.fk.fk_batch(
        trajectory,
        context.spine3_pos,
        context.spine3_aa,
    )
    upper_arm = positions[:, _ELBOW_POS_IDX] - positions[:, _SHOULDER_POS_IDX]
    forearm = positions[:, _WRIST_POS_IDX] - positions[:, _ELBOW_POS_IDX]
    dots = np.sum(upper_arm * forearm, axis=-1)
    norms = np.linalg.norm(upper_arm, axis=-1) * np.linalg.norm(forearm, axis=-1)
    cos = np.divide(dots, norms, out=np.ones_like(dots), where=norms > 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def compute_shoulder_abduction_angles(
    trajectory: np.ndarray,
    context: MpcCostContext,
) -> np.ndarray:
    """Return unsigned upper-arm abduction angles in the spine3 frame.

    The angle is measured between the shoulder-to-elbow vector and torso-down
    direction. Larger values mean the upper arm is farther away from the torso.
    """
    positions = context.fk.fk_batch(
        trajectory,
        context.spine3_pos,
        context.spine3_aa,
    )
    upper_arm_world = positions[:, _ELBOW_POS_IDX] - positions[:, _SHOULDER_POS_IDX]
    upper_arm_norm = np.linalg.norm(upper_arm_world, axis=1)
    safe_upper_arm = np.divide(
        upper_arm_world,
        upper_arm_norm[:, np.newaxis],
        out=np.zeros_like(upper_arm_world),
        where=upper_arm_norm[:, np.newaxis] > 1e-12,
    )
    spine_inv = Rotation.from_rotvec(context.spine3_aa).inv()
    upper_arm_local = spine_inv.apply(safe_upper_arm)
    dots = np.clip(upper_arm_local @ _TORSO_DOWN, -1.0, 1.0)
    return np.arccos(dots)


def update_elbow_cost(
    cost: ElbowHeightCost,
    mdm_heights: np.ndarray,
    mpc_heights: np.ndarray,
    alpha: float = 0.5,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> ElbowHeightCost:
    """Return a new :class:`ElbowHeightCost` with one side updated from MDM.

    Compares where MPC was naturally operating (``mpc_heights``) against where MDM
    demonstrated the elbow should be (``mdm_heights``). If MPC was lower than MDM,
    snap only the lower bound to MDM's robust lower percentile. If MPC was higher
    than MDM, snap only the upper bound to MDM's robust upper percentile.

    Args:
        cost:        Existing cost to update from.
        mdm_heights: ``(n,)`` elbow heights from the MDM trajectory.
        mpc_heights: ``(n,)`` elbow heights from recent executed MPC steps.
        alpha:       Accepted for config compatibility; unused by this snap update.
        low_pct:     Percentile used for MDM's robust lower bound.
        high_pct:    Percentile used for MDM's robust upper bound.

    Returns:
        New frozen :class:`ElbowHeightCost` with updated ``min_height`` / ``max_height``.
    """
    _ = alpha
    mdm_lo, mdm_hi = np.percentile(mdm_heights, [low_pct, high_pct])
    mdm_lo = float(mdm_lo)
    mdm_hi = float(mdm_hi)
    mdm_mean = float(np.mean(mdm_heights))
    mpc_mean = float(np.mean(mpc_heights))

    new_min = cost.min_height
    new_max = cost.max_height
    if np.isclose(mpc_mean, mdm_mean):
        return cost
    if mpc_mean < mdm_mean:
        new_min = mdm_lo
    else:
        new_max = mdm_hi
    if new_min >= new_max:
        new_min = mdm_lo
        new_max = mdm_hi
    return dataclasses.replace(cost, min_height=new_min, max_height=new_max)


def update_preference_cost(
    cost: LearnablePreferenceCost,
    mdm_values: np.ndarray,
    mpc_values: np.ndarray,
    alpha: float = 0.5,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> LearnablePreferenceCost:
    """Return a cost with one side of its scalar range updated from MDM."""
    _ = alpha
    mdm_lo, mdm_hi = np.percentile(mdm_values, [low_pct, high_pct])
    mdm_lo = float(mdm_lo)
    mdm_hi = float(mdm_hi)
    mdm_mean = float(np.mean(mdm_values))
    mpc_mean = float(np.mean(mpc_values))

    new_min = cost.min_value
    new_max = cost.max_value
    if np.isclose(mpc_mean, mdm_mean):
        return cost
    if mpc_mean < mdm_mean:
        new_min = mdm_lo
    else:
        new_max = mdm_hi
    if new_min >= new_max:
        new_min = mdm_lo
        new_max = mdm_hi
    return cost.with_range(new_min, new_max)


def replace_cost_in_composite(
    composite: CompositeTrajectoryCost,
    new_term: TrajectoryCost,
) -> CompositeTrajectoryCost:
    """Return a new composite with the first term of the same type replaced.

    If no term of ``type(new_term)`` exists in the composite, ``new_term`` is
    appended.
    """
    replaced = False
    updated: list[TrajectoryCost] = []
    for term in composite.terms():
        if not replaced and type(term) is type(new_term):
            updated.append(new_term)
            replaced = True
        else:
            updated.append(term)
    if not replaced:
        updated.append(new_term)
    return CompositeTrajectoryCost(updated)
