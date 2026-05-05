"""YAML configuration for MPC controller settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from uncertain_feedback.planners.mpc.costs import available_cost_names

PLANNER_CHOICES = {
    "arm_mpc",
    "arm_mpc_mdm",
    "arm_mpc_mdm_uq",
    "arm_mpc_cartesian",
    "arm_mpc_cartesian_no_mdm",
}


@dataclass(frozen=True)
class UqConfig:
    diffusion_samples: int = 128
    n_clusters: int = 3
    auto_cluster: int | None = None


@dataclass(frozen=True)
class CartesianConfig:
    goals: list[list[float]] = field(default_factory=list)
    threshold: float = 0.05


@dataclass(frozen=True)
class MpcRunConfig:
    planner: str
    steps: int
    horizon: int
    n_mpc_samples: int
    max_angle_delta: float
    pose: Path | None
    goal_threshold: float
    advance_threshold: float
    trajectory_fraction: float
    uq: UqConfig
    cartesian: CartesianConfig
    costs: dict[str, dict[str, Any]]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a path string.")
    return Path(value)


def load_mpc_config(path: Path) -> MpcRunConfig:
    """Load required MPC controller settings from YAML."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    data = _mapping(raw, "MPC config")

    planner = str(data.get("planner", ""))
    if planner not in PLANNER_CHOICES:
        raise ValueError(
            f"planner must be one of {sorted(PLANNER_CHOICES)}; got {planner!r}."
        )

    uq_data = _mapping(data.get("uq"), "uq")
    cartesian_data = _mapping(data.get("cartesian"), "cartesian")
    cost_data = _mapping(data.get("costs"), "costs")

    goals = cartesian_data.get("goals", [])
    if goals is None:
        goals = []
    if not isinstance(goals, list):
        raise ValueError("cartesian.goals must be a list.")
    normalized_goals: list[list[float]] = []
    for idx, goal in enumerate(goals):
        if not isinstance(goal, list) or len(goal) != 3:
            raise ValueError(f"cartesian.goals[{idx}] must be a 3-number list.")
        normalized_goals.append([_float(v, f"cartesian.goals[{idx}]") for v in goal])

    costs: dict[str, dict[str, Any]] = {}
    cost_names = available_cost_names()
    for name, params in cost_data.items():
        if name not in cost_names:
            raise ValueError(f"Unknown MPC cost '{name}'.")
        params_map = _mapping(params, f"costs.{name}")
        costs[name] = dict(params_map)

    return MpcRunConfig(
        planner=planner,
        steps=_positive_int(data.get("steps"), "steps"),
        horizon=_positive_int(data.get("horizon"), "horizon"),
        n_mpc_samples=_positive_int(data.get("n_mpc_samples"), "n_mpc_samples"),
        max_angle_delta=_float(data.get("max_angle_delta"), "max_angle_delta"),
        pose=_optional_path(data.get("pose"), "pose"),
        goal_threshold=_float(data.get("goal_threshold", 0.01), "goal_threshold"),
        advance_threshold=_float(
            data.get("advance_threshold", 0.1), "advance_threshold"
        ),
        trajectory_fraction=_float(
            data.get("trajectory_fraction", 1.0), "trajectory_fraction"
        ),
        uq=UqConfig(
            diffusion_samples=_positive_int(
                uq_data.get("diffusion_samples", 128), "uq.diffusion_samples"
            ),
            n_clusters=_positive_int(uq_data.get("n_clusters", 3), "uq.n_clusters"),
            auto_cluster=(
                None
                if uq_data.get("auto_cluster") is None
                else int(uq_data["auto_cluster"])
            ),
        ),
        cartesian=CartesianConfig(
            goals=normalized_goals,
            threshold=_float(
                cartesian_data.get("threshold", 0.05), "cartesian.threshold"
            ),
        ),
        costs=costs,
    )
