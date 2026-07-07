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
    scale: float = 1.0


@dataclass(frozen=True)
class CartesianConfig:
    goals: list[list[float]] = field(default_factory=list)
    threshold: float = 0.05


@dataclass(frozen=True)
class LlmCostConfig:
    enabled: bool = False
    model: str | None = None
    strict: bool = False
    artifact_dir: Path = Path("llm_cost_artifacts")
    use_images: bool = True
    # Which cost generator to use: "llm" (single-turn, default), "turns"
    # (multi-turn conversation), or "agent" (delegate to the codex CLI).
    backend: str = "llm"
    max_turns: int = 6  # used by the "turns" backend
    # Used by the "agent" backend. --skip-git-repo-check is required because the
    # per-generation artifact run dir is not a git repo. Depending on your codex
    # auth/host you may also need e.g. `-m <model>` and a sandbox flag.
    codex_cmd: str = "codex exec --skip-git-repo-check"


COST_BACKENDS = {"llm", "turns", "agent"}


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
    max_playback_delta: float
    trajectory_fraction: float
    uq: UqConfig
    cartesian: CartesianConfig
    costs: dict[str, dict[str, Any]]
    llm_cost: LlmCostConfig
    mdm_frames: int | None = None
    num_denoising_steps: int | None = None  # kimodo DDIM steps; None = backend default
    text_time: int = 0
    preference_learning: bool = True
    preference_alpha: float = 0.5
    preference_window: int = 50
    motion_generator: str = "mdm"


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


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a path string.")
    return Path(value)


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null.")
    return value


def _cost_backend(value: Any) -> str:
    backend = value if value is not None else "llm"
    if backend not in COST_BACKENDS:
        raise ValueError(
            f"llm_cost.backend must be one of {sorted(COST_BACKENDS)}, got {value!r}."
        )
    return backend


def _str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings.")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{idx}] must be a string.")
        out.append(item)
    return out


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

    from uncertain_feedback.motion_generators import (  # pylint: disable=import-outside-toplevel
        MOTION_GENERATOR_BUILDERS,
    )

    motion_generator = str(data.get("motion_generator", "mdm"))
    if motion_generator not in MOTION_GENERATOR_BUILDERS:
        raise ValueError(
            "motion_generator must be one of "
            f"{sorted(MOTION_GENERATOR_BUILDERS)}; got {motion_generator!r}."
        )

    uq_data = _mapping(data.get("uq"), "uq")
    cartesian_data = _mapping(data.get("cartesian"), "cartesian")
    cost_data = _mapping(data.get("costs"), "costs")
    llm_cost_data = _mapping(data.get("llm_cost"), "llm_cost")

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
        max_playback_delta=_float(
            data.get("max_playback_delta", 0.1), "max_playback_delta"
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
            scale=_float(uq_data.get("scale", 1.0), "uq.scale"),
        ),
        cartesian=CartesianConfig(
            goals=normalized_goals,
            threshold=_float(
                cartesian_data.get("threshold", 0.05), "cartesian.threshold"
            ),
        ),
        costs=costs,
        llm_cost=LlmCostConfig(
            enabled=_bool(llm_cost_data.get("enabled", False), "llm_cost.enabled"),
            model=_optional_str(llm_cost_data.get("model"), "llm_cost.model"),
            strict=_bool(llm_cost_data.get("strict", False), "llm_cost.strict"),
            artifact_dir=Path(
                _optional_str(
                    llm_cost_data.get("artifact_dir", "llm_cost_artifacts"),
                    "llm_cost.artifact_dir",
                )
                or "llm_cost_artifacts"
            ),
            use_images=_bool(
                llm_cost_data.get("use_images", True), "llm_cost.use_images"
            ),
            backend=_cost_backend(llm_cost_data.get("backend", "llm")),
            max_turns=_positive_int(
                llm_cost_data.get("max_turns", 6), "llm_cost.max_turns"
            ),
            codex_cmd=(
                _optional_str(
                    llm_cost_data.get("codex_cmd", "codex exec"), "llm_cost.codex_cmd"
                )
                or "codex exec"
            ),
        ),
        mdm_frames=(
            None
            if data.get("mdm_frames") is None
            else _positive_int(data["mdm_frames"], "mdm_frames")
        ),
        num_denoising_steps=(
            None
            if data.get("num_denoising_steps") is None
            else _positive_int(data["num_denoising_steps"], "num_denoising_steps")
        ),
        text_time=int(data.get("text_time", 0)),
        preference_learning=_bool(
            data.get("preference_learning", True), "preference_learning"
        ),
        preference_alpha=_float(data.get("preference_alpha", 0.5), "preference_alpha"),
        preference_window=_positive_int(
            data.get("preference_window", 50), "preference_window"
        ),
        motion_generator=motion_generator,
    )
