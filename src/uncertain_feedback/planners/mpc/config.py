"""YAML configuration for MPC controller settings.

Planner capabilities are selected by the presence of top-level sections, one
per :class:`~uncertain_feedback.planners.mpc.mpc.ArmMPC` module slot:

* ``cartesian:`` — the Cartesian goal space (targets + reach threshold).
* ``feedback:`` — the MDM feedback method, with an optional nested ``uq:``.
* ``constraints:`` — named feasibility constraints (e.g. ``robot_ik:``).
* ``robot_actions:`` — switch the action space to robot joint deltas.

The parsed section dataclasses are passed straight into ``ArmMPC``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from uncertain_feedback.motion_generators.steering import SteeringConfig
from uncertain_feedback.planners.mpc.action_spaces import RobotActionsConfig
from uncertain_feedback.planners.mpc.constraints import CONSTRAINT_BUILDERS
from uncertain_feedback.planners.mpc.costs import available_cost_names
from uncertain_feedback.planners.mpc.feedback import FeedbackConfig
from uncertain_feedback.planners.mpc.goal_spaces import CartesianConfig
from uncertain_feedback.uncertainty.uq_selector import UqConfig

# Keys of the retired name-based planner schema; present only to fail loudly
# so a stale config cannot silently load with different capabilities (e.g. a
# gated real-robot config running unconstrained).
_LEGACY_KEYS = {
    "planner": "select modules with the cartesian:/feedback:/constraints:/"
    "robot_actions: sections",
    "goal_threshold": "removed with the joint-space goal queue",
    "advance_threshold": "removed with the joint-space goal queue",
    "max_playback_delta": "moved to feedback.max_playback_delta",
    "trajectory_fraction": "moved to feedback.trajectory_fraction",
    "mdm_frames": "moved to feedback.frames",
    "text_time": "moved to feedback.text_time",
    "uq": "moved under feedback.uq",
    "max_grasp_ik_residual": "moved to constraints.robot_ik.max_residual",
    "playback_stall_steps": "moved to constraints.robot_ik.playback_stall_steps",
    "max_robot_joint_delta": "moved to robot_actions.max_joint_delta",
    "robot_joint_delta_std": "moved to robot_actions.joint_delta_std",
    "robot_infeasibility_weight": "moved to robot_actions.infeasibility_weight",
    "max_grasp_residual": "moved to robot_actions.max_grasp_residual",
    "grasp_residual_frames": "moved to constraints.robot_ik / robot_actions",
}


@dataclass(frozen=True)
class TransferConfig:
    """Held-out goals + trigger settings for the simulated-user transfer experiment."""

    goals: list[list[float]] = field(default_factory=list)
    trigger_threshold: float = 0.02


@dataclass(frozen=True)
class CorrectionConfig:
    """Repeated within-trajectory feedback trigger settings."""

    trigger_threshold: float = 0.02


@dataclass(frozen=True)
class SimulatedUserConfig:
    """Automated simulated-user episode settings (verbalizer + re-trigger loop)."""

    verbalizer: str = "everyday"
    seed: int = 0
    max_rounds: int = 3
    magnitudes: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    nominal_steps: int = 20


@dataclass(frozen=True)
class PersonaGoals:
    """Per-persona override of the correction goal and transfer goals.

    Each restricted persona needs goals tuned to its own restriction so the
    default plan visually requires a correction (a frozen-shoulder user needs
    high goals it cannot reach compliantly; a flexor-synergy user needs
    high-but-reachable goals that force a straight-vs-bent elbow contrast).
    """

    cartesian: list[list[float]] = field(default_factory=list)
    transfer: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class LlmCostConfig:
    """Which cost-generation backend runs, with what model and artifacts."""

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
    # per-generation artifact run dir is not a git repo. AgentCostGenerator wraps
    # this command in a fail-closed Bubblewrap filesystem namespace, so an inner
    # danger-full-access flag does not expose the repository or simulator oracle.
    codex_cmd: str = "codex exec --skip-git-repo-check"
    reasoning_effort: str | None = None


COST_BACKENDS = {"llm", "turns", "agent"}


@dataclass(frozen=True)
class MpcRunConfig:
    """Everything one MPC run needs: module sections, solver sizes, sub-configs."""

    steps: int
    horizon: int
    n_mpc_samples: int
    max_angle_delta: float
    pose: Path | None
    # Optional (3, 3) [shoulder, elbow, wrist] axis-angle override for the
    # initial left-arm pose (same semantics as the --arm CLI flag, which wins).
    arm: list[list[float]] | None
    costs: dict[str, dict[str, Any]]
    llm_cost: LlmCostConfig
    # ArmMPC module sections; each is enabled by its section's presence.
    cartesian: CartesianConfig | None = None
    feedback: FeedbackConfig | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    robot_actions: RobotActionsConfig | None = None
    preference_learning: bool = True
    preference_alpha: float = 0.5
    preference_window: int = 50
    motion_generator: str = "mdm"
    env: str = "kinematic"
    # Keyword arguments forwarded to the env constructor via make_env
    # (e.g. sim_mannequin's robot_base_offset / robot_max_joint_delta).
    env_params: dict[str, Any] = field(default_factory=dict)
    transfer: TransferConfig = TransferConfig()
    corrections: CorrectionConfig = CorrectionConfig()
    simulated_user: SimulatedUserConfig = SimulatedUserConfig()
    # Simulated-user persona name (see simulated_users.PERSONAS); every run
    # loads this user alongside the pose.
    user: str = "unrestricted"
    # Optional per-persona goal overrides for the transfer experiment, keyed by
    # persona name. When the active persona is present, its goals replace the
    # top-level cartesian/transfer goals (see PersonaGoals).
    persona_goals: dict[str, PersonaGoals] = field(default_factory=dict)
    # Seed for planner-local MPC action sampling.
    seed: int = 0


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


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative.")
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


def _goal_list(value: Any, name: str) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    goals: list[list[float]] = []
    for idx, goal in enumerate(value):
        if not isinstance(goal, list) or len(goal) != 3:
            raise ValueError(f"{name}[{idx}] must be a 3-number list.")
        goals.append([_float(v, f"{name}[{idx}]") for v in goal])
    return goals


def _str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings.")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{idx}] must be a string.")
        out.append(item)
    return out


def _parse_steering(data: dict[str, Any]) -> SteeringConfig:
    steps = data.get("resample_steps")
    return SteeringConfig(
        mode=str(data.get("mode", "off")),
        resample_steps=(
            SteeringConfig.resample_steps
            if steps is None
            else tuple(
                _nonnegative_int(step, "feedback.uq.steering.resample_steps")
                for step in steps
            )
        ),
        temperature=_float(
            data.get("temperature", 0.5), "feedback.uq.steering.temperature"
        ),
        guide_from=_nonnegative_int(
            data.get("guide_from", 10), "feedback.uq.steering.guide_from"
        ),
        guidance_weight=_float(
            data.get("guidance_weight", 1e5), "feedback.uq.steering.guidance_weight"
        ),
    )


def _parse_uq(data: dict[str, Any]) -> UqConfig:
    return UqConfig(
        diffusion_samples=_positive_int(
            data.get("diffusion_samples", 128), "feedback.uq.diffusion_samples"
        ),
        n_clusters=_positive_int(data.get("n_clusters", 3), "feedback.uq.n_clusters"),
        clusterer=str(data.get("clusterer", "agglo_end_pose")),
        auto_cluster=(
            None if data.get("auto_cluster") is None else int(data["auto_cluster"])
        ),
        scale=_float(data.get("scale", 1.0), "feedback.uq.scale"),
        user_cluster=_bool(data.get("user_cluster", False), "feedback.uq.user_cluster"),
        steering=_parse_steering(
            _mapping(data.get("steering", {}), "feedback.uq.steering")
        ),
    )


def _parse_feedback(data: dict[str, Any]) -> FeedbackConfig:
    return FeedbackConfig(
        max_playback_delta=_float(
            data.get("max_playback_delta", 0.05), "feedback.max_playback_delta"
        ),
        trajectory_fraction=_float(
            data.get("trajectory_fraction", 1.0), "feedback.trajectory_fraction"
        ),
        frames=(
            None
            if data.get("frames") is None
            else _positive_int(data["frames"], "feedback.frames")
        ),
        text_time=int(data.get("text_time", 0)),
        uq=(_parse_uq(_mapping(data["uq"], "feedback.uq")) if "uq" in data else None),
    )


def _parse_constraints(data: dict[str, Any]) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for name, params in data.items():
        if name not in CONSTRAINT_BUILDERS:
            raise ValueError(
                f"Unknown constraint '{name}'; choose from "
                f"{sorted(CONSTRAINT_BUILDERS)}."
            )
        cfg_cls = CONSTRAINT_BUILDERS[name][0]
        params_map = _mapping(params, f"constraints.{name}")
        valid = {f.name for f in fields(cfg_cls)}
        unknown = set(params_map) - valid
        if unknown:
            raise ValueError(f"Unknown constraints.{name} keys: {sorted(unknown)}.")
        constraints[name] = cfg_cls(**params_map)
    return constraints


def _parse_robot_actions(data: dict[str, Any]) -> RobotActionsConfig:
    return RobotActionsConfig(
        max_joint_delta=_float(
            data.get("max_joint_delta", 0.005), "robot_actions.max_joint_delta"
        ),
        joint_delta_std=(
            None
            if data.get("joint_delta_std") is None
            else _float(data["joint_delta_std"], "robot_actions.joint_delta_std")
        ),
        infeasibility_weight=_float(
            data.get("infeasibility_weight", 1.0),
            "robot_actions.infeasibility_weight",
        ),
        max_grasp_residual=_float(
            data.get("max_grasp_residual", 0.02), "robot_actions.max_grasp_residual"
        ),
        grasp_residual_frames=_positive_int(
            data.get("grasp_residual_frames", 3),
            "robot_actions.grasp_residual_frames",
        ),
    )


def load_mpc_config(path: Path) -> MpcRunConfig:
    """Load required MPC controller settings from YAML."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    data = _mapping(raw, "MPC config")

    for key in sorted(_LEGACY_KEYS.keys() & data.keys()):
        raise ValueError(f"'{key}' is retired: {_LEGACY_KEYS[key]}.")

    from uncertain_feedback.motion_generators import (  # pylint: disable=import-outside-toplevel
        MOTION_GENERATOR_BUILDERS,
    )

    motion_generator = str(data.get("motion_generator", "mdm"))
    if motion_generator not in MOTION_GENERATOR_BUILDERS:
        raise ValueError(
            "motion_generator must be one of "
            f"{sorted(MOTION_GENERATOR_BUILDERS)}; got {motion_generator!r}."
        )

    from uncertain_feedback.envs import (  # pylint: disable=import-outside-toplevel
        ENV_BUILDERS,
    )

    env = str(data.get("env", "kinematic"))
    if env not in ENV_BUILDERS:
        raise ValueError(f"env must be one of {sorted(ENV_BUILDERS)}; got {env!r}.")
    env_params = _mapping(data.get("env_params"), "env_params")

    cost_data = _mapping(data.get("costs"), "costs")
    llm_cost_data = _mapping(data.get("llm_cost"), "llm_cost")

    cartesian: CartesianConfig | None = None
    if "cartesian" in data:
        cartesian_data = _mapping(data["cartesian"], "cartesian")
        goals = _goal_list(cartesian_data.get("goals", []), "cartesian.goals")
        if not goals:
            raise ValueError("cartesian.goals must be non-empty.")
        cartesian = CartesianConfig(
            goals=goals,
            threshold=_float(
                cartesian_data.get("threshold", 0.05), "cartesian.threshold"
            ),
        )

    feedback: FeedbackConfig | None = None
    if "feedback" in data:
        feedback = _parse_feedback(_mapping(data["feedback"], "feedback"))

    constraints = _parse_constraints(_mapping(data.get("constraints"), "constraints"))

    robot_actions: RobotActionsConfig | None = None
    if "robot_actions" in data:
        robot_actions = _parse_robot_actions(
            _mapping(data["robot_actions"], "robot_actions")
        )

    if constraints and robot_actions is not None:
        raise ValueError(
            "constraints and robot_actions are mutually exclusive; robot "
            "rollouts are feasible by construction."
        )

    arm_data = data.get("arm")
    arm: list[list[float]] | None = None
    if arm_data is not None:
        arm = _goal_list(arm_data, "arm")
        if len(arm) != 3:
            raise ValueError(
                "arm must be a 3x3 list of [shoulder, elbow, wrist] axis-angles."
            )
    transfer_data = _mapping(data.get("transfer"), "transfer")
    corrections_data = _mapping(data.get("corrections"), "corrections")
    transfer_goals = _goal_list(transfer_data.get("goals", []), "transfer.goals")

    persona_goals: dict[str, PersonaGoals] = {}
    for persona, goals in _mapping(data.get("persona_goals"), "persona_goals").items():
        goals_map = _mapping(goals, f"persona_goals.{persona}")
        persona_goals[persona] = PersonaGoals(
            cartesian=_goal_list(
                goals_map.get("cartesian", []), f"persona_goals.{persona}.cartesian"
            ),
            transfer=_goal_list(
                goals_map.get("transfer", []), f"persona_goals.{persona}.transfer"
            ),
        )

    costs: dict[str, dict[str, Any]] = {}
    cost_names = available_cost_names()
    for name, params in cost_data.items():
        if name not in cost_names:
            raise ValueError(f"Unknown MPC cost '{name}'.")
        params_map = _mapping(params, f"costs.{name}")
        costs[name] = dict(params_map)

    trigger_threshold = _float(
        corrections_data.get(
            "trigger_threshold", transfer_data.get("trigger_threshold", 0.02)
        ),
        "corrections.trigger_threshold",
    )
    if trigger_threshold < 0.0:
        raise ValueError("corrections.trigger_threshold must be nonnegative.")

    simulated_user_data = _mapping(data.get("simulated_user"), "simulated_user")
    default_sim_user = SimulatedUserConfig()
    magnitudes_value = simulated_user_data.get(
        "magnitudes", list(default_sim_user.magnitudes)
    )
    if not isinstance(magnitudes_value, list) or not magnitudes_value:
        raise ValueError("simulated_user.magnitudes must be a non-empty list.")
    simulated_user = SimulatedUserConfig(
        verbalizer=str(
            simulated_user_data.get("verbalizer", default_sim_user.verbalizer)
        ),
        seed=_nonnegative_int(
            simulated_user_data.get("seed", default_sim_user.seed),
            "simulated_user.seed",
        ),
        max_rounds=_positive_int(
            simulated_user_data.get("max_rounds", default_sim_user.max_rounds),
            "simulated_user.max_rounds",
        ),
        magnitudes=tuple(
            _float(value, f"simulated_user.magnitudes[{idx}]")
            for idx, value in enumerate(magnitudes_value)
        ),
        nominal_steps=_positive_int(
            simulated_user_data.get("nominal_steps", default_sim_user.nominal_steps),
            "simulated_user.nominal_steps",
        ),
    )

    return MpcRunConfig(
        steps=_positive_int(data.get("steps"), "steps"),
        horizon=_positive_int(data.get("horizon"), "horizon"),
        n_mpc_samples=_positive_int(data.get("n_mpc_samples"), "n_mpc_samples"),
        max_angle_delta=_float(data.get("max_angle_delta"), "max_angle_delta"),
        pose=_optional_path(data.get("pose"), "pose"),
        arm=arm,
        cartesian=cartesian,
        feedback=feedback,
        constraints=constraints,
        robot_actions=robot_actions,
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
        seed=_nonnegative_int(data.get("seed", 0), "seed"),
        preference_learning=_bool(
            data.get("preference_learning", True), "preference_learning"
        ),
        preference_alpha=_float(data.get("preference_alpha", 0.5), "preference_alpha"),
        preference_window=_positive_int(
            data.get("preference_window", 50), "preference_window"
        ),
        motion_generator=motion_generator,
        env=env,
        env_params=env_params,
        user=str(data.get("user", "unrestricted")),
        persona_goals=persona_goals,
        transfer=TransferConfig(
            goals=transfer_goals,
            trigger_threshold=trigger_threshold,
        ),
        corrections=CorrectionConfig(trigger_threshold=trigger_threshold),
        simulated_user=simulated_user,
    )
