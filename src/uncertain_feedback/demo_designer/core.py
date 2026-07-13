"""Session state and pipeline wrappers for the demo-designer web tool.

One :class:`DemoSession` per server process. It wraps the same headless
pipeline the simulated-user experiments use — base Cartesian MPC rollout,
MDM/UQ correction, magnitude scaling, full-correction assembly, and cost
generation — and packages every trajectory into a JSON-ready dict (FK arm
positions, joint-feature series, hidden-bound violations, per-frame bound
thresholds, and generated-cost feature limits) for the browser UI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.demo_designer.smpl_mesh import SmplMeshCache

from uncertain_feedback.experiments.experiment_pipeline import (
    CostGenerationResult,
    generate_cost_for_cluster,
    goal_reach,
    oracle_cluster_scores,
    rollout_to_goal,
)
from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.planners.mpc.config import (
    COST_BACKENDS,
    MpcRunConfig,
    load_mpc_config,
)
from uncertain_feedback.planners.mpc import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.costs import (
    CombineCostGenerator,
    CompositeTrajectoryCost,
    CostRound,
    MpcCostContext,
    build_extra_costs,
    replace_generated_costs,
)
from uncertain_feedback.planners.mpc.costs.generated import GeneratedPythonCost
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SMPL_BONE_PAIRS_22,
    SmplLeftArmFK,
)
from uncertain_feedback.planners.correction_session import (
    CorrectionTrigger,
    TriggerReason,
)
from uncertain_feedback.planners.run import _assemble_full_correction_traj
from uncertain_feedback.simulated_users.base import (
    FEATURE_NAMES,
    Bound,
    CoupledBound,
    HiddenCostTerm,
    HiddenBound,
    JOINT_SLOTS,
    SimulatedUser,
    compute_violations,
    feature_series,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.simulated_users.personas import (
    DEFAULT_ARM_JOINT_LIMITS,
    PERSONAS,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory
from uncertain_feedback.uncertainty.clustering.xyz_clusterer import (
    XyzPositionClusterer,
)

_LOG_PREFIX = "[demo_designer]"


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start : end + 1])
    return data if isinstance(data, dict) else {}


def _generated_bounds_from_artifacts(cost_dir: Path) -> list[dict[str, Any]]:
    ground_response = cost_dir / "ground_response.txt"
    if ground_response.exists():
        payload = _json_object(ground_response.read_text(encoding="utf-8"))
    else:
        stage_log = cost_dir / "stage_log.md"
        if not stage_log.exists():
            return []
        text = stage_log.read_text(encoding="utf-8")
        marker = "## Stage 2 response"
        start = text.find(marker)
        if start < 0:
            return []
        start += len(marker)
        end = text.find("\n## ", start)
        payload = _json_object(text[start:] if end < 0 else text[start:end])

    bounds: list[dict[str, Any]] = []
    for term in payload.get("terms", []):
        values = term.get("values", {})
        bound_type = term.get("bound_type")
        if bound_type not in ("lower_bound", "upper_bound", "band"):
            continue
        # Pose-dependent (coupled) bound: threshold slides with another feature.
        if bound_type != "band" and "cond_feature" in values and values.get("points"):
            bounds.append(
                {
                    "feature": term.get("feature"),
                    "bound_type": bound_type,
                    "coupled": True,
                    "cond_feature": values.get("cond_feature"),
                    "points": [[float(c), float(t)] for c, t in values["points"]],
                }
            )
            continue
        if bound_type == "lower_bound":
            low, high = values.get("threshold"), None
        elif bound_type == "upper_bound":
            low, high = None, values.get("threshold")
        else:
            low, high = values.get("low"), values.get("high")
        bounds.append(
            {
                "feature": term.get("feature"),
                "bound_type": bound_type,
                "low": low,
                "high": high,
            }
        )
    return bounds


# --- Persona (de)serialization ------------------------------------------------


def bound_to_json(bound: Bound) -> dict[str, Any]:
    if isinstance(bound, CoupledBound):
        return {
            "kind": "coupled",
            "feature": bound.feature,
            "bound_type": bound.bound_type,
            "cond_feature": bound.cond_feature,
            "intercept": bound.intercept,
            "slope": bound.slope,
        }
    return {
        "kind": "hidden",
        "feature": bound.feature,
        "bound_type": bound.bound_type,
        "low": bound.low,
        "high": bound.high,
    }


def bound_from_json(data: dict[str, Any]) -> Bound:
    if data["kind"] == "coupled":
        return CoupledBound(
            feature=data["feature"],
            bound_type=data["bound_type"],
            cond_feature=data["cond_feature"],
            intercept=float(data["intercept"]),
            slope=float(data["slope"]),
        )
    return HiddenBound(
        feature=data["feature"],
        bound_type=data["bound_type"],
        low=None if data.get("low") is None else float(data["low"]),
        high=None if data.get("high") is None else float(data["high"]),
    )


def persona_to_json(user: SimulatedUser, builtin: bool) -> dict[str, Any]:
    return {
        "name": user.name,
        "description": user.description,
        "feedback_text": user.feedback_text,
        "bounds": [bound_to_json(b) for b in user.bounds],
        "joint_limits": [
            {"joint": limit.joint, "low": list(limit.low), "high": list(limit.high)}
            for limit in user.joint_limits
        ],
        "builtin": builtin,
    }


def persona_from_json(data: dict[str, Any]) -> SimulatedUser:
    return SimulatedUser(
        name=data["name"],
        description=data.get("description", ""),
        feedback_text=data.get("feedback_text", ""),
        bounds=tuple(bound_from_json(b) for b in data.get("bounds", [])),
        joint_limits=DEFAULT_ARM_JOINT_LIMITS,
    )


# --- Session -------------------------------------------------------------------


@dataclass
class ManualTrajectorySession:
    mpc: LeftArmMPCCartesian
    trigger: CorrectionTrigger
    q: np.ndarray
    executed: list[np.ndarray]
    artifact_dir: Path
    step: int = 0
    paused: bool = False
    complete: bool = False
    reached_goal: bool = False
    error: str | None = None


@dataclass
class ClusterLevel:
    """One recursively clustered subset of the generated MDM samples."""

    sample_indices: np.ndarray
    labels: np.ndarray
    selected_label: int | None
    n_clusters: int
    scale: float



class DemoSession:
    """Holds the loaded pose/config and the results of each pipeline stage."""

    def __init__(self, config_path: Path, personas_path: Path) -> None:
        self.cfg: MpcRunConfig = load_mpc_config(config_path)
        self.personas_path = personas_path
        self.personas: dict[str, SimulatedUser] = dict(PERSONAS)
        self.builtin_names = set(PERSONAS)
        self._load_custom_personas()

        self.gen = make_motion_generator(
            self.cfg.motion_generator, None, self.cfg.num_denoising_steps
        )
        if self.cfg.pose is None:
            raise ValueError("demo_designer requires a config with a pose file.")
        self.initial_hml_pose = self.gen.load_pose(self.cfg.pose)
        arm_aa, body_pos, spine3_aa, collar_aa = self.gen.decode_pose(
            self.initial_hml_pose
        )
        self.default_arm_aa = np.asarray(arm_aa, dtype=np.float64)
        self.body_pos = np.asarray(body_pos, dtype=np.float64)
        self.spine3_pos = np.asarray(body_pos[9], dtype=np.float64)
        self.spine3_aa = np.asarray(spine3_aa, dtype=np.float64)
        self.fk = SmplLeftArmFK()
        self.fk.collar_aa = np.asarray(collar_aa, dtype=np.float64)
        self.meshes = SmplMeshCache(self.body_pos)
        self.context = MpcCostContext(
            fk=self.fk, spine3_pos=self.spine3_pos, spine3_aa=self.spine3_aa
        )

        # Stage state (set by the pipeline endpoints)
        self.persona_name: str = self.cfg.user
        self.start_arm_aa: np.ndarray = self.default_arm_aa.copy()
        self.goal: np.ndarray | None = None
        self.base_traj: np.ndarray | None = None
        self.trigger_step: int | None = None
        self.q_feedback: np.ndarray | None = None
        self.q_history: list[np.ndarray] = []
        self.samples: np.ndarray | None = None  # (N, T, 22, 3)
        self.cluster_levels: list[ClusterLevel] = []
        self.prompt: str | None = None
        self.labels: np.ndarray | None = None
        self.cluster_means: dict[int, np.ndarray] = {}
        self.cluster_corrections: dict[int, np.ndarray] = {}
        self.cluster_fulls: dict[int, np.ndarray] = {}
        self.chosen_label: int | None = None
        self.scale: float = self.cfg.uq.scale
        self.scaled_correction: np.ndarray | None = None
        self.manual_trajectory: ManualTrajectorySession | None = None
        self.trigger_reason: TriggerReason | None = None
        self.trigger_violation: float | None = None

        # Multi-round state (unified cost is unpicklable; lost on restart)
        self.rounds: list[CostRound] = []
        self.round_records: list[dict[str, Any]] = []
        self._round_costs: list[GeneratedPythonCost] = []
        self._round_generations: list[CostGenerationResult] = []
        self.unified_cost: GeneratedPythonCost | None = None
        self._latest_round_generation: CostGenerationResult | None = None
        self._last_generation: CostGenerationResult | None = None
        self._last_cost: GeneratedPythonCost | None = None
        self._last_cost_dir: Path | None = None
        self._last_instruction: str | None = None
        self._last_cost_payload: dict[str, Any] | None = None

    # --- personas ---------------------------------------------------------

    def _load_custom_personas(self) -> None:
        if not self.personas_path.exists():
            return
        with open(self.personas_path, encoding="utf-8") as f:
            for data in json.load(f):
                self.personas[data["name"]] = persona_from_json(data)

    def _save_custom_personas(self) -> None:
        customs = [
            persona_to_json(user, builtin=False)
            for name, user in sorted(self.personas.items())
            if name not in self.builtin_names
        ]
        with open(self.personas_path, "w", encoding="utf-8") as f:
            json.dump(customs, f, indent=2)

    def personas_payload(self) -> list[dict[str, Any]]:
        payload = []
        for name, user in sorted(self.personas.items()):
            data = persona_to_json(user, builtin=name in self.builtin_names)
            data["feature_box_ranges"] = self._feature_box_ranges(user)
            payload.append(data)
        return payload

    def _feature_box_ranges(self, user: SimulatedUser) -> dict[str, list[float]]:
        """Return feature-space ranges induced by the persona's joint boxes."""
        if not user.joint_limits:
            return {}
        sample_count = 8192
        rng = np.random.default_rng(0)
        sampled = np.zeros((sample_count, 3, 3), dtype=np.float64)
        for limit in user.joint_limits:
            sampled[:, JOINT_SLOTS[limit.joint]] = rng.uniform(
                np.asarray(limit.low),
                np.asarray(limit.high),
                size=(sample_count, 3),
            )
        corners = [
            np.asarray(
                list(product(*zip(limit.low, limit.high))), dtype=np.float64
            )
            for limit in user.joint_limits
        ]
        corner_poses = np.zeros((prod(len(values) for values in corners), 3, 3))
        for index, choices in enumerate(product(*corners)):
            for limit, choice in zip(user.joint_limits, choices):
                corner_poses[index, JOINT_SLOTS[limit.joint]] = choice
        poses = np.concatenate((sampled, corner_poses), axis=0)
        features = feature_series(self.context, poses)
        return {
            name: [
                float(np.min(values) - 0.02 * (np.max(values) - np.min(values))),
                float(np.max(values) + 0.02 * (np.max(values) - np.min(values))),
            ]
            for name, values in features.items()
        }

    def upsert_persona(self, data: dict[str, Any]) -> dict[str, Any]:
        user = persona_from_json(data)
        self.personas[user.name] = user
        if user.name not in self.builtin_names:
            self._save_custom_personas()
        if self.base_traj is not None and user.name == self.persona_name:
            if (
                getattr(self, "manual_trajectory", None) is not None
                and self.manual_trajectory.paused
                and self.q_feedback is not None
            ):
                self.trigger_violation = float(
                    compute_violations(
                        user, self.context, self.q_feedback[np.newaxis]
                    )[0]
                )
                return {"retriggered": True, "trigger_step": self.trigger_step}
            self._set_trigger(user, self.base_traj)
            return {"retriggered": True, "trigger_step": self.trigger_step}
        return {"retriggered": False, "trigger_step": None}

    def delete_persona(self, name: str) -> None:
        if name in self.builtin_names:
            raise ValueError(f"Built-in persona {name!r} cannot be deleted.")
        if name not in self.personas:
            raise ValueError(f"Unknown persona {name!r}.")
        del self.personas[name]
        self._save_custom_personas()

    def get_persona(self, name: str) -> SimulatedUser:
        if name not in self.personas:
            raise ValueError(f"Unknown persona {name!r}.")
        return self.personas[name]

    # --- shared helpers ---------------------------------------------------

    def _cfg_with_goal(self, goal: np.ndarray) -> MpcRunConfig:
        return replace(
            self.cfg,
            cartesian=replace(self.cfg.cartesian, goals=[list(map(float, goal))]),
        )

    def _extra_costs(self, user: SimulatedUser) -> CompositeTrajectoryCost:
        extra = build_extra_costs(self.cfg.costs, self.context)
        if user.joint_limits:
            extra = CompositeTrajectoryCost([*extra.terms(), user.limit_cost()])
        return extra

    def package_trajectory(
        self, traj: np.ndarray, user: SimulatedUser
    ) -> dict[str, Any]:
        """Return the JSON payload the UI needs to draw one trajectory.

        Feature-bound violations and coupled per-frame thresholds are computed
        client-side from ``features`` + the live persona bounds (so on-graph
        bound edits update instantly); only the joint-box part is baked in,
        since the raw joint angles are not in the payload.
        """
        traj = np.asarray(traj, dtype=np.float64)
        arm_pos = self.fk.fk_batch(traj, self.spine3_pos, self.spine3_aa)
        feats = feature_series(self.context, traj)
        return {
            "n_frames": int(traj.shape[0]),
            "arm_positions": arm_pos.tolist(),
            "mesh_id": self.meshes.register(arm_pos),
            "features": {k: v.tolist() for k, v in feats.items()},
            "limit_violations": user.limit_violation_series(traj).tolist(),
        }

    def generated_cost_field(self, cost: GeneratedPythonCost) -> dict[str, Any]:
        """Evaluate the compiled generated cost over a cloud of plausible poses.

        Returns per-pose joint-feature values and the cost's penalty there, read
        straight from the compiled cost (no declared-bound parsing), so the UI can
        draw the cost's ACTUAL penalized region in any feature plane for any cost
        shape and any backend. The cloud is the executed/candidate trajectory frames
        plus small joint-angle perturbations to fill the plane around them.
        """
        frames = [
            np.asarray(f, dtype=np.float64).reshape(-1, 3, 3)
            for f in [
                self.base_traj,
                *self.cluster_fulls.values(),
                *self.cluster_corrections.values(),
            ]
            if f is not None
        ]
        anchors = np.concatenate(frames, axis=0)
        rng = np.random.default_rng(0)
        perturbed = anchors[None] + rng.normal(0.0, 0.25, size=(8, *anchors.shape))
        poses = np.concatenate([anchors[None], perturbed], axis=0).reshape(-1, 3, 3)
        if poses.shape[0] > 1500:
            poses = poses[rng.choice(poses.shape[0], 1500, replace=False)]
        feats = feature_series(self.context, poses)
        batch = np.repeat(poses[:, None, :, :], 2, axis=1)
        try:
            penalty = np.asarray(cost(batch), dtype=np.float64)
        except Exception:  # pragma: no cover - viz must not sink an expensive cost
            return {"features": {}, "penalty": []}
        return {
            "features": {k: v.tolist() for k, v in feats.items()},
            "penalty": penalty.tolist(),
        }

    def init_payload(self) -> dict[str, Any]:
        arm_bones = {
            (p, c)
            for p, c in SMPL_BONE_PAIRS_22
            if c in LEFT_ARM_CHAIN_INDICES and c != 9
        }
        return {
            "personas": self.personas_payload(),
            "current_persona": self.persona_name,
            "feature_names": list(FEATURE_NAMES),
            "start_arm_aa": self.start_arm_aa.tolist(),
            "default_goal": (
                list(self.cfg.cartesian.goals[0])
                if self.cfg.cartesian.goals
                else [0.4, 0.3, 0.1]
            ),
            "persona_goals": {
                name: {"cartesian": [list(g) for g in goals.cartesian]}
                for name, goals in self.cfg.persona_goals.items()
            },
            "body_pos": self.body_pos.tolist(),
            "spine3_pos": self.spine3_pos.tolist(),
            "bone_pairs": [
                [int(p), int(c)]
                for p, c in SMPL_BONE_PAIRS_22
                if (p, c) not in arm_bones
            ],
            "smpl_faces": self.meshes.faces.tolist(),
            "smpl_reference_vertices": self.meshes.preview(
                self.fk.fk(self.default_arm_aa, self.spine3_pos, self.spine3_aa)
            ).tolist(),
            "arm_chain_indices": list(LEFT_ARM_CHAIN_INDICES),
            "uq": {
                "diffusion_samples": self.cfg.uq.diffusion_samples,
                "n_clusters": self.cfg.uq.n_clusters,
                "scale": self.cfg.uq.scale,
            },
            "cost_backend": self.cfg.llm_cost.backend,
            "rounds": self.round_records,
            "unified": self._unified_record(),
            "mdm_frames": self.cfg.mdm_frames,
            "steps": self.cfg.steps,
            "trigger_threshold": self.cfg.corrections.trigger_threshold,
            "cartesian_threshold": self.cfg.cartesian.threshold,
            "manual_trajectory": (
                None
                if self.manual_trajectory is None
                else self._manual_trajectory_payload()
            ),
            "pending_cost": self._last_cost_payload,
        }

    # --- pipeline stages ---------------------------------------------------

    def preview_pose(self, arm_aa: list[list[float]]) -> dict[str, Any]:
        q = np.asarray(arm_aa, dtype=np.float64)
        arm_pos = self.fk.fk(q, self.spine3_pos, self.spine3_aa)
        return {
            "arm_positions": arm_pos.tolist(),
            "mesh_vertices": self.meshes.preview(arm_pos).tolist(),
            "wrist_rel": (arm_pos[-1] - self.spine3_pos).tolist(),
        }

    def mesh_vertices(self, mesh_id: str) -> np.ndarray:
        return self.meshes.vertices(mesh_id)

    def _set_trigger(self, user: SimulatedUser, traj: np.ndarray) -> None:
        self.trigger_step = first_violation_step(
            user, self.context, traj, self.cfg.corrections.trigger_threshold
        )
        if self.trigger_step is not None:
            self.q_feedback = traj[self.trigger_step]
            self.q_history = [
                np.asarray(q, dtype=np.float64) for q in traj[: self.trigger_step + 1]
            ]
        else:
            self.q_feedback = None
            self.q_history = []

    def run_base(
        self,
        arm_aa: list[list[float]],
        goal: list[float],
        persona: str,
        show_oracle: bool = False,
    ) -> dict[str, Any]:
        self.manual_trajectory = None
        self.trigger_reason = None
        self.trigger_violation = None
        user = self.get_persona(persona)
        self.persona_name = persona
        self.start_arm_aa = np.asarray(arm_aa, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        cfg_goal = self._cfg_with_goal(self.goal)
        base_costs = self._extra_costs(user)
        t0 = time.perf_counter()
        _log(f"base rollout: persona={persona} goal={goal}")
        traj = rollout_to_goal(
            cfg_goal,
            self.start_arm_aa,
            self.goal,
            self.context,
            base_costs,
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
            progress_label="base",
            log_prefix=_LOG_PREFIX,
        )
        _log(f"base rollout done in {time.perf_counter() - t0:.1f}s")
        self.base_traj = traj
        self._set_trigger(user, traj)
        oracle_traj = None
        if show_oracle:
            oracle_costs = CompositeTrajectoryCost(
                [*base_costs.terms(), HiddenCostTerm(user, self.context)]
            )
            oracle_start = (
                self.q_feedback if self.q_feedback is not None else self.start_arm_aa
            )
            oracle_mode = (
                "resuming from the feedback pose"
                if self.q_feedback is not None
                else "starting from the initial pose"
            )
            _log(f"oracle rollout: {oracle_mode}")
            oracle_traj = rollout_to_goal(
                cfg_goal,
                oracle_start,
                self.goal,
                self.context,
                oracle_costs,
                self.body_pos,
                self.spine3_pos,
                self.spine3_aa,
                progress_label="oracle",
                log_prefix=_LOG_PREFIX,
            )
            if self.q_history:
                oracle_traj = np.concatenate(
                    [np.asarray(self.q_history[:-1]), oracle_traj], axis=0
                )
        # Downstream stages are stale once the base changes.
        self.samples = None
        self.cluster_levels = []
        self.labels = None
        self.cluster_means = {}
        self.cluster_corrections = {}
        self.cluster_fulls = {}
        self.chosen_label = None
        self.scaled_correction = None
        self._last_generation = None
        self._last_cost = None
        self._last_cost_dir = None
        self._last_instruction = None
        self._last_cost_payload = None
        oracle_payload = None
        if oracle_traj is not None:
            oracle_payload = {
                "trajectory": self.package_trajectory(oracle_traj, user),
                "metrics": violation_metrics(user, self.context, oracle_traj),
                "goal_reach": goal_reach(
                    self.context, cfg_goal, oracle_traj, self.goal
                ),
            }
        return {
            "trajectory": self.package_trajectory(traj, user),
            "oracle": oracle_payload,
            "trigger_step": self.trigger_step,
            "metrics": violation_metrics(user, self.context, traj),
            "goal_reach": goal_reach(self.context, cfg_goal, traj, self.goal),
        }

    def _manual_planner(
        self, start_arm_aa: np.ndarray, goal: np.ndarray, user: SimulatedUser
    ) -> LeftArmMPCCartesian:
        return LeftArmMPCCartesian(
            cartesian_goals=[goal.copy()],
            initial_arm_aa=start_arm_aa,
            cartesian_threshold=self.cfg.cartesian.threshold,
            horizon=self.cfg.horizon,
            n_mpc_samples=self.cfg.n_mpc_samples,
            max_angle_delta=self.cfg.max_angle_delta,
            goal_threshold=self.cfg.goal_threshold,
            visualize=False,
            fk=self.fk,
            spine3_pos=self.spine3_pos,
            spine3_aa=self.spine3_aa,
            body_pos=self.body_pos,
            extra_costs=self._extra_costs(user),
            seed=self.cfg.seed,
            advance_threshold=self.cfg.advance_threshold,
            max_playback_delta=self.cfg.max_playback_delta,
            trajectory_fraction=self.cfg.trajectory_fraction,
            n_diffusion_samples=self.cfg.uq.diffusion_samples,
            n_clusters=self.cfg.uq.n_clusters,
        )

    def _clear_pending_feedback(self) -> None:
        self.samples = None
        self.cluster_levels = []
        self.labels = None
        self.cluster_means = {}
        self.cluster_corrections = {}
        self.cluster_fulls = {}
        self.chosen_label = None
        self.scaled_correction = None
        self._last_generation = None
        self._last_cost = None
        self._last_cost_dir = None
        self._last_instruction = None
        self._last_cost_payload = None

    def start_manual_trajectory(
        self,
        arm_aa: list[list[float]],
        goal: list[float],
        persona: str,
    ) -> dict[str, Any]:
        """Start one trajectory and pause before each manually handled correction."""
        if self.cfg.planner != "arm_mpc_cartesian":
            raise ValueError(
                "Multi-turn trajectories require planner: arm_mpc_cartesian."
            )
        user = self.get_persona(persona)
        self.persona_name = persona
        self.start_arm_aa = np.asarray(arm_aa, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        artifact_dir = (
            Path("demo_designer_artifacts")
            / f"{time.strftime('%Y%m%d_%H%M%S')}_trajectory"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.manual_trajectory = ManualTrajectorySession(
            mpc=self._manual_planner(self.start_arm_aa, self.goal, user),
            trigger=CorrectionTrigger(
                threshold=self.cfg.corrections.trigger_threshold,
                text_time=None,
                automatic=bool(user.bounds and user.feedback_text),
            ),
            q=self.start_arm_aa.copy(),
            executed=[self.start_arm_aa.copy()],
            artifact_dir=artifact_dir,
        )
        self.rounds = []
        self.round_records = []
        self._round_costs = []
        self._round_generations = []
        self.unified_cost = None
        self._latest_round_generation = None
        self._clear_pending_feedback()
        _log(f"multi-turn trajectory started: persona={persona} goal={goal}")
        self._advance_manual_trajectory()
        return self._manual_trajectory_payload()

    def exit_manual_trajectory(self) -> dict[str, Any]:
        """Discard the active trajectory so a new scenario can be designed."""
        if self.manual_trajectory is None:
            raise ValueError("No trajectory is active.")
        artifact_dir = self.manual_trajectory.artifact_dir
        self.manual_trajectory = None
        self.base_traj = None
        self.goal = None
        self.trigger_step = None
        self.trigger_reason = None
        self.trigger_violation = None
        self.q_feedback = None
        self.q_history = []
        self._clear_pending_feedback()
        self.reset_rounds()
        _log(f"trajectory exited (artifacts retained at {artifact_dir})")
        return {"ok": True, "artifact_dir": str(artifact_dir)}

    def _advance_manual_trajectory(self) -> None:
        state = self.manual_trajectory
        if state is None:
            raise ValueError("Start a multi-turn trajectory first.")
        user = self.get_persona(self.persona_name)
        state.paused = False
        while state.step < self.cfg.steps:
            violation = None
            if state.trigger.automatic:
                violation = float(
                    compute_violations(user, self.context, state.q[np.newaxis])[0]
                )
            reason = state.trigger.evaluate(state.step, violation)
            if reason is not None:
                state.paused = True
                self.trigger_step = state.step
                self.trigger_reason = reason
                self.trigger_violation = violation
                self.q_feedback = state.q.copy()
                self.q_history = [q.copy() for q in state.executed]
                _log(
                    f"trajectory paused at frame {state.step}: "
                    f"{reason} (violation={violation})"
                )
                break
            try:
                state.q = state.mpc.step(state.q)
            except RuntimeError as exc:
                state.error = str(exc)
                state.complete = True
                break
            state.executed.append(state.q.copy())
            state.step += 1
            if state.mpc.mdm_ready_to_terminate and state.mpc.goal_reached(state.q):
                state.reached_goal = True
                state.complete = True
                break
        if state.step >= self.cfg.steps:
            state.complete = True
        self.base_traj = np.asarray(state.executed, dtype=np.float64)
        np.save(state.artifact_dir / "executed_trajectory.npy", self.base_traj)
        if state.complete:
            self.trigger_step = None
            self.trigger_reason = None
            self.trigger_violation = None
            self.q_feedback = None
            self.q_history = [q.copy() for q in state.executed]
            _log(
                f"multi-turn trajectory complete: steps={state.step} "
                f"reached_goal={state.reached_goal}"
            )

    def _manual_trajectory_payload(self) -> dict[str, Any]:
        state = self.manual_trajectory
        if state is None or self.goal is None:
            raise ValueError("Start a multi-turn trajectory first.")
        user = self.get_persona(self.persona_name)
        trajectory = np.asarray(state.executed, dtype=np.float64)
        trigger = (
            None
            if not state.paused
            else {
                "step": self.trigger_step,
                "reason": self.trigger_reason,
                "violation": self.trigger_violation,
            }
        )
        return {
            "trajectory": self.package_trajectory(trajectory, user),
            "metrics": violation_metrics(user, self.context, trajectory),
            "goal_reach": goal_reach(
                self.context, self._cfg_with_goal(self.goal), trajectory, self.goal
            ),
            "status": "complete" if state.complete else "paused",
            "trigger": trigger,
            "step": state.step,
            "step_limit": self.cfg.steps,
            "reached_goal": state.reached_goal,
            "error": state.error,
            "artifact_dir": str(state.artifact_dir),
            "rounds": self.round_records,
            "unified": self._unified_record(),
        }

    def generate(
        self, prompt: str, n_samples: int, n_clusters: int, scale: float
    ) -> dict[str, Any]:
        if self.base_traj is None:
            raise ValueError("Run the base rollout first.")
        q_start = self.q_feedback if self.q_feedback is not None else self.start_arm_aa
        start_pose = self.gen.build_pose_from_arm_aa(self.initial_hml_pose, q_start)
        t0 = time.perf_counter()
        _log(f"MDM generation: {n_samples} samples for {prompt!r}")
        self.samples = self.gen.generate_left_arm_position_samples(
            prompt,
            start_pose=start_pose,
            num_samples=n_samples,
            num_frames=self.cfg.mdm_frames,
        )
        _log(f"MDM generation done in {time.perf_counter() - t0:.1f}s")
        self.prompt = prompt
        self.cluster_levels = []
        return self.recluster(n_clusters, scale)

    def recluster(self, n_clusters: int, scale: float) -> dict[str, Any]:
        """Cluster the current sample subset and assemble each option.

        Every cluster option is integrated into the full corrected trajectory
        (executed history → scaled correction → comfort-only goal continuation)
        so the cards show what actually happens if that option is taken, not
        just the raw MDM segment.
        """
        if self.samples is None:
            raise ValueError("Generate MDM samples first.")
        if self.goal is None:
            raise ValueError("Run the base rollout first.")
        sample_indices = (
            self.cluster_levels[-1].sample_indices
            if self.cluster_levels
            else np.arange(self.samples.shape[0], dtype=np.intp)
        )
        labels = XyzPositionClusterer(n_clusters, fk=self.fk).cluster_positions(
            self.samples[sample_indices]
        )
        level = ClusterLevel(
            sample_indices=sample_indices,
            labels=np.asarray(labels, dtype=np.intp),
            selected_label=None,
            n_clusters=n_clusters,
            scale=scale,
        )
        if self.cluster_levels:
            self.cluster_levels[-1] = level
        else:
            self.cluster_levels.append(level)
        return self._activate_cluster_level()

    def _activate_cluster_level(self) -> dict[str, Any]:
        """Build the current level's trajectories and navigation payload."""
        if self.samples is None or not self.cluster_levels:
            raise ValueError("Generate and cluster MDM samples first.")
        if self.goal is None:
            raise ValueError("Run the base rollout first.")
        user = self.get_persona(self.persona_name)
        cfg_goal = self._cfg_with_goal(self.goal)
        level = self.cluster_levels[-1]
        scale = level.scale
        level_samples = self.samples[level.sample_indices]
        self.labels = level.labels
        self.cluster_means = {
            label: self.gen.smpl_positions_to_left_arm_trajectory(
                level_samples[level.labels == label].mean(axis=0),
                spine3_aa=self.spine3_aa,
            )
            for label in sorted(int(v) for v in np.unique(level.labels))
        }
        self.chosen_label = level.selected_label
        self.scaled_correction = None
        self.scale = scale
        self.cluster_corrections = {}
        self.cluster_fulls = {}
        oracle = oracle_cluster_scores(user, self.context, self.cluster_means, scale)
        clusters = []
        for label, mean in self.cluster_means.items():
            scaled = scale_trajectory(mean, scale)
            full = (
                np.concatenate(
                    [np.asarray(self.q_history, dtype=np.float64), scaled], axis=0
                )
                if self.q_history
                else scaled
            )
            self.cluster_corrections[label] = scaled
            self.cluster_fulls[label] = full
            count = int(np.sum(level.labels == label))
            clusters.append(
                {
                    "label": label,
                    "count": count,
                    "can_refine": level.n_clusters >= 2 and count >= level.n_clusters,
                    "oracle_score": oracle[label],
                    "correction": self.package_trajectory(scaled, user),
                    "full": self.package_trajectory(full, user),
                    "full_segments": {
                        "history": len(self.q_history),
                        "correction": int(scaled.shape[0]),
                    },
                    "full_metrics": violation_metrics(user, self.context, full),
                    "full_goal_reach": goal_reach(
                        self.context, cfg_goal, full, self.goal
                    ),
                }
            )
        if self.chosen_label is not None:
            self.scaled_correction = self.cluster_corrections[self.chosen_label]
        path = [
            int(parent.selected_label)
            for parent in self.cluster_levels[:-1]
            if parent.selected_label is not None
        ]
        return {
            "clusters": clusters,
            "selected_label": self.chosen_label,
            "depth": len(self.cluster_levels) - 1,
            "path": path,
            "can_go_back": len(self.cluster_levels) > 1,
            "active_sample_count": int(level.sample_indices.size),
            "scale": scale,
        }

    def pick_cluster(self, label: int) -> dict[str, Any]:
        if label not in self.cluster_corrections:
            raise ValueError(f"Unknown cluster label {label}.")
        if not self.cluster_levels:
            raise ValueError("Cluster samples first.")
        self.cluster_levels[-1].selected_label = label
        self.chosen_label = label
        self.scaled_correction = self.cluster_corrections[label]
        return {"ok": True}

    def refine_cluster(
        self, label: int, n_clusters: int, scale: float
    ) -> dict[str, Any]:
        """Cluster only the raw samples belonging to the selected option."""
        if self.samples is None or not self.cluster_levels:
            raise ValueError("Generate and cluster MDM samples first.")
        level = self.cluster_levels[-1]
        if label not in self.cluster_corrections:
            raise ValueError(f"Unknown cluster label {label}.")
        selected_indices = level.sample_indices[level.labels == label]
        if n_clusters < 2 or selected_indices.size < n_clusters:
            raise ValueError(
                "Refinement needs at least two clusters and enough selected samples."
            )
        level.selected_label = label
        labels = XyzPositionClusterer(n_clusters, fk=self.fk).cluster_positions(
            self.samples[selected_indices]
        )
        self.cluster_levels.append(
            ClusterLevel(
                sample_indices=selected_indices,
                labels=np.asarray(labels, dtype=np.intp),
                selected_label=None,
                n_clusters=n_clusters,
                scale=scale,
            )
        )
        return self._activate_cluster_level()

    def back_cluster(self) -> dict[str, Any]:
        """Return to the parent cluster level and restore its selection."""
        if len(self.cluster_levels) <= 1:
            raise ValueError("Already at the root cluster level.")
        self.cluster_levels.pop()
        return self._activate_cluster_level()

    def generate_cost(self, backend: str) -> dict[str, Any]:
        if self.scaled_correction is None or self.goal is None:
            raise ValueError("Pick a cluster first.")
        if backend not in COST_BACKENDS:
            raise ValueError(f"Unknown cost-generation backend {backend!r}.")
        user = self.get_persona(self.persona_name)
        cfg_goal = self._cfg_with_goal(self.goal)
        extra = self._extra_costs(user)
        cost_q_start = (
            self.q_feedback if self.q_feedback is not None else self.start_arm_aa
        )
        cost_dir = (
            Path("demo_designer_artifacts")
            / f"{time.strftime('%Y%m%d_%H%M%S')}_{backend}"
        )
        instruction = self.prompt or user.feedback_text
        result = generate_cost_for_cluster(
            mpc=None,
            cfg=cfg_goal,
            instruction=instruction,
            cluster_traj=self.scaled_correction,
            current_q=cost_q_start,
            q_history=self.q_history,
            context=self.context,
            base_extra_costs=extra,
            cost_dir=cost_dir,
            body_pos=self.body_pos,
            spine3_pos=self.spine3_pos,
            spine3_aa=self.spine3_aa,
            candidate_trajs=self.cluster_means,
            highlight_label=self.chosen_label,
            backend=backend,
            install=False,
            log_prefix=_LOG_PREFIX,
        )
        cost = result.generated_cost
        if cost is None:
            raise ValueError(f"Cost generation produced no cost (see {cost_dir}).")
        self._last_generation = result
        self._last_cost = cost
        self._last_cost_dir = cost_dir
        self._last_instruction = instruction
        cost_set = CompositeTrajectoryCost([*extra.terms(), cost])
        t0 = time.perf_counter()
        _log("assembling the chosen corrected path with the generated cost")
        rollout = _assemble_full_correction_traj(
            cfg_goal,
            self.q_history,
            self.scaled_correction,
            self.context,
            cost_set,
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
        )
        _log(
            "generated-cost corrected path assembled in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        t0 = time.perf_counter()
        _log("rolling out the generated cost from the initial pose")
        start_rollout = rollout_to_goal(
            cfg_goal,
            self.start_arm_aa,
            self.goal,
            self.context,
            cost_set,
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
            progress_label="generated-from-start",
            log_prefix=_LOG_PREFIX,
        )
        _log(
            "generated-cost rollout from the initial pose done in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        payload = {
            "trajectory": self.package_trajectory(rollout, user),
            "metrics": violation_metrics(user, self.context, rollout),
            "goal_reach": goal_reach(self.context, cfg_goal, rollout, self.goal),
            "start_trajectory": self.package_trajectory(start_rollout, user),
            "start_metrics": violation_metrics(user, self.context, start_rollout),
            "start_goal_reach": goal_reach(
                self.context, cfg_goal, start_rollout, self.goal
            ),
            "cost_field": self.generated_cost_field(cost),
            "generated_bounds": _generated_bounds_from_artifacts(cost_dir),
            "description": cost.description,
            "code": cost.code,
            "artifact_dir": str(cost_dir),
        }
        self._last_cost_payload = payload
        (cost_dir / "demo_designer_payload.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    # --- multi-round feedback ----------------------------------------------

    def _unified_record(self) -> dict[str, Any] | None:
        if self.unified_cost is None:
            return None
        return {
            "description": self.unified_cost.description,
            "code": self.unified_cost.code,
        }

    def commit_round(self) -> dict[str, Any]:
        if (
            self._last_generation is None
            or self._last_cost is None
            or self._last_cost_dir is None
            or self._last_instruction is None
            or self.goal is None
        ):
            raise ValueError("Generate a cost first.")
        state_path = self._last_cost_dir / "state.pkl"
        self._last_generation.eval_state.save(state_path)
        cost_round = CostRound(
            index=len(self.rounds),
            goal=(float(self.goal[0]), float(self.goal[1]), float(self.goal[2])),
            feedback_text=self._last_instruction,
            trigger_step=self.trigger_step if self.trigger_step is not None else 0,
            round_dir=self._last_cost_dir.resolve(),
            state_path=state_path.resolve(),
            cost_code=self._last_cost.code,
            params=self._last_cost.params,
            summaries=self._last_generation.summaries,
            image_paths=tuple(
                path.resolve() for path in self._last_generation.images.values()
            ),
            trigger_reason=self.trigger_reason or "discomfort",
            trigger_violation=self.trigger_violation,
        )
        self.rounds.append(cost_round)
        self._round_costs.append(self._last_cost)
        self._round_generations.append(self._last_generation)
        self.round_records.append(
            {
                "index": cost_round.index,
                "goal": list(cost_round.goal),
                "feedback_text": cost_round.feedback_text,
                "trigger_step": cost_round.trigger_step,
                "trigger_reason": cost_round.trigger_reason,
                "trigger_violation": cost_round.trigger_violation,
                "description": self._last_cost.description,
                "artifact_dir": str(self._last_cost_dir),
            }
        )
        self.unified_cost = self._last_cost if len(self.rounds) == 1 else None
        self._latest_round_generation = self._last_generation
        self._last_generation = None
        self._last_cost = None
        self._last_cost_dir = None
        self._last_instruction = None
        self._last_cost_payload = None
        _log(f"committed round {cost_round.index} (goal={list(cost_round.goal)})")
        return {"rounds": self.round_records, "unified": self._unified_record()}

    def apply_round_and_continue(self) -> dict[str, Any]:
        """Commit the selected feedback, install it, and resume to the next pause."""
        state = self.manual_trajectory
        if state is None or not state.paused:
            raise ValueError("The multi-turn trajectory is not paused for feedback.")
        if self.scaled_correction is None or self._last_cost is None:
            raise ValueError("Generate a cost for the selected cluster first.")
        correction = self.scaled_correction.copy()
        self.commit_round()
        cutoff = max(1, round(len(correction) * self.cfg.trajectory_fraction))
        correction = correction[:cutoff]
        state.mpc.set_mdm_goal(correction[-1])
        state.mpc.push_trajectory(correction)
        state.mpc.set_extra_costs(
            CompositeTrajectoryCost(
                [
                    *self._extra_costs(self.get_persona(self.persona_name)).terms(),
                    *self._round_costs,
                ]
            )
        )
        state.paused = False
        self._clear_pending_feedback()
        self._advance_manual_trajectory()
        return self._manual_trajectory_payload()

    def ignore_comfort_violation(self) -> dict[str, Any]:
        """Resume without feedback after ignoring the current discomfort event."""
        state = self.manual_trajectory
        if state is None or not state.paused:
            raise ValueError("The multi-turn trajectory is not paused for feedback.")
        if self.trigger_reason != "discomfort":
            raise ValueError("The current feedback trigger is not a comfort violation.")
        _log(f"ignored comfort violation at frame {state.step}")
        state.paused = False
        self._clear_pending_feedback()
        self._advance_manual_trajectory()
        return self._manual_trajectory_payload()

    def combine_rounds(self) -> dict[str, Any]:
        if len(self.rounds) < 2:
            raise ValueError("Commit at least two rounds before combining.")
        generation = self._latest_round_generation
        if generation is None or self.goal is None:
            raise ValueError("No committed round generation available.")
        user = self.get_persona(self.persona_name)
        combine_dir = (
            Path("demo_designer_artifacts")
            / f"{time.strftime('%Y%m%d_%H%M%S')}_combine"
        )
        combinator = CombineCostGenerator(
            context=generation.generated_context,
            instruction=self.rounds[-1].feedback_text,
            summaries=generation.summaries,
            run_dir=combine_dir,
            images=generation.images,
            use_images=self.cfg.llm_cost.use_images,
            model=self.cfg.llm_cost.model,
            strict=self.cfg.llm_cost.strict,
            mpc=None,
            rollout_fn=generation.eval_state.make_rollout_fn(),
            eval_state=generation.eval_state,
            codex_cmd=self.cfg.llm_cost.codex_cmd,
            rounds=self.rounds,
        )
        t0 = time.perf_counter()
        _log(f"combining {len(self.rounds)} rounds (artifacts={combine_dir})")
        combined = combinator.generate(install=False)
        if combined is None:
            raise ValueError(f"Round combination failed (see {combine_dir}).")
        _log(f"combination done in {time.perf_counter() - t0:.1f}s")
        self.unified_cost = combined
        cfg_goal = self._cfg_with_goal(self.goal)
        installed = replace_generated_costs(self._extra_costs(user), combined)
        if getattr(self, "manual_trajectory", None) is not None:
            self.manual_trajectory.mpc.set_extra_costs(installed)
        t0 = time.perf_counter()
        _log("rolling out the unified cost from the initial pose")
        rollout = rollout_to_goal(
            cfg_goal,
            self.start_arm_aa,
            self.goal,
            self.context,
            installed,
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
            progress_label="unified-from-start",
            log_prefix=_LOG_PREFIX,
        )
        _log(f"unified-cost rollout done in {time.perf_counter() - t0:.1f}s")
        scores_path = combine_dir / "scores.json"
        scores = (
            json.loads(scores_path.read_text(encoding="utf-8"))
            if scores_path.exists()
            else None
        )
        return {
            "trajectory": self.package_trajectory(rollout, user),
            "cost_field": self.generated_cost_field(combined),
            "metrics": violation_metrics(user, self.context, rollout),
            "goal_reach": goal_reach(self.context, cfg_goal, rollout, self.goal),
            "description": combined.description,
            "code": combined.code,
            "scores": scores,
            "artifact_dir": str(combine_dir),
            "rounds": self.round_records,
        }

    def reset_rounds(self) -> dict[str, Any]:
        self.rounds = []
        self.round_records = []
        self._round_costs = []
        self._round_generations = []
        self.unified_cost = None
        self._latest_round_generation = None
        self._last_generation = None
        self._last_cost = None
        self._last_cost_dir = None
        self._last_instruction = None
        self._last_cost_payload = None
        if getattr(self, "manual_trajectory", None) is not None:
            self.manual_trajectory.mpc.set_extra_costs(
                self._extra_costs(self.get_persona(self.persona_name))
            )
        _log("multi-round state reset")
        return {"ok": True}
