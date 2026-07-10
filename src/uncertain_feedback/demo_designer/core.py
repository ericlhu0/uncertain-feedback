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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.experiments.experiment_pipeline import (
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
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    MpcCostContext,
    build_extra_costs,
)
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SMPL_BONE_PAIRS_22,
    SmplLeftArmFK,
)
from uncertain_feedback.planners.run import _assemble_full_correction_traj
from uncertain_feedback.simulated_users.base import (
    FEATURE_NAMES,
    Bound,
    CoupledBound,
    HiddenBound,
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
        if bound_type == "lower_bound":
            low, high = values.get("threshold"), None
        elif bound_type == "upper_bound":
            low, high = None, values.get("threshold")
        elif bound_type == "band":
            low, high = values.get("low"), values.get("high")
        else:
            continue
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
        self.prompt: str | None = None
        self.labels: np.ndarray | None = None
        self.cluster_means: dict[int, np.ndarray] = {}
        self.cluster_corrections: dict[int, np.ndarray] = {}
        self.cluster_fulls: dict[int, np.ndarray] = {}
        self.chosen_label: int | None = None
        self.scale: float = self.cfg.uq.scale
        self.scaled_correction: np.ndarray | None = None

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
        return [
            persona_to_json(user, builtin=name in self.builtin_names)
            for name, user in sorted(self.personas.items())
        ]

    def upsert_persona(self, data: dict[str, Any]) -> None:
        user = persona_from_json(data)
        self.personas[user.name] = user
        if user.name not in self.builtin_names:
            self._save_custom_personas()

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
        """Return the JSON payload the UI needs to draw one trajectory."""
        traj = np.asarray(traj, dtype=np.float64)
        arm_pos = self.fk.fk_batch(traj, self.spine3_pos, self.spine3_aa)
        feats = feature_series(self.context, traj)
        violations = compute_violations(user, self.context, traj)
        bounds = []
        for bound in user.bounds:
            if isinstance(bound, CoupledBound):
                thr = bound.threshold(feats[bound.cond_feature]).tolist()
                bounds.append(
                    {
                        "feature": bound.feature,
                        "bound_type": bound.bound_type,
                        "low": thr if bound.bound_type == "lower_bound" else None,
                        "high": thr if bound.bound_type == "upper_bound" else None,
                    }
                )
            else:
                bounds.append(
                    {
                        "feature": bound.feature,
                        "bound_type": bound.bound_type,
                        "low": bound.low,
                        "high": bound.high,
                    }
                )
        return {
            "n_frames": int(traj.shape[0]),
            "arm_positions": arm_pos.tolist(),
            "features": {k: v.tolist() for k, v in feats.items()},
            "violations": violations.tolist(),
            "bounds": bounds,
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
                list(self.cfg.cartesian.goals[0]) if self.cfg.cartesian.goals else [0.4, 0.3, 0.1]
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
            "arm_chain_indices": list(LEFT_ARM_CHAIN_INDICES),
            "uq": {
                "diffusion_samples": self.cfg.uq.diffusion_samples,
                "n_clusters": self.cfg.uq.n_clusters,
                "scale": self.cfg.uq.scale,
            },
            "cost_backend": self.cfg.llm_cost.backend,
            "mdm_frames": self.cfg.mdm_frames,
            "steps": self.cfg.steps,
            "trigger_threshold": self.cfg.transfer.trigger_threshold,
            "cartesian_threshold": self.cfg.cartesian.threshold,
        }

    # --- pipeline stages ---------------------------------------------------

    def preview_pose(self, arm_aa: list[list[float]]) -> dict[str, Any]:
        q = np.asarray(arm_aa, dtype=np.float64)
        arm_pos = self.fk.fk(q, self.spine3_pos, self.spine3_aa)
        return {
            "arm_positions": arm_pos.tolist(),
            "wrist_rel": (arm_pos[-1] - self.spine3_pos).tolist(),
        }

    def run_base(
        self, arm_aa: list[list[float]], goal: list[float], persona: str
    ) -> dict[str, Any]:
        user = self.get_persona(persona)
        self.persona_name = persona
        self.start_arm_aa = np.asarray(arm_aa, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        cfg_goal = self._cfg_with_goal(self.goal)
        t0 = time.perf_counter()
        _log(f"base rollout: persona={persona} goal={goal}")
        traj = rollout_to_goal(
            cfg_goal,
            self.start_arm_aa,
            self.goal,
            self.context,
            self._extra_costs(user),
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
            progress_label="base",
            log_prefix=_LOG_PREFIX,
        )
        _log(f"base rollout done in {time.perf_counter() - t0:.1f}s")
        self.base_traj = traj
        self.trigger_step = first_violation_step(
            user, self.context, traj, self.cfg.transfer.trigger_threshold
        )
        if self.trigger_step is not None:
            self.q_feedback = traj[self.trigger_step]
            self.q_history = [
                np.asarray(q, dtype=np.float64)
                for q in traj[: self.trigger_step + 1]
            ]
        else:
            self.q_feedback = None
            self.q_history = []
        # Downstream stages are stale once the base changes.
        self.samples = None
        self.labels = None
        self.cluster_means = {}
        self.cluster_corrections = {}
        self.cluster_fulls = {}
        self.chosen_label = None
        self.scaled_correction = None
        return {
            "trajectory": self.package_trajectory(traj, user),
            "trigger_step": self.trigger_step,
            "metrics": violation_metrics(user, self.context, traj),
            "goal_reach": goal_reach(self.context, cfg_goal, traj, self.goal),
        }

    def generate(
        self, prompt: str, n_samples: int, n_clusters: int, scale: float
    ) -> dict[str, Any]:
        if self.base_traj is None:
            raise ValueError("Run the base rollout first.")
        q_start = (
            self.q_feedback if self.q_feedback is not None else self.start_arm_aa
        )
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
        return self.recluster(n_clusters, scale)

    def recluster(self, n_clusters: int, scale: float) -> dict[str, Any]:
        """Cluster the cached samples and assemble each option into a full path.

        Every cluster option is integrated into the full corrected trajectory
        (executed history → scaled correction → comfort-only goal continuation)
        so the cards show what actually happens if that option is taken, not
        just the raw MDM segment.
        """
        if self.samples is None:
            raise ValueError("Generate MDM samples first.")
        if self.goal is None:
            raise ValueError("Run the base rollout first.")
        user = self.get_persona(self.persona_name)
        cfg_goal = self._cfg_with_goal(self.goal)
        extra = self._extra_costs(user)
        self.labels = XyzPositionClusterer(n_clusters, fk=self.fk).cluster_positions(
            self.samples
        )
        self.cluster_means = {
            label: self.gen.smpl_positions_to_left_arm_trajectory(
                self.samples[self.labels == label].mean(axis=0),
                spine3_aa=self.spine3_aa,
            )
            for label in sorted(int(v) for v in np.unique(self.labels))
        }
        self.chosen_label = None
        self.scaled_correction = None
        self.scale = scale
        self.cluster_corrections = {}
        self.cluster_fulls = {}
        oracle = oracle_cluster_scores(user, self.context, self.cluster_means, scale)
        clusters = []
        for label, mean in self.cluster_means.items():
            scaled = scale_trajectory(mean, scale)
            t0 = time.perf_counter()
            _log(
                f"assembling full corrected path for cluster {label} "
                f"({label + 1}/{len(self.cluster_means)})"
            )
            full = _assemble_full_correction_traj(
                cfg_goal,
                self.q_history,
                scaled,
                self.context,
                extra,
                self.body_pos,
                self.spine3_pos,
                self.spine3_aa,
            )
            _log(f"cluster {label} full path done in {time.perf_counter() - t0:.1f}s")
            self.cluster_corrections[label] = scaled
            self.cluster_fulls[label] = full
            clusters.append(
                {
                    "label": label,
                    "count": int(np.sum(self.labels == label)),
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
        return {"clusters": clusters}

    def pick_cluster(self, label: int) -> dict[str, Any]:
        if label not in self.cluster_corrections:
            raise ValueError(f"Unknown cluster label {label}.")
        self.chosen_label = label
        self.scaled_correction = self.cluster_corrections[label]
        return {"ok": True}

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
        t0 = time.perf_counter()
        _log("rolling out with the generated cost installed")
        rollout = rollout_to_goal(
            cfg_goal,
            self.start_arm_aa,
            self.goal,
            self.context,
            CompositeTrajectoryCost([*extra.terms(), cost]),
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
            progress_label="generated",
            log_prefix=_LOG_PREFIX,
        )
        _log(
            "generated-cost rollout from the original start pose done in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return {
            "trajectory": self.package_trajectory(rollout, user),
            "generated_bounds": _generated_bounds_from_artifacts(cost_dir),
            "metrics": violation_metrics(user, self.context, rollout),
            "goal_reach": goal_reach(self.context, cfg_goal, rollout, self.goal),
            "description": cost.description,
            "code": cost.code,
            "artifact_dir": str(cost_dir),
        }
