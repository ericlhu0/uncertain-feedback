"""Rig state and session management for the demo-runner web tool.

One :class:`DemoRig` per server process holds everything that outlives an
individual session: the loaded config, the persona library (with CRUD), the
motion generator, the initial pose / body / spine, forward kinematics, the mesh
cache, and the shared :class:`MpcCostContext`. A :class:`~uncertain_feedback.demo_runner.session.Session`
is spawned from the rig; it owns one simulated user plus the context accumulated
while correcting them (trajectory corpus, correction rounds, unified cost) and
each trajectory driven under that user.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from itertools import product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from uncertain_feedback.demo_runner.smpl_mesh import SmplMeshCache
from uncertain_feedback.experiments.trajectory_corpus import TrajectoryCorpus
from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.planners.mpc import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.config import (
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
from uncertain_feedback.simulated_users.base import (
    FEATURE_NAMES,
    Bound,
    CoupledBound,
    HiddenBound,
    JOINT_SLOTS,
    SimulatedUser,
    feature_series,
)
from uncertain_feedback.simulated_users.personas import (
    DEFAULT_ARM_JOINT_LIMITS,
    PERSONAS,
)

if TYPE_CHECKING:
    from uncertain_feedback.demo_runner.session import Session

_LOG_PREFIX = "[demo_runner]"
_DEFAULT_PROMPTS = {
    "triceps_long_head_contracture": "Bring my left elbow closer to my body.",
}


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


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


# --- Rig ----------------------------------------------------------------------


class DemoRig:
    """Holds the loaded pose/config and manages the active session."""

    def __init__(
        self,
        config_path: Path,
        personas_path: Path,
        trajectory_configs_path: Path,
    ) -> None:
        self.cfg: MpcRunConfig = load_mpc_config(config_path)
        self.artifact_root = Path("demo_runner_artifacts").resolve()
        self.personas_path = personas_path.resolve()
        self.trajectory_configs_path = trajectory_configs_path.resolve()
        self.trajectory_configs: dict[str, list[dict[str, Any]]] = {
            "initial_poses": [],
            "goals": [],
        }
        self._load_trajectory_configs()
        self.personas: dict[str, SimulatedUser] = dict(PERSONAS)
        self.builtin_names = set(PERSONAS)
        self._load_custom_personas()

        self.gen = make_motion_generator(
            self.cfg.motion_generator,
            None,
            self.cfg.num_denoising_steps,
            seed=self.cfg.seed,
            lock_seed=self.cfg.motion_generator == "mdm",
        )
        if self.cfg.pose is None:
            raise ValueError("demo_runner requires a config with a pose file.")
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

        self.session: "Session | None" = None

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
            np.asarray(list(product(*zip(limit.low, limit.high))), dtype=np.float64)
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
        if self.session is not None and user.name == self.session.persona_name:
            return self.session.update_persona(user)
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

    # --- named trajectory configs ---------------------------------------

    def _load_trajectory_configs(self) -> None:
        if not self.trajectory_configs_path.exists():
            return
        data = json.loads(self.trajectory_configs_path.read_text(encoding="utf-8"))
        self.trajectory_configs = {
            "initial_poses": list(data.get("initial_poses", [])),
            "goals": list(data.get("goals", [])),
        }

    def _save_trajectory_configs(self) -> None:
        self.trajectory_configs_path.write_text(
            json.dumps(self.trajectory_configs, indent=2), encoding="utf-8"
        )

    def trajectory_configs_payload(self) -> dict[str, list[dict[str, Any]]]:
        return self.trajectory_configs

    def upsert_trajectory_config(
        self, kind: str, data: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        name = data["name"].strip()
        if not name:
            raise ValueError("Config name is required.")
        if kind == "initial_poses":
            value = np.asarray(data["arm_aa"], dtype=np.float64)
            if value.shape != (3, 3):
                raise ValueError("Initial pose must contain three axis-angle joints.")
            entry = {"name": name, "arm_aa": value.tolist()}
        elif kind == "goals":
            value = np.asarray(data["goal"], dtype=np.float64)
            if value.shape != (3,):
                raise ValueError("Goal must contain three Cartesian coordinates.")
            entry = {"name": name, "goal": value.tolist()}
        else:
            raise ValueError(f"Unknown trajectory config kind {kind!r}.")

        configs = self.trajectory_configs[kind]
        existing = next(
            (index for index, config in enumerate(configs) if config["name"] == name),
            None,
        )
        if existing is None:
            configs.append(entry)
        else:
            configs[existing] = entry
        configs.sort(key=lambda config: config["name"])
        self._save_trajectory_configs()
        return self.trajectory_configs_payload()

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

    def _manual_planner(
        self,
        start_arm_aa: np.ndarray,
        goal: np.ndarray,
        extra_costs: CompositeTrajectoryCost,
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
            extra_costs=extra_costs,
            seed=self.cfg.seed,
            advance_threshold=self.cfg.advance_threshold,
            max_playback_delta=self.cfg.max_playback_delta,
            trajectory_fraction=self.cfg.trajectory_fraction,
            n_diffusion_samples=self.cfg.uq.diffusion_samples,
            n_clusters=self.cfg.uq.n_clusters,
        )

    def package_trajectory(
        self,
        traj: np.ndarray,
        user: SimulatedUser,
        current_mesh_only: bool = False,
        pin_mesh: bool = False,
    ) -> dict[str, Any]:
        """Return the JSON payload the UI needs to draw one trajectory.

        Feature-bound violations and coupled per-frame thresholds are computed
        client-side from ``features`` + the live persona bounds (so on-graph
        bound edits update instantly); only the joint-box part is baked in,
        since the raw joint angles are not in the payload.

        ``current_mesh_only`` keeps live MPC updates linear in trajectory length;
        terminal payloads still register every frame for scrubber playback.
        ``pin_mesh`` keeps the mesh alive for the trajectory's whole life.
        """
        traj = np.asarray(traj, dtype=np.float64)
        arm_pos = self.fk.fk_batch(traj, self.spine3_pos, self.spine3_aa)
        feats = feature_series(self.context, traj)
        return {
            "n_frames": int(traj.shape[0]),
            "arm_positions": arm_pos.tolist(),
            "mesh_id": self.meshes.register(
                arm_pos[-1:] if current_mesh_only else arm_pos, pin=pin_mesh
            ),
            "features": {k: v.tolist() for k, v in feats.items()},
            "limit_violations": user.limit_violation_series(traj).tolist(),
        }

    def preview_pose(self, arm_aa: list[list[float]]) -> dict[str, Any]:
        q = np.asarray(arm_aa, dtype=np.float64)
        arm_pos = self.fk.fk(q, self.spine3_pos, self.spine3_aa)
        return {
            "arm_positions": arm_pos.tolist(),
            "mesh_vertices": self.meshes.preview(arm_pos).tolist(),
            "wrist_rel": (arm_pos[-1] - self.spine3_pos).tolist(),
        }

    def mesh_vertices(self, mesh_id: str, frame: int | None = None) -> np.ndarray:
        if frame is None:
            return self.meshes.vertices(mesh_id)
        return self.meshes.vertices_at(mesh_id, frame)

    def init_payload(self) -> dict[str, Any]:
        arm_bones = {
            (p, c)
            for p, c in SMPL_BONE_PAIRS_22
            if c in LEFT_ARM_CHAIN_INDICES and c != 9
        }
        return {
            "personas": self.personas_payload(),
            "feature_names": list(FEATURE_NAMES),
            "start_arm_aa": self.default_arm_aa.tolist(),
            "default_goal": (
                list(self.cfg.cartesian.goals[0])
                if self.cfg.cartesian.goals
                else [0.4, 0.3, 0.1]
            ),
            "default_prompts": _DEFAULT_PROMPTS,
            "persona_goals": {
                name: {"cartesian": [list(g) for g in goals.cartesian]}
                for name, goals in self.cfg.persona_goals.items()
            },
            "trajectory_configs": self.trajectory_configs_payload(),
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
                "clusterer": self.cfg.uq.clusterer,
                "scale": self.cfg.uq.scale,
            },
            "cost_backend": self.cfg.llm_cost.backend,
            "default_persona": self.cfg.user,
            "mdm_frames": self.cfg.mdm_frames,
            "steps": self.cfg.steps,
            "trigger_threshold": self.cfg.corrections.trigger_threshold,
            "cartesian_threshold": self.cfg.cartesian.threshold,
            "session": None if self.session is None else self.session.payload(),
        }

    # --- session management -----------------------------------------------

    def begin_session(self, persona: str) -> "Session":
        from uncertain_feedback.demo_runner.session import Session

        user = self.get_persona(persona)
        session_dir = self.artifact_root / (
            f"{time.strftime('%Y%m%d_%H%M%S')}_session_{persona}"
        )
        self.session = Session(
            rig=self,
            persona_name=persona,
            user=user,
            dir=session_dir,
            corpus=TrajectoryCorpus.create(
                session_dir / "trajectory_corpus", self.context
            ),
        )
        self.session._save()
        _log(f"session started: persona={persona} dir={session_dir}")
        return self.session

    def list_sessions(self) -> list[dict[str, Any]]:
        root = getattr(self, "artifact_root", Path("demo_runner_artifacts"))
        sessions: list[dict[str, Any]] = []
        if not root.exists():
            return sessions
        for path in sorted(root.glob("*_session_*"), reverse=True):
            manifest = path / "session.json"
            if not manifest.exists():
                continue
            data = json.loads(manifest.read_text(encoding="utf-8"))
            corpus_manifest = path / "trajectory_corpus" / "manifest.json"
            corpus_entries = (
                json.loads(corpus_manifest.read_text(encoding="utf-8"))
                if corpus_manifest.exists()
                else []
            )
            sessions.append(
                {
                    "name": path.name,
                    "dir": str(path),
                    "persona": data.get("persona"),
                    "started": data.get("started"),
                    "trajectory_count": data.get("trajectory_count", 0),
                    "round_count": len(data.get("rounds", [])),
                    "corpus_count": len(corpus_entries),
                }
            )
        return sessions

    def resume_session(self, dir: str) -> "Session":
        from uncertain_feedback.demo_runner.session import Session

        self.session = Session.load(self, Path(dir))
        _log(f"session resumed: dir={dir} persona={self.session.persona_name}")
        return self.session

    def fork_session(self, name: str) -> "Session":
        """Copy a saved session into a fresh live one and resume the copy.

        Forking (rather than resuming) keeps the recorded original pristine so
        replay still reads it while manual inputs accumulate on the branch. Every
        artifact path lives under the session dir, so a prefix rewrite in
        ``session.json`` is enough to repoint the copied costs' state files.
        """
        from uncertain_feedback.demo_runner.session import Session

        root = self.artifact_root.resolve()
        src = (root / name).resolve()
        if src.parent != root or not (src / "session.json").exists():
            raise ValueError("Unknown session.")
        persona = json.loads((src / "session.json").read_text(encoding="utf-8"))[
            "persona"
        ]
        dst = self.artifact_root / (
            f"{time.strftime('%Y%m%d_%H%M%S')}_session_{persona}"
        )
        shutil.copytree(src, dst)
        manifest = dst / "session.json"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(str(src), str(dst.resolve())),
            encoding="utf-8",
        )
        self.session = Session.load(self, dst)
        _log(f"session forked: src={src} dir={dst} persona={persona}")
        return self.session

    def delete_session(self, name: str) -> dict[str, Any]:
        root = self.artifact_root.resolve()
        session_dir = (root / name).resolve()
        if session_dir.parent != root or not (session_dir / "session.json").exists():
            raise ValueError("Unknown session.")
        active_deleted = (
            self.session is not None
            and self.session.dir.resolve() == session_dir
        )
        if active_deleted:
            self.session = None
        shutil.rmtree(session_dir)
        _log(f"session deleted: dir={session_dir}")
        return {"sessions": self.list_sessions(), "active_deleted": active_deleted}
