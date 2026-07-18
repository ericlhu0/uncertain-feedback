"""Session and trajectory objects for the demo-runner web tool.

A :class:`Session` is one simulated user plus the context that accumulates while
correcting them: an on-disk :class:`TrajectoryCorpus`, committed correction
rounds, and one unified cost. A :class:`Trajectory` is spawned from a session and
holds the live MPC stepping state plus every per-trajectory correction-pipeline
scratch buffer; it cannot exist without its session, so the lifetimes are enforced
by structure rather than runtime checks. Sessions persist to ``session.json`` and
are resumable after a server restart (recompiling round/unified costs from stored
code + params + the pickled eval state).

Sessions also record a replay stream under ``<session>/replay/``: one file per
beat, each holding the exact payload the browser received plus a persona
snapshot. That is what lets the demo runner step through a past session without
re-running MDM, whose samples are stochastic and never persisted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from uncertain_feedback.demo_runner.core import _LOG_PREFIX, _log, persona_to_json
from uncertain_feedback.experiments.experiment_pipeline import (
    CostGenerationResult,
    generate_cost_for_cluster,
    goal_reach,
    oracle_cluster_scores,
    rollout_to_goal,
)
from uncertain_feedback.experiments.trajectory_corpus import TrajectoryCorpus
from uncertain_feedback.planners.correction_session import (
    CorrectionTrigger,
    TriggerReason,
)
from uncertain_feedback.planners.mpc import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.costs import (
    CombineCostGenerator,
    CompositeTrajectoryCost,
    CostRound,
    generated_cost_feature_dependencies,
    replace_generated_costs,
)
from uncertain_feedback.planners.mpc.costs.cost_feedback import EvalState
from uncertain_feedback.planners.mpc.costs.generated import GeneratedPythonCost
from uncertain_feedback.planners.run import _assemble_full_correction_traj
from uncertain_feedback.simulated_users.base import (
    FEATURE_NAMES,
    HiddenCostTerm,
    SimulatedUser,
    compute_violations,
    feature_series,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory
from uncertain_feedback.uncertainty.clustering import make_clusterer

if TYPE_CHECKING:
    from uncertain_feedback.demo_runner.core import DemoRig


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
        if start < 0 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _ground_specification_from_artifacts(cost_dir: Path) -> dict[str, Any]:
    rationale = cost_dir / "rationale.json"
    if rationale.exists():
        payload = _json_object(rationale.read_text(encoding="utf-8"))
        ground = payload.get("ground")
        if isinstance(ground, dict):
            return ground

    ground_response = cost_dir / "ground_response.txt"
    if ground_response.exists():
        return _json_object(ground_response.read_text(encoding="utf-8"))

    stage_log = cost_dir / "stage_log.md"
    if not stage_log.exists():
        return {}
    text = stage_log.read_text(encoding="utf-8")
    marker = "## Stage 2 response"
    start = text.find(marker)
    if start < 0:
        return {}
    start += len(marker)
    end = text.find("\n## ", start)
    return _json_object(text[start:] if end < 0 else text[start:end])


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _generated_bounds_from_artifacts(cost_dir: Path) -> list[dict[str, Any]]:
    payload = _ground_specification_from_artifacts(cost_dir)
    terms = payload.get("terms")
    if not isinstance(terms, list):
        return []

    bounds: list[dict[str, Any]] = []
    for term in terms:
        if not isinstance(term, dict):
            continue
        feature = term.get("feature")
        values = term.get("values", {})
        bound_type = term.get("bound_type")
        if (
            feature not in FEATURE_NAMES
            or bound_type not in ("lower_bound", "upper_bound", "band")
            or not isinstance(values, dict)
        ):
            continue
        cond_feature = values.get("cond_feature")
        raw_points = values.get("points")
        if (
            bound_type != "band"
            and cond_feature in FEATURE_NAMES
            and cond_feature != feature
            and isinstance(raw_points, list)
        ):
            points: list[list[float]] = []
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    points = []
                    break
                cond = _finite_float(point[0])
                threshold = _finite_float(point[1])
                if cond is None or threshold is None:
                    points = []
                    break
                points.append([cond, threshold])
            if len(points) < 2 or any(
                right[0] <= left[0] for left, right in zip(points, points[1:])
            ):
                continue
            bounds.append(
                {
                    "feature": feature,
                    "bound_type": bound_type,
                    "coupled": True,
                    "cond_feature": cond_feature,
                    "points": points,
                }
            )
            continue
        if bound_type == "lower_bound":
            low, high = _finite_float(values.get("threshold")), None
        elif bound_type == "upper_bound":
            low, high = None, _finite_float(values.get("threshold"))
        else:
            low = _finite_float(values.get("low"))
            high = _finite_float(values.get("high"))
        if (bound_type == "lower_bound" and low is None) or (
            bound_type == "upper_bound" and high is None
        ):
            continue
        if bound_type == "band" and (low is None or high is None or low > high):
            continue
        bounds.append(
            {
                "feature": feature,
                "bound_type": bound_type,
                "low": low,
                "high": high,
            }
        )
    return bounds


def _rationale_from_artifacts(cost_dir: Path) -> dict[str, Any] | None:
    """Read the structured cost rationale when the backend produced one."""
    path = cost_dir / "rationale.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class ClusterLevel:
    """One recursively clustered subset of the generated MDM samples."""

    sample_indices: np.ndarray
    labels: np.ndarray
    selected_label: int | None
    n_clusters: int
    scale: float
    medoid_indices: dict[int, int] = field(default_factory=dict)
    undesirable_labels: set[int] = field(default_factory=set)
    replay_payload: dict[str, Any] | None = None


@dataclass
class Trajectory:
    """Live MPC stepping state plus all per-trajectory pipeline scratch.

    Spawned only by :meth:`Session.start_trajectory`. Correction scratch (MDM
    samples, cluster levels, pending cost) is reset in place between corrections
    and dies with the object.
    """

    mpc: LeftArmMPCCartesian
    trigger: CorrectionTrigger
    q: np.ndarray
    executed: list[np.ndarray]
    artifact_dir: Path
    start_arm_aa: np.ndarray
    goal: np.ndarray
    scale: float
    step: int = 0
    paused: bool = False
    complete: bool = False
    reached_goal: bool = False
    error: str | None = None
    logged_frames: int = 0
    base_traj: np.ndarray | None = None
    trigger_step: int | None = None
    trigger_reason: TriggerReason | None = None
    trigger_violation: float | None = None
    q_feedback: np.ndarray | None = None
    q_history: list[np.ndarray] = field(default_factory=list)
    oracle_traj: np.ndarray | None = None
    oracle_source: str | None = None
    oracle_package: dict[str, Any] | None = None
    clean_traj: np.ndarray | None = None
    clean_package: dict[str, Any] | None = None
    samples: np.ndarray | None = None  # (N, T, 22, 3)
    cluster_levels: list[ClusterLevel] = field(default_factory=list)
    prompt: str | None = None
    labels: np.ndarray | None = None
    cluster_means: dict[int, np.ndarray] = field(default_factory=dict)
    cluster_corrections: dict[int, np.ndarray] = field(default_factory=dict)
    cluster_fulls: dict[int, np.ndarray] = field(default_factory=dict)
    chosen_label: int | None = None
    scaled_correction: np.ndarray | None = None
    _last_generation: CostGenerationResult | None = None
    _last_cost: GeneratedPythonCost | None = None
    _last_cost_dir: Path | None = None
    _last_instruction: str | None = None
    _last_cost_payload: dict[str, Any] | None = None

    def clear_pending_feedback(self) -> None:
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

    def advance(self, session: "Session", max_steps: int | None = None) -> None:
        """Step the planner to the next pause and log the executed segment."""
        rig = session.rig
        user = session.user
        self.paused = False
        stop_step = (
            rig.cfg.steps
            if max_steps is None
            else min(rig.cfg.steps, self.step + max_steps)
        )
        while self.step < stop_step:
            violation = None
            if self.trigger.automatic:
                violation = float(
                    compute_violations(user, rig.context, self.q[np.newaxis])[0]
                )
            reason = self.trigger.evaluate(self.step, violation)
            if reason is not None:
                self.paused = True
                self.trigger_step = self.step
                self.trigger_reason = reason
                self.trigger_violation = violation
                self.q_feedback = self.q.copy()
                self.q_history = [q.copy() for q in self.executed]
                _log(
                    f"trajectory paused at frame {self.step}: "
                    f"{reason} (violation={violation})"
                )
                break
            try:
                self.q = self.mpc.step(self.q)
            except RuntimeError as exc:
                self.error = str(exc)
                self.complete = True
                break
            self.executed.append(self.q.copy())
            self.step += 1
            if self.mpc.mdm_tracking_complete and self.mpc.goal_reached(self.q):
                self.reached_goal = True
                self.complete = True
                break
        if self.step >= rig.cfg.steps:
            self.complete = True
        self.base_traj = np.asarray(self.executed, dtype=np.float64)
        np.save(self.artifact_dir / "executed_trajectory.npy", self.base_traj)
        if self.complete:
            self.trigger_step = None
            self.trigger_reason = None
            self.trigger_violation = None
            self.q_feedback = None
            self.q_history = [q.copy() for q in self.executed]
            _log(
                f"multi-turn trajectory complete: steps={self.step} "
                f"reached_goal={self.reached_goal}"
            )
        if self.trigger.automatic and (self.paused or self.complete):
            segment = self.base_traj[self.logged_frames :]
            if segment.shape[0]:
                discomfort = self.paused and self.trigger_reason == "discomfort"
                session.corpus.log(
                    segment,
                    kind="executed_segment",
                    round_index=len(session.rounds),
                    goal=tuple(map(float, self.goal)),
                    trigger_step=segment.shape[0] - 1 if discomfort else None,
                    trigger_violation=self.trigger_violation if discomfort else None,
                    feedback_text=user.feedback_text if discomfort else None,
                )
                self.logged_frames = len(self.executed)


class Session:
    """One simulated user and the context accumulated while correcting them."""

    def __init__(
        self,
        rig: "DemoRig",
        persona_name: str,
        user: SimulatedUser,
        dir: Path,
        corpus: TrajectoryCorpus,
        started: str | None = None,
    ) -> None:
        self.rig = rig
        self.persona_name = persona_name
        self.user = user
        self.dir = dir
        self.corpus = corpus
        self.started = started or time.strftime("%Y-%m-%d %H:%M:%S")
        self.rounds: list[CostRound] = []
        self.round_records: list[dict[str, Any]] = []
        self._round_costs: list[GeneratedPythonCost] = []
        self.unified_cost: GeneratedPythonCost | None = None
        self.combined_corpus_count: int = 0
        self.trajectory: Trajectory | None = None
        self.trajectory_count: int = 0
        self.beats: list[dict[str, Any]] = []

    # --- replay recording --------------------------------------------------

    def _record(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        """Append one replay beat and return ``data`` unchanged.

        The persona is snapshotted per beat, not per session: the client derives
        every violation and graph bound from it, so a later persona edit would
        otherwise silently redraw a past demo.
        """
        replay_dir = self.dir / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "kind": kind,
            "time": time.strftime("%H:%M:%S"),
            "file": f"{len(self.beats):04d}_{kind}.json",
        }
        (replay_dir / entry["file"]).write_text(
            json.dumps(
                {
                    **entry,
                    "persona": persona_to_json(self.user, builtin=False),
                    "data": data,
                }
            ),
            encoding="utf-8",
        )
        self.beats.append(entry)
        (replay_dir / "index.json").write_text(
            json.dumps(
                {
                    "persona": self.persona_name,
                    "started": self.started,
                    "beats": self.beats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return data

    def _record_final_feedback(self) -> None:
        """Record only the selection path and cost that were accepted."""
        traj = self.trajectory
        if traj is None:
            return
        for level in traj.cluster_levels:
            if level.replay_payload is None or level.selected_label is None:
                continue
            payload = {
                **level.replay_payload,
                "selected_label": None,
                "undesirable_labels": sorted(level.undesirable_labels),
            }
            self._record("clusters", payload)
            self._record("pick", {"label": int(level.selected_label)})
        if traj._last_cost_payload is not None:
            self._record("cost", traj._last_cost_payload)

    # --- persistence -------------------------------------------------------

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "persona": self.persona_name,
            "started": self.started,
            "trajectory_count": self.trajectory_count,
            "combined_corpus_count": self.combined_corpus_count,
            "rounds": [round_.to_json() for round_ in self.rounds],
            "unified": (
                None
                if self.unified_cost is None
                else {
                    "code": self.unified_cost.code,
                    "params": self.unified_cost.params,
                    "description": self.unified_cost.description,
                }
            ),
        }
        (self.dir / "session.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, rig: "DemoRig", dir: Path) -> "Session":
        """Rebuild a session from ``session.json``, recompiling every cost.

        The live trajectory is not restored (MPC state is not persisted): a
        resumed session starts with no active trajectory but full context.
        """
        dir = Path(dir)
        data = json.loads((dir / "session.json").read_text(encoding="utf-8"))
        persona_name = str(data["persona"])
        session = cls(
            rig=rig,
            persona_name=persona_name,
            user=rig.get_persona(persona_name),
            dir=dir,
            corpus=TrajectoryCorpus.create(dir / "trajectory_corpus", rig.context),
            started=data.get("started"),
        )
        session.trajectory_count = int(data.get("trajectory_count", 0))
        session.combined_corpus_count = int(data.get("combined_corpus_count", 0))
        replay_index = dir / "replay" / "index.json"
        if replay_index.exists():
            session.beats = json.loads(replay_index.read_text(encoding="utf-8"))[
                "beats"
            ]
        session.rounds = [CostRound.from_json(r) for r in data.get("rounds", [])]
        for round_ in session.rounds:
            context = EvalState.load(round_.state_path).make_generated_context()
            session._round_costs.append(
                GeneratedPythonCost(
                    round_.cost_code, round_.params, context, round_.description
                )
            )
        session.round_records = [
            session._round_record(round_) for round_ in session.rounds
        ]
        unified = data.get("unified")
        if unified is not None and session.rounds:
            context = EvalState.load(
                session.rounds[-1].state_path
            ).make_generated_context()
            session.unified_cost = GeneratedPythonCost(
                unified["code"],
                dict(unified["params"]),
                context,
                unified.get("description", ""),
            )
        return session

    # --- persona ----------------------------------------------------------

    def update_persona(self, user: SimulatedUser) -> dict[str, Any]:
        """Adopt an edited persona and re-evaluate the active trigger."""
        self.user = user
        traj = self.trajectory
        if traj is None or traj.base_traj is None:
            return {"retriggered": False, "trigger_step": None}
        if traj.paused and traj.q_feedback is not None:
            traj.trigger_violation = float(
                compute_violations(user, self.rig.context, traj.q_feedback[np.newaxis])[
                    0
                ]
            )
            return {"retriggered": True, "trigger_step": traj.trigger_step}
        self._set_trigger(traj.base_traj)
        return {"retriggered": True, "trigger_step": traj.trigger_step}

    def _set_trigger(self, traj_arr: np.ndarray) -> None:
        traj = self.trajectory
        assert traj is not None
        traj.trigger_step = first_violation_step(
            self.user,
            self.rig.context,
            traj_arr,
            self.rig.cfg.corrections.trigger_threshold,
        )
        if traj.trigger_step is not None:
            traj.q_feedback = traj_arr[traj.trigger_step]
            traj.q_history = [
                np.asarray(q, dtype=np.float64)
                for q in traj_arr[: traj.trigger_step + 1]
            ]
        else:
            traj.q_feedback = None
            traj.q_history = []

    # --- trajectory lifetime ----------------------------------------------

    def start_trajectory(
        self,
        arm_aa: list[list[float]],
        goal: list[float],
        advance: bool = True,
    ) -> Trajectory:
        """Start one trajectory with the session costs installed from frame 0."""
        rig = self.rig
        if rig.cfg.planner != "arm_mpc_cartesian":
            raise ValueError(
                "Multi-turn trajectories require planner: arm_mpc_cartesian."
            )
        start_arm_aa = np.asarray(arm_aa, dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)
        artifact_dir = self.dir / f"{time.strftime('%Y%m%d_%H%M%S')}_trajectory"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        base_terms = rig._extra_costs(self.user)
        if self.unified_cost is not None:
            extra = replace_generated_costs(base_terms, self.unified_cost)
        else:
            extra = CompositeTrajectoryCost([*base_terms.terms(), *self._round_costs])
        self._release_pinned_meshes()
        self.trajectory = Trajectory(
            mpc=rig._manual_planner(start_arm_aa, goal_arr, extra),
            trigger=CorrectionTrigger(
                threshold=rig.cfg.corrections.trigger_threshold,
                text_time=None,
                automatic=bool(self.user.bounds and self.user.feedback_text),
            ),
            q=start_arm_aa.copy(),
            executed=[start_arm_aa.copy()],
            artifact_dir=artifact_dir,
            start_arm_aa=start_arm_aa,
            goal=goal_arr,
            scale=rig.cfg.uq.scale,
        )
        self.trajectory_count += 1
        self._save()
        _log(f"trajectory started: persona={self.persona_name} goal={goal}")
        if advance:
            self.trajectory.advance(self)
        self.run_oracle(from_trigger=False)
        self.run_clean_baseline()
        if advance:
            self._record("trajectory", self._trajectory_payload())
        return self.trajectory

    def advance_trajectory(
        self, max_steps: int = 1, current_mesh_only: bool = False
    ) -> dict[str, Any]:
        """Advance a live trajectory incrementally for browser animation."""
        traj = self.trajectory
        if traj is None or traj.paused or traj.complete:
            raise ValueError("The trajectory is not running.")
        traj.advance(self, max_steps=max_steps)
        payload = self._trajectory_payload(
            current_mesh_only=current_mesh_only and not (traj.paused or traj.complete)
        )
        if traj.paused or traj.complete:
            return self._record("trajectory", payload)
        return payload

    def exit_trajectory(self) -> dict[str, Any]:
        """Discard the active trajectory; rounds, costs, and corpus stay."""
        if self.trajectory is None:
            raise ValueError("No trajectory is active.")
        artifact_dir = self.trajectory.artifact_dir
        self._release_pinned_meshes()
        self.trajectory = None
        _log(f"trajectory exited (artifacts retained at {artifact_dir})")
        return {"ok": True, "artifact_dir": str(artifact_dir)}

    # --- oracle -----------------------------------------------------------

    def run_oracle(self, from_trigger: bool) -> dict[str, Any]:
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.goal is None:
            raise ValueError("Start a trajectory first.")
        if from_trigger and traj.q_feedback is None:
            raise ValueError("The trajectory is not paused at an MDM trigger point.")
        cfg_goal = rig._cfg_with_goal(traj.goal)
        oracle_costs = CompositeTrajectoryCost(
            [
                *rig._extra_costs(self.user).terms(),
                HiddenCostTerm(self.user, rig.context),
            ]
        )
        start = traj.q_feedback if from_trigger else traj.start_arm_aa
        source = "trigger" if from_trigger else "initial"
        _log(f"oracle rollout: starting from the {source} pose")
        trajectory = rollout_to_goal(
            cfg_goal,
            start,
            traj.goal,
            rig.context,
            oracle_costs,
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            progress_label="oracle",
            log_prefix=_LOG_PREFIX,
        )
        if from_trigger and traj.q_history:
            trajectory = np.concatenate(
                [np.asarray(traj.q_history[:-1]), trajectory], axis=0
            )
        traj.oracle_traj = trajectory
        traj.oracle_source = source
        traj.oracle_package = self._pin_package(trajectory, traj.oracle_package)
        payload = self._oracle_payload()
        # The initial oracle is already embedded in the trajectory beat; only the
        # explicit trigger rollout is a beat of its own.
        return self._record("oracle", payload) if from_trigger else payload

    def _oracle_payload(self) -> dict[str, Any]:
        traj = self.trajectory
        rig = self.rig
        if (
            traj is None
            or traj.oracle_traj is None
            or traj.oracle_source is None
            or traj.goal is None
        ):
            raise ValueError("Run an oracle rollout first.")
        return {
            "trajectory": traj.oracle_package,
            "metrics": violation_metrics(self.user, rig.context, traj.oracle_traj),
            "goal_reach": goal_reach(
                rig.context,
                rig._cfg_with_goal(traj.goal),
                traj.oracle_traj,
                traj.goal,
            ),
            "source": traj.oracle_source,
        }

    def run_clean_baseline(self) -> None:
        """Roll the default MPC from the start pose to the goal.

        Keeps only the shared anatomical joint-box limits (``_extra_costs``); no
        generated/round costs, no hidden persona preference, no MDM correction.
        """
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.goal is None:
            raise ValueError("Start a trajectory first.")
        _log("clean baseline: rolling out with box limits, no feedback")
        traj.clean_traj = rollout_to_goal(
            rig._cfg_with_goal(traj.goal),
            traj.start_arm_aa,
            traj.goal,
            rig.context,
            rig._extra_costs(self.user),
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            progress_label="clean-base",
            log_prefix=_LOG_PREFIX,
        )
        traj.clean_package = self._pin_package(traj.clean_traj, traj.clean_package)

    def _pin_package(
        self, trajectory: np.ndarray, previous: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Package a reference rollout once, with its mesh pinned.

        Packaging per payload would re-register the mesh on every live step and
        evict the id the browser is holding, so the reference rollouts are
        packaged when they are computed and reused verbatim afterwards.
        """
        if previous is not None:
            self.rig.meshes.unpin(previous["mesh_id"])
        return self.rig.package_trajectory(trajectory, self.user, pin_mesh=True)

    def _release_pinned_meshes(self) -> None:
        """Drop the outgoing trajectory's pins so its meshes become evictable."""
        traj = self.trajectory
        if traj is None:
            return
        for package in (traj.oracle_package, traj.clean_package):
            if package is not None:
                self.rig.meshes.unpin(package["mesh_id"])

    # --- MDM / clustering -------------------------------------------------

    def generate(
        self,
        prompt: str,
        n_samples: int,
        n_clusters: int,
        scale: float,
        clusterer: str = "agglo_end_pose",
    ) -> dict[str, Any]:
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.base_traj is None:
            raise ValueError("Start a trajectory first.")
        q_start = traj.q_feedback if traj.q_feedback is not None else traj.start_arm_aa
        start_pose = rig.gen.build_pose_from_arm_aa(rig.initial_hml_pose, q_start)
        t0 = time.perf_counter()
        _log(f"MDM generation: {n_samples} samples for {prompt!r}")
        traj.samples = rig.gen.generate_left_arm_position_samples(
            prompt,
            start_pose=start_pose,
            num_samples=n_samples,
            num_frames=rig.cfg.mdm_frames,
        )
        _log(f"MDM generation done in {time.perf_counter() - t0:.1f}s")
        traj.prompt = prompt
        traj.cluster_levels = []
        return self.recluster(n_clusters, scale, clusterer)

    def recluster(
        self, n_clusters: int, scale: float, clusterer: str = "agglo_end_pose"
    ) -> dict[str, Any]:
        """Cluster the current sample subset and assemble each option."""
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.samples is None:
            raise ValueError("Generate MDM samples first.")
        if traj.goal is None:
            raise ValueError("Start a trajectory first.")
        sample_indices = (
            traj.cluster_levels[-1].sample_indices
            if traj.cluster_levels
            else np.arange(traj.samples.shape[0], dtype=np.intp)
        )
        if sample_indices.size < n_clusters:
            labels = np.arange(sample_indices.size, dtype=np.intp)
            medoid_indices = {int(i): int(i) for i in labels}
        else:
            c = make_clusterer(clusterer, n_clusters, fk=rig.fk)
            labels = np.asarray(
                c.cluster_positions(traj.samples[sample_indices]), dtype=np.intp
            )
            medoid_indices = c.medoid_indices(labels)
        level = ClusterLevel(
            sample_indices=sample_indices,
            labels=labels,
            selected_label=None,
            n_clusters=n_clusters,
            scale=scale,
            medoid_indices=medoid_indices,
        )
        if traj.cluster_levels:
            traj.cluster_levels[-1] = level
        else:
            traj.cluster_levels.append(level)
        return self._activate_cluster_level()

    def rescale(self, scale: float) -> dict[str, Any]:
        """Re-assemble the current level's options at a new magnitude.

        Skips clustering (the samples and their labels do not depend on scale)
        so the UI can redraw while the magnitude slider moves; the selected
        label is kept.
        """
        traj = self.trajectory
        if traj is None or not traj.cluster_levels:
            raise ValueError("Cluster samples first.")
        traj.cluster_levels[-1].scale = scale
        return self._activate_cluster_level()

    def _activate_cluster_level(self) -> dict[str, Any]:
        """Build the current level's trajectories and navigation payload."""
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.samples is None or not traj.cluster_levels:
            raise ValueError("Generate and cluster MDM samples first.")
        if traj.goal is None:
            raise ValueError("Start a trajectory first.")
        user = self.user
        cfg_goal = rig._cfg_with_goal(traj.goal)
        level = traj.cluster_levels[-1]
        scale = level.scale
        level_samples = traj.samples[level.sample_indices]
        traj.labels = level.labels
        traj.cluster_means = {
            label: rig.gen.smpl_positions_to_left_arm_trajectory(
                level_samples[level.medoid_indices[label]],
                spine3_aa=rig.spine3_aa,
            )
            for label in sorted(int(v) for v in np.unique(level.labels))
        }
        traj.chosen_label = level.selected_label
        traj.scaled_correction = None
        traj.scale = scale
        traj.cluster_corrections = {}
        traj.cluster_fulls = {}
        oracle = oracle_cluster_scores(user, rig.context, traj.cluster_means, scale)
        clusters = []
        for label, mean in traj.cluster_means.items():
            scaled = scale_trajectory(mean, scale)
            full = (
                np.concatenate(
                    [np.asarray(traj.q_history, dtype=np.float64), scaled], axis=0
                )
                if traj.q_history
                else scaled
            )
            traj.cluster_corrections[label] = scaled
            traj.cluster_fulls[label] = full
            count = int(np.sum(level.labels == label))
            clusters.append(
                {
                    "label": label,
                    "count": count,
                    "can_refine": level.n_clusters >= 2,
                    "oracle_score": oracle[label],
                    "correction": rig.package_trajectory(scaled, user),
                    "full": rig.package_trajectory(full, user),
                    "full_segments": {
                        "history": len(traj.q_history),
                        "correction": int(scaled.shape[0]),
                    },
                    "full_metrics": violation_metrics(user, rig.context, full),
                    "full_goal_reach": goal_reach(
                        rig.context, cfg_goal, full, traj.goal
                    ),
                }
            )
        if traj.chosen_label is not None:
            traj.scaled_correction = traj.cluster_corrections[traj.chosen_label]
        path = [
            int(parent.selected_label)
            for parent in traj.cluster_levels[:-1]
            if parent.selected_label is not None
        ]
        payload = {
            "clusters": clusters,
            "selected_label": traj.chosen_label,
            "undesirable_labels": sorted(level.undesirable_labels),
            "depth": len(traj.cluster_levels) - 1,
            "path": path,
            "can_go_back": len(traj.cluster_levels) > 1,
            "active_sample_count": int(level.sample_indices.size),
            "scale": scale,
            "prompt": traj.prompt,
        }
        level.replay_payload = payload
        return payload

    def pick_cluster(self, label: int) -> dict[str, Any]:
        traj = self.trajectory
        if traj is None or label not in traj.cluster_corrections:
            raise ValueError(f"Unknown cluster label {label}.")
        if not traj.cluster_levels:
            raise ValueError("Cluster samples first.")
        level = traj.cluster_levels[-1]
        level.selected_label = label
        traj.chosen_label = label
        traj.scaled_correction = traj.cluster_corrections[label]
        return {"ok": True}

    def mark_cluster(self, label: int, undesirable: bool) -> dict[str, Any]:
        traj = self.trajectory
        if traj is None or label not in traj.cluster_corrections:
            raise ValueError(f"Unknown cluster label {label}.")
        if not traj.cluster_levels:
            raise ValueError("Cluster samples first.")
        level = traj.cluster_levels[-1]
        if undesirable:
            level.undesirable_labels.add(label)
        else:
            level.undesirable_labels.discard(label)
        return {
            "ok": True,
            "undesirable_labels": sorted(level.undesirable_labels),
        }

    def refine_cluster(
        self,
        label: int,
        n_clusters: int,
        scale: float,
        clusterer: str = "agglo_end_pose",
    ) -> dict[str, Any]:
        """Cluster only the raw samples belonging to the selected option."""
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.samples is None or not traj.cluster_levels:
            raise ValueError("Generate and cluster MDM samples first.")
        level = traj.cluster_levels[-1]
        if label not in traj.cluster_corrections:
            raise ValueError(f"Unknown cluster label {label}.")
        selected_indices = level.sample_indices[level.labels == label]
        if n_clusters < 2:
            raise ValueError("Refinement needs at least two clusters.")
        level.selected_label = label
        if selected_indices.size < n_clusters:
            labels = np.arange(selected_indices.size, dtype=np.intp)
            medoid_indices = {int(label): int(label) for label in labels}
        else:
            c = make_clusterer(clusterer, n_clusters, fk=rig.fk)
            labels = np.asarray(
                c.cluster_positions(traj.samples[selected_indices]), dtype=np.intp
            )
            medoid_indices = c.medoid_indices(labels)
        traj.cluster_levels.append(
            ClusterLevel(
                sample_indices=selected_indices,
                labels=labels,
                selected_label=None,
                n_clusters=n_clusters,
                scale=scale,
                medoid_indices=medoid_indices,
            )
        )
        return self._activate_cluster_level()

    def back_cluster(self) -> dict[str, Any]:
        """Return to the parent cluster level and restore its selection."""
        traj = self.trajectory
        if traj is None or len(traj.cluster_levels) <= 1:
            raise ValueError("Already at the root cluster level.")
        traj.cluster_levels.pop()
        return self._activate_cluster_level()

    # --- cost generation --------------------------------------------------

    def generated_cost_field(self, cost: GeneratedPythonCost) -> dict[str, Any]:
        """Evaluate the compiled cost over a cloud of plausible poses."""
        traj = self.trajectory
        assert traj is not None
        frames = [
            np.asarray(f, dtype=np.float64).reshape(-1, 3, 3)
            for f in [
                traj.base_traj,
                *traj.cluster_fulls.values(),
                *traj.cluster_corrections.values(),
            ]
            if f is not None
        ]
        anchors = np.concatenate(frames, axis=0)
        rng = np.random.default_rng(0)
        perturbed = anchors[None] + rng.normal(0.0, 0.25, size=(8, *anchors.shape))
        poses = np.concatenate([anchors[None], perturbed], axis=0).reshape(-1, 3, 3)
        if poses.shape[0] > 1500:
            poses = poses[rng.choice(poses.shape[0], 1500, replace=False)]
        feats = feature_series(self.rig.context, poses)
        batch = np.repeat(poses[:, None, :, :], 2, axis=1)
        active_features = generated_cost_feature_dependencies(cost.code)
        try:
            penalty = np.asarray(cost(batch), dtype=np.float64)
        except Exception:  # pragma: no cover - viz must not sink an expensive cost
            return {
                "features": {},
                "penalty": [],
                "active_features": list(active_features),
            }
        return {
            "features": {k: v.tolist() for k, v in feats.items()},
            "penalty": penalty.tolist(),
            "active_features": list(active_features),
        }

    def generate_cost(self, backend: str) -> dict[str, Any]:
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.scaled_correction is None or traj.goal is None:
            raise ValueError("Pick a cluster first.")
        from uncertain_feedback.planners.mpc.config import COST_BACKENDS

        if backend not in COST_BACKENDS:
            raise ValueError(f"Unknown cost-generation backend {backend!r}.")
        user = self.user
        cfg_goal = rig._cfg_with_goal(traj.goal)
        extra = rig._extra_costs(user)
        cost_q_start = (
            traj.q_feedback if traj.q_feedback is not None else traj.start_arm_aa
        )
        cost_dir = self.dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{backend}"
        instruction = traj.prompt or user.feedback_text
        result = generate_cost_for_cluster(
            mpc=None,
            cfg=cfg_goal,
            instruction=instruction,
            cluster_traj=traj.scaled_correction,
            current_q=cost_q_start,
            q_history=traj.q_history,
            context=rig.context,
            base_extra_costs=extra,
            cost_dir=cost_dir,
            corpus_dir=self.corpus.dir,
            body_pos=rig.body_pos,
            spine3_pos=rig.spine3_pos,
            spine3_aa=rig.spine3_aa,
            candidate_trajs=traj.cluster_corrections,
            highlight_label=traj.chosen_label,
            undesirable_labels=frozenset(
                traj.cluster_levels[-1].undesirable_labels
                if traj.cluster_levels
                else ()
            ),
            backend=backend,
            install=False,
            log_prefix=_LOG_PREFIX,
        )
        cost = result.generated_cost
        if cost is None:
            raise ValueError(f"Cost generation produced no cost (see {cost_dir}).")
        traj._last_generation = result
        traj._last_cost = cost
        traj._last_cost_dir = cost_dir
        traj._last_instruction = instruction
        cost_set = CompositeTrajectoryCost([*extra.terms(), cost])
        t0 = time.perf_counter()
        _log("assembling the chosen corrected path with the generated cost")
        rollout = _assemble_full_correction_traj(
            cfg_goal,
            traj.q_history,
            traj.scaled_correction,
            rig.context,
            cost_set,
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
        )
        _log(
            "generated-cost corrected path assembled in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        t0 = time.perf_counter()
        _log("rolling out the generated cost from the initial pose")
        start_rollout = rollout_to_goal(
            cfg_goal,
            traj.start_arm_aa,
            traj.goal,
            rig.context,
            cost_set,
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            progress_label="generated-from-start",
            log_prefix=_LOG_PREFIX,
        )
        _log(
            "generated-cost rollout from the initial pose done in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        payload = {
            "trajectory": rig.package_trajectory(rollout, user),
            "metrics": violation_metrics(user, rig.context, rollout),
            "goal_reach": goal_reach(rig.context, cfg_goal, rollout, traj.goal),
            "start_trajectory": rig.package_trajectory(start_rollout, user),
            "start_metrics": violation_metrics(user, rig.context, start_rollout),
            "start_goal_reach": goal_reach(
                rig.context, cfg_goal, start_rollout, traj.goal
            ),
            "cost_field": self.generated_cost_field(cost),
            "generated_bounds": _generated_bounds_from_artifacts(cost_dir),
            "rationale": _rationale_from_artifacts(cost_dir),
            "description": cost.description,
            "code": cost.code,
            "artifact_dir": str(cost_dir),
        }
        traj._last_cost_payload = payload
        (cost_dir / "demo_runner_payload.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    # --- multi-round feedback ---------------------------------------------

    def _unified_record(self) -> dict[str, Any] | None:
        if self.unified_cost is None:
            return None
        return {
            "description": self.unified_cost.description,
            "code": self.unified_cost.code,
        }

    def _round_record(self, round_: CostRound) -> dict[str, Any]:
        return {
            "index": round_.index,
            "goal": list(round_.goal) if round_.goal is not None else None,
            "feedback_text": round_.feedback_text,
            "trigger_step": round_.trigger_step,
            "trigger_reason": round_.trigger_reason,
            "trigger_violation": round_.trigger_violation,
            "cluster_labels": list(round_.cluster_labels),
            "description": round_.description,
            "code": round_.cost_code,
            "rationale": _rationale_from_artifacts(round_.round_dir),
            "generated_bounds": _generated_bounds_from_artifacts(round_.round_dir),
            "artifact_dir": str(round_.round_dir),
        }

    def commit_round(self) -> dict[str, Any]:
        traj = self.trajectory
        if (
            traj is None
            or traj._last_generation is None
            or traj._last_cost is None
            or traj._last_cost_dir is None
            or traj._last_instruction is None
            or traj.goal is None
        ):
            raise ValueError("Generate a cost first.")
        generation = traj._last_generation
        state_path = traj._last_cost_dir / "state.pkl"
        generation.eval_state.save(state_path)
        cluster_labels = tuple(
            int(level.selected_label)
            for level in traj.cluster_levels
            if level.selected_label is not None
        )
        cost_round = CostRound(
            index=len(self.rounds),
            goal=(float(traj.goal[0]), float(traj.goal[1]), float(traj.goal[2])),
            feedback_text=traj._last_instruction,
            trigger_step=traj.trigger_step if traj.trigger_step is not None else 0,
            round_dir=traj._last_cost_dir.resolve(),
            state_path=state_path.resolve(),
            cost_code=traj._last_cost.code,
            params=traj._last_cost.params,
            summaries=generation.summaries,
            image_paths=tuple(path.resolve() for path in generation.images.values()),
            trajectory_index=self.trajectory_count - 1,
            cluster_labels=cluster_labels,
            trigger_reason=traj.trigger_reason or "discomfort",
            trigger_violation=traj.trigger_violation,
            description=generation.description,
            explanation=generation.explanation,
            interpretation=generation.interpretation,
            grounding=generation.grounding,
        )
        self.rounds.append(cost_round)
        self._round_costs.append(traj._last_cost)
        self.round_records.append(self._round_record(cost_round))
        self.unified_cost = traj._last_cost if len(self.rounds) == 1 else None
        if len(self.rounds) == 1:
            self.combined_corpus_count = len(self.corpus.entries())
        self._record_final_feedback()
        traj._last_generation = None
        traj._last_cost = None
        traj._last_cost_dir = None
        traj._last_instruction = None
        traj._last_cost_payload = None
        self._save()
        _log(
            f"committed round {cost_round.index} "
            f"(goal={list(cost_round.goal)}, cluster_labels={list(cluster_labels)})"
        )
        return self._record(
            "round",
            {
                "rounds": self.round_records,
                "unified": self._unified_record(),
                "combined_corpus_count": self.combined_corpus_count,
            },
        )

    def apply_round_and_continue(
        self, advance: bool = True, current_mesh_only: bool = False
    ) -> dict[str, Any]:
        """Commit the selected feedback, install it, and resume to the next pause."""
        traj = self.trajectory
        rig = self.rig
        if traj is None or not traj.paused:
            raise ValueError("The multi-turn trajectory is not paused for feedback.")
        if traj.scaled_correction is None or traj._last_cost is None:
            raise ValueError("Generate a cost for the selected cluster first.")
        correction = traj.scaled_correction.copy()
        self.commit_round()
        cutoff = max(1, round(len(correction) * rig.cfg.trajectory_fraction))
        correction = correction[:cutoff]
        traj.mpc.set_mdm_goal(correction[-1])
        traj.mpc.push_trajectory(correction)
        traj.mpc.set_extra_costs(
            CompositeTrajectoryCost(
                [*rig._extra_costs(self.user).terms(), *self._round_costs]
            )
        )
        traj.paused = False
        traj.clear_pending_feedback()
        if advance:
            traj.advance(self)
            return self._record("trajectory", self._trajectory_payload())
        return self._trajectory_payload(current_mesh_only=current_mesh_only)

    def ignore_comfort_violation(self) -> dict[str, Any]:
        """Resume without feedback after ignoring the current discomfort event."""
        traj = self.trajectory
        if traj is None or not traj.paused:
            raise ValueError("The multi-turn trajectory is not paused for feedback.")
        if traj.trigger_reason != "discomfort":
            raise ValueError("The current feedback trigger is not a comfort violation.")
        _log(f"ignored comfort violation at frame {traj.step}")
        traj.paused = False
        traj.clear_pending_feedback()
        traj.advance(self)
        return self._record("trajectory", self._trajectory_payload())

    def remove_round(self, index: int) -> dict[str, Any]:
        rig = self.rig
        if index < 0 or index >= len(self.rounds):
            raise ValueError(f"Unknown feedback round {index}.")
        removed = self.rounds.pop(index)
        self.round_records.pop(index)
        self._round_costs.pop(index)
        self.rounds = [replace(round_, index=i) for i, round_ in enumerate(self.rounds)]
        for i, record in enumerate(self.round_records):
            record["index"] = i
        self.unified_cost = (
            self._round_costs[0] if len(self._round_costs) == 1 else None
        )
        if self.trajectory is not None:
            self.trajectory.mpc.set_extra_costs(
                CompositeTrajectoryCost(
                    [*rig._extra_costs(self.user).terms(), *self._round_costs]
                )
            )
        self._save()
        _log(f"removed feedback round {removed.index}")
        return {"rounds": self.round_records, "unified": self._unified_record()}

    def combine_rounds(self) -> dict[str, Any]:
        traj = self.trajectory
        rig = self.rig
        if not self.rounds:
            raise ValueError("Commit at least one round before combining.")
        round_ = self.rounds[-1]
        state = EvalState.load(round_.state_path)
        combine_dir = self.dir / f"{time.strftime('%Y%m%d_%H%M%S')}_combine"
        combinator = CombineCostGenerator(
            context=state.make_generated_context(),
            instruction=round_.feedback_text,
            summaries=round_.summaries,
            run_dir=combine_dir,
            images={path.name: path for path in round_.image_paths},
            use_images=rig.cfg.llm_cost.use_images,
            model=rig.cfg.llm_cost.model,
            strict=rig.cfg.llm_cost.strict,
            mpc=None,
            rollout_fn=state.make_rollout_fn(),
            eval_state=state,
            codex_cmd=rig.cfg.llm_cost.codex_cmd,
            corpus_dir=self.corpus.dir,
            rounds=self.rounds,
        )
        t0 = time.perf_counter()
        _log(f"combining {len(self.rounds)} rounds (artifacts={combine_dir})")
        combined = combinator.generate(install=False)
        if combined is None:
            raise ValueError(f"Round combination failed (see {combine_dir}).")
        _log(f"combination done in {time.perf_counter() - t0:.1f}s")
        self.unified_cost = combined
        self.combined_corpus_count = len(self.corpus.entries())
        self._save()
        scores_path = combine_dir / "scores.json"
        scores = (
            json.loads(scores_path.read_text(encoding="utf-8"))
            if scores_path.exists()
            else None
        )
        payload = {
            "cost_field": self.generated_cost_field(combined),
            "description": combined.description,
            "code": combined.code,
            "scores": scores,
            "artifact_dir": str(combine_dir),
            "rounds": self.round_records,
            "combined_corpus_count": self.combined_corpus_count,
        }
        # Combined between trajectories: there is no pose or goal to demonstrate
        # the unified cost on, and start_trajectory installs it on the next one.
        if traj is None or traj.goal is None:
            return payload
        cfg_goal = rig._cfg_with_goal(traj.goal)
        installed = replace_generated_costs(rig._extra_costs(self.user), combined)
        traj.mpc.set_extra_costs(installed)
        t0 = time.perf_counter()
        _log("rolling out the unified cost from the initial pose")
        rollout = rollout_to_goal(
            cfg_goal,
            traj.start_arm_aa,
            traj.goal,
            rig.context,
            installed,
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            progress_label="unified-from-start",
            log_prefix=_LOG_PREFIX,
        )
        _log(f"unified-cost rollout done in {time.perf_counter() - t0:.1f}s")
        payload.update(
            trajectory=rig.package_trajectory(rollout, self.user),
            metrics=violation_metrics(self.user, rig.context, rollout),
            goal_reach=goal_reach(rig.context, cfg_goal, rollout, traj.goal),
        )
        return payload

    def reset_rounds(self) -> dict[str, Any]:
        rig = self.rig
        self.rounds = []
        self.round_records = []
        self._round_costs = []
        self.unified_cost = None
        self.combined_corpus_count = 0
        if self.trajectory is not None:
            self.trajectory.clear_pending_feedback()
            self.trajectory.mpc.set_extra_costs(rig._extra_costs(self.user))
        self._save()
        _log("multi-round state reset")
        return {"ok": True}

    # --- corpus -----------------------------------------------------------

    def remove_corpus_entry(self, index: int) -> dict[str, Any]:
        self.corpus.remove(index)
        self._save()
        return {"corpus": self.corpus.entries()}

    # --- payloads ---------------------------------------------------------

    def _trajectory_payload(self, current_mesh_only: bool = False) -> dict[str, Any]:
        traj = self.trajectory
        rig = self.rig
        if traj is None or traj.goal is None:
            raise ValueError("Start a trajectory first.")
        trajectory = np.asarray(traj.executed, dtype=np.float64)
        trigger = (
            None
            if not traj.paused
            else {
                "step": traj.trigger_step,
                "reason": traj.trigger_reason,
                "violation": traj.trigger_violation,
            }
        )
        return {
            "trajectory": rig.package_trajectory(
                trajectory, self.user, current_mesh_only=current_mesh_only
            ),
            "oracle": None if traj.oracle_traj is None else self._oracle_payload(),
            "clean_base": traj.clean_package,
            "metrics": violation_metrics(self.user, rig.context, trajectory),
            "goal_reach": goal_reach(
                rig.context, rig._cfg_with_goal(traj.goal), trajectory, traj.goal
            ),
            "status": (
                "complete" if traj.complete else "paused" if traj.paused else "running"
            ),
            "trigger": trigger,
            "step": traj.step,
            "step_limit": rig.cfg.steps,
            "reached_goal": traj.reached_goal,
            "error": traj.error,
            "artifact_dir": str(traj.artifact_dir),
            "trajectory_count": self.trajectory_count,
            "corpus": self.corpus.entries(),
            "combined_corpus_count": self.combined_corpus_count,
            "rounds": self.round_records,
            "unified": self._unified_record(),
            "pending_cost": traj._last_cost_payload,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "persona": self.persona_name,
            "dir": str(self.dir),
            "started": self.started,
            "trajectory_count": self.trajectory_count,
            "corpus": self.corpus.entries(),
            "combined_corpus_count": self.combined_corpus_count,
            "rounds": self.round_records,
            "unified": self._unified_record(),
            "trajectory": (
                None if self.trajectory is None else self._trajectory_payload()
            ),
        }
