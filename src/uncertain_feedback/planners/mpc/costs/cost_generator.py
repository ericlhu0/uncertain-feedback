"""Shared base for MPC cost-function generators.

Three sibling strategies generate an executable :class:`GeneratedPythonCost` from a
user correction, all constructed and called the same way:

- ``llm``   — single-turn LLM call (:class:`~...costs.llm_costs.LlmCostGenerator`).
- ``turns`` — multi-turn LLM conversation that refines the cost
  (:class:`~...costs.turns_costs.TurnsCostGenerator`).
- ``agent`` — delegates iteration to the ``codex`` CLI
  (:class:`~...costs.agent_costs.AgentCostGenerator`).

:class:`CostGenerator` holds the shared inputs and helpers (prompt building, model
construction, response parsing/validation, artifact saving, installation) so each
subclass body stays small. :func:`create_cost_generator` selects the strategy from
config; this is the only place that branches on the backend.
"""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedCostValidationError,
    GeneratedPythonCost,
    LlmCostResponse,
    build_joint_angle_series,
    extract_json_object,
    parse_llm_cost_response,
)
from uncertain_feedback.planners.mpc.costs.prompts import (
    build_author_prompt,
    build_ground_prompt,
    build_interpret_prompt,
)

_SYSTEM_PROMPT = (
    "You generate safe, vectorized Python MPC trajectory cost functions. "
    "Return only the requested JSON object."
)
_DEFAULT_LLM_MODEL = "gpt-5.6-luna"
# Reasoning effort per model; a model absent here is sent without one.
_REASONING_EFFORT = {"gpt-5.6-luna": "xhigh", "gpt-5.6-sol": "low"}


def _make_llm_model(model_name: str) -> Any:
    """Build the cost-generator LLM wrapper."""
    from uncertain_feedback.llm import (  # pylint: disable=import-outside-toplevel
        OpenAIModel,
    )

    return OpenAIModel(
        model=model_name,
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=None,
        reasoning_effort=_REASONING_EFFORT.get(model_name),
        stream_reasoning_summary=True,
    )


def artifact_run_dir(base_dir: Path, artifact_dir: Path) -> Path:
    """Return a unique, timestamped artifact directory for one generation."""
    root = artifact_dir if artifact_dir.is_absolute() else base_dir / artifact_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def resample_equidistant(traj: np.ndarray, n: int) -> np.ndarray:
    """Resample a ``(T, ...)`` trajectory to ``n`` points equidistant in joint-space
    arclength.

    Removes timing entirely — speed differences and dwell frames carry no signal
    here (MDM frames and MPC steps are on different clocks, and MDM output is
    systematically slower than a fresh rollout), so trajectories compare purely by
    path. A stationary trajectory becomes its pose repeated ``n`` times.
    """
    traj = np.asarray(traj, dtype=np.float64)
    flat = traj.reshape(traj.shape[0], -1)
    segments = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segments)])
    if arclength[-1] <= 0.0:
        return np.repeat(traj[:1], n, axis=0)
    targets = np.linspace(0.0, arclength[-1], n)
    out = np.stack(
        [np.interp(targets, arclength, flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape((n, *traj.shape[1:]))


@dataclass(frozen=True)
class CostRanking:
    """How a candidate cost orders the preferences the user actually revealed.

    ``rank_accuracy`` is the fraction of preference pairs the cost orders
    correctly; ``normalized_margin`` is the mean z-scored separation of the chosen
    correction below the alternatives (a scale-free tiebreak, not a gate).
    ``inert`` marks a cost that returned (near-)identical values for every
    trajectory — it never discriminates, so its ranking is vacuous.
    """

    rank_accuracy: float
    normalized_margin: float
    inert: bool
    costs: dict[str, float]

    @property
    def improves_original_plan(self) -> bool | None:
        """Whether the chosen correction costs strictly less than the original plan."""
        chosen = self.costs.get("chosen_correction")
        original = self.costs.get("original_plan")
        if chosen is None or original is None:
            return None
        return chosen < original

    @property
    def sort_key(self) -> tuple[float, float]:
        """Lower-is-better selection key: rank accuracy first, margin as tiebreak."""
        if self.inert:
            return (math.inf, math.inf)
        return (1.0 - self.rank_accuracy, -self.normalized_margin)

    def as_json(self) -> dict[str, Any]:
        """JSON-safe payload for score/rationale artifacts."""
        return {
            "rank_accuracy": self.rank_accuracy,
            "normalized_margin": self.normalized_margin,
            "inert": self.inert,
            "improves_original_plan": self.improves_original_plan,
            "costs": self.costs,
        }


def rank_candidate_cost(
    context: GeneratedCostContext, cost: GeneratedPythonCost
) -> CostRanking | None:
    """Evaluate a candidate cost by ranking consistency, not trajectory matching.

    The cost function itself is applied to the trajectories whose preference order
    the user revealed: the chosen correction (``mdm_traj``) must cost strictly less
    than the original plan the user interrupted (``reference_traj``) and every
    cluster the user explicitly marked undesirable (``rejected_trajs``).
    Any cost that captures the intent satisfies this; recreating the correction
    trajectory is not required. All trajectories are resampled to equidistant
    joint-space points first (timing is a pipeline artifact, not intent) and
    compared after z-normalization (generated costs have arbitrary scale).

    Returns ``None`` when the context has no comparison trajectories (no reference
    rollout and no marked-undesirable clusters), in which case callers fall back to
    the L2 rollout score.
    """
    chosen = np.asarray(context.mdm_traj, dtype=np.float64)
    if chosen.ndim != 3 or chosen.shape[0] == 0:
        return None
    trajs: dict[str, np.ndarray] = {}
    if context.reference_traj is not None and context.reference_traj.size > 0:
        trajs["original_plan"] = context.reference_traj
    for i, rejected in enumerate(context.rejected_trajs):
        trajs[f"rejected_cluster_{i}"] = rejected
    if not trajs:
        return None
    trajs["chosen_correction"] = chosen
    n = max(traj.shape[0] for traj in trajs.values())
    batch = np.stack([resample_equidistant(traj, n) for traj in trajs.values()])
    raw = np.asarray(cost(batch), dtype=np.float64)
    costs = {name: float(value) for name, value in zip(trajs, raw)}
    if not np.all(np.isfinite(raw)) or float(raw.std()) < 1e-12:
        return CostRanking(0.0, 0.0, True, costs)
    z = dict(zip(trajs, (raw - raw.mean()) / raw.std()))
    z_chosen = z["chosen_correction"]
    pairs: list[bool] = []
    margins: list[float] = []
    if "original_plan" in z:
        pairs.append(bool(z_chosen < z["original_plan"]))
        margins.append(float(z["original_plan"] - z_chosen))
    for name in trajs:
        if name.startswith("rejected_cluster_"):
            pairs.append(bool(z_chosen < z[name]))
            margins.append(float(z[name] - z_chosen))
    return CostRanking(
        rank_accuracy=float(np.mean(pairs)),
        normalized_margin=float(np.mean(margins)),
        inert=False,
        costs=costs,
    )


def _score_rollout(
    context: GeneratedCostContext, rollout: np.ndarray
) -> float:
    """Mean per-frame FK L2 distance between a rollout and the MDM correction.

    Both trajectories are resampled to equidistant joint-space points
    (:func:`resample_equidistant`) so the comparison is purely about path — MDM
    output is systematically slower than a fresh rollout, so frame-wise timing is
    a pipeline artifact. ``math.inf`` when either trajectory is empty / malformed.
    """
    rollout = np.asarray(rollout, dtype=np.float64)
    target = np.asarray(context.mdm_traj, dtype=np.float64)
    if (
        rollout.ndim != 3
        or rollout.shape[0] == 0
        or target.ndim != 3
        or target.shape[0] == 0
    ):
        return math.inf
    n = max(rollout.shape[0], target.shape[0])
    rollout_positions = context.fk_batch(resample_equidistant(rollout, n))
    mdm_positions = context.fk_batch(resample_equidistant(target, n))
    return float(np.linalg.norm(rollout_positions - mdm_positions, axis=-1).mean())


def goal_reach_report(
    context: GeneratedCostContext, rollout: np.ndarray | None
) -> dict[str, Any] | None:
    """Whether a candidate rollout still reaches the Cartesian goal.

    Reproduces the MPC's own criterion (``ArmMPCCartesian.goal_reached``): forward-
    kinematics the final rollout frame, take the spine3-relative wrist, and compare its
    distance to ``context.cartesian_goal`` against ``context.cartesian_threshold``.
    Returns ``None`` when no Cartesian goal is available (non-Cartesian planners) or the
    rollout is empty/malformed, so callers degrade to no goal feedback.
    """
    if context.cartesian_goal is None or context.cartesian_threshold is None:
        return None
    if rollout is None:
        return None
    rollout = np.asarray(rollout, dtype=np.float64)
    if rollout.ndim != 3 or rollout.shape[0] == 0:
        return None
    arm_pos = context.fk.fk(rollout[-1], context.spine3_pos, context.spine3_aa)
    wrist_rel = arm_pos[-1] - context.spine3_pos
    distance = float(np.linalg.norm(wrist_rel - context.cartesian_goal))
    return {
        "reached": distance < context.cartesian_threshold,
        "distance": distance,
        "threshold": float(context.cartesian_threshold),
    }


def parse_goal_conflict(interpret_text: str) -> bool:
    """Read the stage-1 ``goal_conflict`` flag; ``False`` if absent/unparseable."""
    data = extract_json_object(interpret_text)
    return bool(data is not None and data.get("goal_conflict", False))


def evaluate_candidate_cost(
    context: GeneratedCostContext,
    cost: GeneratedPythonCost,
    rollout_fn: Callable[[GeneratedPythonCost], np.ndarray | None] | None = None,
) -> tuple[float, np.ndarray | None]:
    """Score a candidate cost; lower is better.

    Rolls the goal-seeking MPC toward its original goal with this cost installed
    (via ``rollout_fn``), resamples the result to the corrected (MDM) trajectory's
    length, and returns the mean per-frame Cartesian (FK) L2 distance to it plus the
    raw rollout trajectory. A cost that steers the goal-seeking motion to match the
    user's correction scores low.

    Returns ``(math.inf, None)`` when no rollout is available (``rollout_fn`` is
    ``None`` or yields no trajectory, e.g. planners without a persistent Cartesian
    goal).
    """
    if rollout_fn is None:
        return math.inf, None
    rollout = rollout_fn(cost)
    if rollout is None:
        return math.inf, None
    return _score_rollout(context, rollout), rollout


def evaluate_and_render(
    context: GeneratedCostContext,
    cost: GeneratedPythonCost,
    rollout_fn: Callable[[GeneratedPythonCost], np.ndarray | None] | None,
    image_path: Path,
    *,
    angle_path: Path | None = None,
    rollout_path: Path | None = None,
    video_path: Path | None = None,
) -> tuple[float, Path | None, np.ndarray | None]:
    """Score a candidate cost and render the rollout-vs-correction comparison.

    Rolls the goal-seeking MPC **once** with ``cost`` installed, computes the L2
    score (:func:`_score_rollout`), and renders ``image_path`` overlaying that
    rollout against the MDM correction
    (:meth:`ArmVisualizer.render_cost_feedback_overlay`). When ``angle_path`` is
    given, also renders a joint-angle-over-time comparison there
    (:meth:`ArmVisualizer.render_joint_angle_comparison`). Returns
    ``(score, image_path, rollout)``, or ``(math.inf, None, None)`` when no rollout
    is available (planners without a persistent Cartesian goal) so callers degrade
    to text-only feedback.
    """
    from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
        ArmVisualizer,
    )

    if rollout_fn is None:
        return math.inf, None, None
    rollout = rollout_fn(cost)
    if rollout is None:
        return math.inf, None, None
    rollout = np.asarray(rollout, dtype=np.float64)
    if rollout.ndim != 3 or rollout.shape[0] == 0:
        return math.inf, None, None
    score = _score_rollout(context, rollout)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = ArmVisualizer(context.fk)
    correction_traj = _reference_with_correction_traj(context)
    visualizer.render_cost_feedback_overlay(
        image_path,
        rollout_traj=rollout,
        correction_traj=correction_traj,
        current_q=context.current_q,
        spine3_pos=context.spine3_pos,
        spine3_aa=context.spine3_aa,
        body_pos=context.body_pos,
    )
    if angle_path is not None:
        angle_path.parent.mkdir(parents=True, exist_ok=True)
        reference_series = (
            build_joint_angle_series(context, context.reference_traj)
            if context.reference_traj is not None and context.reference_traj.size > 0
            else None
        )
        visualizer.render_joint_angle_comparison(
            angle_path,
            target_series=build_joint_angle_series(context, correction_traj),
            rollout_series=build_joint_angle_series(context, rollout),
            reference_series=reference_series,
        )
    if rollout_path is not None:
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rollout_path, rollout)
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.render_rollout_video(
            rollout,
            video_path,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=context.body_pos,
        )
    return score, image_path, rollout


def _reference_with_correction_traj(context: GeneratedCostContext) -> np.ndarray:
    """Return the target path that includes the user's correction."""
    if context.full_correction_traj is not None:
        return context.full_correction_traj
    return context.mdm_traj


class CostGenerator(ABC):
    """Base class for cost-function generators.

    Subclasses implement :meth:`generate`; the base supplies prompt building, LLM
    construction, response parsing/validation, artifact saving, and installation.
    """

    def __init__(
        self,
        context: GeneratedCostContext,
        instruction: str,
        summaries: dict[str, Any],
        *,
        run_dir: Path,
        images: dict[str, Path] | None = None,
        use_images: bool = True,
        model: str | None = None,
        strict: bool = False,
        mpc: Any | None = None,
        llm_model_factory: Callable[[str], Any] = _make_llm_model,
        rollout_fn: (
            Callable[[GeneratedPythonCost], np.ndarray | None] | None
        ) = None,
        eval_state: Any | None = None,
        save_candidate_videos: bool = False,
        corpus_dir: Path | None = None,
    ) -> None:
        self.context = context
        self.instruction = instruction
        self.summaries = summaries
        self.run_dir = run_dir
        self.images = images or {}
        self.use_images = use_images
        self.model = model
        self.strict = strict
        self.mpc = mpc
        self.llm_model_factory = llm_model_factory
        self.rollout_fn = rollout_fn
        self.eval_state = eval_state
        self.save_candidate_videos = save_candidate_videos
        self.corpus_dir = corpus_dir
        self._comfortable_corpus: tuple[np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )

    # -- shared helpers -----------------------------------------------------

    @property
    def model_name(self) -> str:
        """Resolved model name (config, then env, then default)."""
        return self.model or os.getenv("OPENAI_MODEL", _DEFAULT_LLM_MODEL)

    def make_llm(self) -> Any:
        """Construct the cost-generator LLM wrapper for this run's model."""
        return self.llm_model_factory(self.model_name)

    def _run_stage(
        self,
        llm: Any,
        name: str,
        prompt_text: str,
        *,
        image_input: list[str] | None = None,
    ) -> str:
        """Run one staged LLM call, persist its prompt and raw response, return the text."""
        (self.run_dir / f"{name}_prompt.txt").write_text(prompt_text, encoding="utf-8")
        raw = llm.get_full_output(prompt_text, image_input=image_input)
        (self.run_dir / f"{name}_response.txt").write_text(raw, encoding="utf-8")
        self._append_stage_log(name, prompt_text, raw)
        return raw

    def _append_stage_log(
        self,
        name: str,
        prompt_text: str,
        response_text: str,
    ) -> None:
        """Append one prompt/response pair to the readable stage log."""
        with open(self.run_dir / "stage_log.md", "a", encoding="utf-8") as f:
            f.write(f"\n\n## {name}\n\n")
            f.write("### Prompt\n\n")
            f.write("```text\n")
            f.write(prompt_text)
            if not prompt_text.endswith("\n"):
                f.write("\n")
            f.write("```\n\n")
            f.write("### Response\n\n")
            f.write("```text\n")
            f.write(response_text)
            if not response_text.endswith("\n"):
                f.write("\n")
            f.write("```\n")

    def interpret(self, llm: Any) -> str:
        """Stage one: read the correction (instruction + contrast images + compact summary)."""
        text, image_paths = build_interpret_prompt(
            self.instruction, self.summaries, self.images if self.use_images else {}
        )
        image_input = [str(path) for path in image_paths] or None
        return self._run_stage(llm, "interpret", text, image_input=image_input)

    def ground(self, llm: Any, interpretation: str) -> str:
        """Stage two: turn the preference into a concrete numeric spec (full summaries)."""
        text = build_ground_prompt(
            interpretation, self.summaries, self.corpus_grounding_note()
        )
        return self._run_stage(llm, "ground", text)

    def comfortable_corpus(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return accepted poses and their corpus entry/frame identifiers."""
        if self._comfortable_corpus is not None:
            return self._comfortable_corpus
        if self.corpus_dir is None:
            return None
        root = self.corpus_dir.resolve()
        manifest = root / "manifest.json"
        if not manifest.exists():
            return None
        poses: list[np.ndarray] = []
        entry_ids: list[np.ndarray] = []
        frame_ids: list[np.ndarray] = []
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            trajectory_path = (root / entry["traj_file"]).resolve()
            if not trajectory_path.is_relative_to(root):
                raise GeneratedCostValidationError(
                    f"corpus trajectory leaves the corpus directory: {trajectory_path}"
                )
            trajectory = np.asarray(np.load(trajectory_path), dtype=np.float64)
            cutoff = min(
                int(entry.get("comfortable_until", trajectory.shape[0])),
                int(trajectory.shape[0]),
            )
            if cutoff <= 0:
                continue
            poses.append(trajectory[:cutoff])
            entry_ids.append(np.full(cutoff, int(entry["index"]), dtype=np.int64))
            frame_ids.append(np.arange(cutoff, dtype=np.int64))
        if not poses:
            return None
        self._comfortable_corpus = (
            np.concatenate(poses),
            np.concatenate(entry_ids),
            np.concatenate(frame_ids),
        )
        return self._comfortable_corpus

    def corpus_grounding_note(self) -> str | None:
        """Summarize accepted-pose feature ranges for in-process LLM backends."""
        corpus = self.comfortable_corpus()
        if corpus is None:
            return None
        poses, entry_ids, _ = corpus
        feature_methods = {
            "elbow_flexion": self.context.elbow_flexion_angles,
            "shoulder_flexion_extension": (
                self.context.shoulder_flexion_extension_angles
            ),
            "shoulder_abduction_adduction": (
                self.context.shoulder_abduction_adduction_angles
            ),
            "shoulder_elevation": self.context.shoulder_elevation_angles,
            "shoulder_internal_external_rotation": (
                self.context.shoulder_internal_external_rotation_angles
            ),
        }
        lines = [
            "ADDITIONAL GROUNDING EVIDENCE — previously accepted poses. Every pose "
            "below was executed without a discomfort report. Any generated bound "
            "must leave every one unpenalized; move thresholds beyond these ranges."
        ]
        for entry_id in np.unique(entry_ids):
            selected = poses[entry_ids == entry_id]
            ranges = {
                name: {
                    "min": float(np.min(method(selected))),
                    "max": float(np.max(method(selected))),
                }
                for name, method in feature_methods.items()
            }
            lines.append(
                f"- corpus entry {entry_id}: {json.dumps(ranges, sort_keys=True)}"
            )
        lines.append(
            "The runtime will reject the authored cost if it assigns positive cost "
            "to any of these accepted poses."
        )
        return "\n".join(lines)

    def require_comfortable_corpus_unpenalized(self, cost: GeneratedPythonCost) -> None:
        """Fail closed when a generated cost penalizes a previously accepted pose."""
        corpus = self.comfortable_corpus()
        if corpus is None:
            return
        poses, entry_ids, frame_ids = corpus
        stationary_rollouts = np.repeat(poses[:, np.newaxis], repeats=2, axis=1)
        values = cost(stationary_rollouts)
        worst = int(np.argmax(values))
        if values[worst] <= 1e-10:
            return
        raise GeneratedCostValidationError(
            "generated cost penalizes a previously accepted pose: corpus entry "
            f"{entry_ids[worst]}, frame {frame_ids[worst]}, cost {values[worst]:.6g}; "
            "all frames before comfortable_until must have zero cost"
        )

    def author(
        self, llm: Any, specification: str
    ) -> tuple[LlmCostResponse, GeneratedPythonCost]:
        """Stage three: implement the spec faithfully as the compiled cost JSON."""
        text = build_author_prompt(specification)
        raw = self._run_stage(llm, "author", text)
        return self.parse_cost(raw)

    def compile_cost(
        self,
        code: str,
        params: dict[str, Any] | None = None,
        description: str = "",
    ) -> GeneratedPythonCost:
        """Compile and validate generated code into an executable cost."""
        cost = GeneratedPythonCost(
            code=code,
            params=params or {},
            description=description,
            context=self.context,
        )
        # Smoke-test on a dummy rollout batch so a broken cost fails here.
        validation_q = np.repeat(
            self.context.current_q[np.newaxis, np.newaxis], repeats=11, axis=1
        )
        cost(validation_q)
        return cost

    def parse_cost(self, raw: str) -> tuple[LlmCostResponse, GeneratedPythonCost]:
        """Parse a JSON LLM response and compile it into a cost."""
        response = parse_llm_cost_response(raw)
        cost = self.compile_cost(
            response.code, response.params, response.description
        )
        self.require_comfortable_corpus_unpenalized(cost)
        return response, cost

    def require_original_plan_improvement(self, ranking: CostRanking | None) -> None:
        """Reject a cost that does not prefer the correction to the interrupted plan."""
        if ranking is None or ranking.improves_original_plan is None:
            return
        if ranking.improves_original_plan:
            return
        chosen = ranking.costs["chosen_correction"]
        original = ranking.costs["original_plan"]
        raise GeneratedCostValidationError(
            "generated cost does not improve on the original plan: "
            f"chosen_correction cost {chosen:.6g} must be strictly less than "
            f"original_plan cost {original:.6g}"
        )

    def begin(self) -> None:
        """Create the run dir and persist the prompt inputs."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "summaries.json", "w", encoding="utf-8") as f:
            json.dump(self.summaries, f, indent=2, sort_keys=True)
        self._save_reference_video()

    def _save_reference_video(self) -> None:
        from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
            ArmVisualizer,
        )

        reference = np.asarray(
            _reference_with_correction_traj(self.context), dtype=np.float64
        )
        if reference.ndim != 3 or reference.shape[0] == 0:
            return
        video_path = self.run_dir / "reference_with_correction.mp4"
        mdm_traj = np.asarray(self.context.mdm_traj, dtype=np.float64)
        mdm_goal_q = (
            mdm_traj[-1]
            if mdm_traj.ndim == 3 and mdm_traj.shape[0] > 0
            else reference[-1]
        )
        try:
            ArmVisualizer(self.context.fk).render_rollout_video(
                reference,
                video_path,
                spine3_pos=self.context.spine3_pos,
                spine3_aa=self.context.spine3_aa,
                body_pos=self.context.body_pos,
                mdm_goal_q=mdm_goal_q,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[cost-gen] failed to render {video_path}: {exc}")

    def save_response(
        self, response: LlmCostResponse, *, dir_: Path | None = None
    ) -> None:
        """Write the winning cost code, params, and explanations to ``dir_``."""
        out = dir_ or self.run_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "cost.py").write_text(response.code, encoding="utf-8")
        if response.explanation:
            (out / "explanation.txt").write_text(
                response.explanation, encoding="utf-8"
            )
            print(f"[cost-gen] explanation: {response.explanation}")
        if response.recipient_explanation:
            (out / "recipient_explanation.txt").write_text(
                response.recipient_explanation, encoding="utf-8"
            )
            print(
                "[cost-gen] recipient explanation: "
                f"{response.recipient_explanation}"
            )
        with open(out / "params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": response.description,
                    "explanation": response.explanation,
                    "recipient_explanation": response.recipient_explanation,
                    "params": response.params,
                    "model": self.model_name,
                    "backend": type(self).__name__,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    def save_rationale(
        self,
        response: LlmCostResponse,
        *,
        interpret_raw: str | None,
        ground_raw: str | None,
        ranking: CostRanking | None,
    ) -> None:
        """Write ``rationale.json`` chaining instruction -> interpret -> ground -> final.

        Stage sections are the stage JSONs parsed leniently and dumped verbatim —
        no key validation, so a malformed or missing stage degrades to ``null``
        rather than failing the generation.
        """
        interpret = extract_json_object(interpret_raw) if interpret_raw else None
        ground = extract_json_object(ground_raw) if ground_raw else None
        if isinstance(ground, dict):
            terms = ground.get("terms", [])
            for term in terms if isinstance(terms, list) else []:
                if isinstance(term, dict) and term.get("source"):
                    print(
                        f"[cost-gen] grounding source ({term.get('feature')}): "
                        f"{term['source']}"
                    )
        payload = {
            "instruction": self.instruction,
            "backend": type(self).__name__,
            "model": self.model_name,
            "interpret": interpret,
            "ground": ground,
            "final": {
                "description": response.description,
                "params": response.params,
                "explanation": response.explanation,
                "recipient_explanation": response.recipient_explanation,
            },
            "ranking": ranking.as_json() if ranking is not None else None,
        }
        with open(self.run_dir / "rationale.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def install(self, cost: GeneratedPythonCost) -> None:
        """Append the generated cost to the planner's extra-cost set."""
        if self.mpc is None:
            return
        from uncertain_feedback.planners.mpc.costs.base import (  # pylint: disable=import-outside-toplevel
            CompositeTrajectoryCost,
        )

        existing = self.mpc._extra_costs  # pylint: disable=protected-access
        self.mpc.set_extra_costs(
            CompositeTrajectoryCost([*existing.terms(), cost])
        )

    def _write_validation(self, payload: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "validation.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _on_success(self, cost: GeneratedPythonCost, *, installed: bool) -> None:
        self._write_validation({"ok": True, "artifact_dir": str(self.run_dir)})
        action = "installed" if installed else "generated"
        print(f"[cost-gen] {action} cost: {cost.description}")
        print(f"[cost-gen] artifacts saved to: {self.run_dir}")

    def _on_failure(self, exc: Exception) -> None:
        self._write_validation(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "artifact_dir": str(self.run_dir),
            }
        )
        print(f"[cost-gen] failed to generate cost: {exc}")
        if self.strict:
            raise exc

    @abstractmethod
    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        """Generate (and optionally install) an MPC cost, or ``None`` on failure."""


def create_cost_generator(
    cfg: Any,
    context: GeneratedCostContext,
    instruction: str,
    *,
    summaries: dict[str, Any],
    run_dir: Path,
    images: dict[str, Path] | None = None,
    mpc: Any | None = None,
    llm_model_factory: Callable[[str], Any] = _make_llm_model,
    rollout_fn: Callable[[GeneratedPythonCost], np.ndarray | None] | None = None,
    eval_state: Any | None = None,
    save_candidate_videos: bool = False,
    corpus_dir: Path | None = None,
) -> CostGenerator:
    """Build the cost generator selected by ``cfg.backend``.

    ``cfg`` is an ``LlmCostConfig`` (see config.py); the per-backend params
    (``max_turns``, ``codex_cmd``) are read here. This is the only branch on the
    backend — callers always construct and call the result identically.
    """
    # Local imports keep the subclass modules from importing this one at import time.
    from uncertain_feedback.planners.mpc.costs.agent_costs import (  # pylint: disable=import-outside-toplevel
        AgentCostGenerator,
    )
    from uncertain_feedback.planners.mpc.costs.llm_costs import (  # pylint: disable=import-outside-toplevel
        LlmCostGenerator,
    )
    from uncertain_feedback.planners.mpc.costs.turns_costs import (  # pylint: disable=import-outside-toplevel
        TurnsCostGenerator,
    )

    common: dict[str, Any] = dict(
        context=context,
        instruction=instruction,
        summaries=summaries,
        run_dir=run_dir,
        images=images,
        use_images=cfg.use_images,
        model=cfg.model,
        strict=cfg.strict,
        mpc=mpc,
        llm_model_factory=llm_model_factory,
        rollout_fn=rollout_fn,
        eval_state=eval_state,
        save_candidate_videos=save_candidate_videos,
        corpus_dir=corpus_dir,
    )
    backend = cfg.backend
    if backend == "llm":
        return LlmCostGenerator(**common)
    if backend == "turns":
        return TurnsCostGenerator(max_turns=cfg.max_turns, **common)
    if backend == "agent":
        return AgentCostGenerator(codex_cmd=cfg.codex_cmd, **common)
    raise GeneratedCostValidationError(
        f"unknown cost-generator backend {backend!r}; "
        "expected one of 'llm', 'turns', 'agent'"
    )
