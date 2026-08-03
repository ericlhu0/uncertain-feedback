"""Shared base for MPC cost-function generators.

Three sibling strategies generate an executable :class:`GeneratedPythonCost` from a
user correction, all constructed and called the same way:

- ``llm``   — single-turn LLM call
  (:class:`~uncertain_feedback.cost_generation.llm_costs.LlmCostGenerator`).
- ``turns`` — multi-turn LLM conversation that refines the cost
  (:class:`~uncertain_feedback.cost_generation.turns_costs.TurnsCostGenerator`).
- ``agent`` — delegates iteration to the ``codex`` CLI
  (:class:`~uncertain_feedback.cost_generation.agent_costs.AgentCostGenerator`).

:class:`CostGenerator` holds the shared inputs and helpers (prompt building, model
construction, response parsing/validation, artifact saving, installation) so each
subclass body stays small. :func:`create_cost_generator` selects the strategy from
config; this is the only place that branches on the backend.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.cost_generation.prompts import (
    build_author_prompt,
    build_ground_prompt,
    build_interpret_prompt,
)
from uncertain_feedback.evaluation_mechanism import (
    CostRanking,
    reference_with_correction_traj,
)
from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.costs import (
    GeneratedCostContext,
    GeneratedCostValidationError,
    GeneratedPythonCost,
    LlmCostResponse,
    extract_json_object,
    parse_llm_cost_response,
)

_SYSTEM_PROMPT = (
    "You generate safe, vectorized Python MPC trajectory cost functions. "
    "Return only the requested JSON object."
)
_DEFAULT_LLM_MODEL = "gpt-5.6-luna"
# Reasoning effort per model; a model absent here is sent without one.
_REASONING_EFFORT = {"gpt-5.6-luna": "high", "gpt-5.6-sol": "low"}


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


def parse_goal_conflict(interpret_text: str) -> bool:
    """Read the stage-1 ``goal_conflict`` flag; ``False`` if absent/unparseable."""
    data = extract_json_object(interpret_text)
    return bool(data is not None and data.get("goal_conflict", False))


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
        rollout_fn: Callable[[GeneratedPythonCost], np.ndarray | None] | None = None,
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
        if self.model:
            return self.model
        return os.getenv("OPENAI_MODEL", _DEFAULT_LLM_MODEL)

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
        cost = self.compile_cost(response.code, response.params, response.description)
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

        reference = canonical_arm_q(
            reference_with_correction_traj(self.context), self.context
        )
        if reference.ndim != 2 or reference.shape[0] == 0:
            return
        video_path = self.run_dir / "reference_with_correction.mp4"
        mdm_traj = canonical_arm_q(self.context.mdm_traj, self.context)
        mdm_goal_q = (
            mdm_traj[-1]
            if mdm_traj.ndim == 2 and mdm_traj.shape[0] > 0
            else reference[-1]
        )
        try:
            ArmVisualizer(self.context.fk).render_rollout_video(
                self.context.arm_aa(reference),
                video_path,
                spine3_pos=self.context.spine3_pos,
                spine3_aa=self.context.spine3_aa,
                body_pos=self.context.body_pos,
                mdm_goal_q=self.context.arm_aa(mdm_goal_q),
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
            (out / "explanation.txt").write_text(response.explanation, encoding="utf-8")
            print(f"[cost-gen] explanation: {response.explanation}")
        if response.recipient_explanation:
            (out / "recipient_explanation.txt").write_text(
                response.recipient_explanation, encoding="utf-8"
            )
            print(
                "[cost-gen] recipient explanation: " f"{response.recipient_explanation}"
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
        from uncertain_feedback.planners.mpc.costs import (  # pylint: disable=import-outside-toplevel
            CompositeTrajectoryCost,
        )

        existing = self.mpc._extra_costs  # pylint: disable=protected-access
        self.mpc.set_extra_costs(CompositeTrajectoryCost([*existing.terms(), cost]))

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
    from uncertain_feedback.cost_generation.agent_costs import (  # pylint: disable=import-outside-toplevel
        AgentCostGenerator,
    )
    from uncertain_feedback.cost_generation.llm_costs import (  # pylint: disable=import-outside-toplevel
        LlmCostGenerator,
    )
    from uncertain_feedback.cost_generation.turns_costs import (  # pylint: disable=import-outside-toplevel
        TurnsCostGenerator,
    )

    common: dict[str, Any] = {
        "context": context,
        "instruction": instruction,
        "summaries": summaries,
        "run_dir": run_dir,
        "images": images,
        "use_images": cfg.use_images,
        "model": cfg.model,
        "strict": cfg.strict,
        "mpc": mpc,
        "llm_model_factory": llm_model_factory,
        "rollout_fn": rollout_fn,
        "eval_state": eval_state,
        "save_candidate_videos": save_candidate_videos,
        "corpus_dir": corpus_dir,
    }
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
