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
    parse_llm_cost_response,
)
from uncertain_feedback.planners.mpc.costs.prompts import build_llm_cost_prompt

_SYSTEM_PROMPT = (
    "You generate safe, vectorized Python MPC trajectory cost functions. "
    "Return only the requested JSON object."
)


def _make_llm_model(model_name: str) -> Any:
    """Build the cost-generator LLM wrapper."""
    from uncertain_feedback.llm import (  # pylint: disable=import-outside-toplevel
        OpenAIModel,
    )

    return OpenAIModel(
        model=model_name,
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=1800,
    )


def artifact_run_dir(base_dir: Path, artifact_dir: Path) -> Path:
    """Return a unique, timestamped artifact directory for one generation."""
    root = artifact_dir if artifact_dir.is_absolute() else base_dir / artifact_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def _resample_trajectory(traj: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a ``(T, ...)`` trajectory to ``n`` frames along axis 0."""
    t = traj.shape[0]
    if t == n:
        return traj
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, n)
    flat = traj.reshape(t, -1)
    out = np.stack(
        [np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1
    )
    return out.reshape((n, *traj.shape[1:]))


def _score_rollout(
    context: GeneratedCostContext, rollout: np.ndarray
) -> float:
    """Mean per-frame FK L2 distance between a rollout and the MDM correction.

    Resamples ``rollout`` to the corrected (MDM) trajectory's length and compares
    FK joint positions. ``math.inf`` when either trajectory is empty / malformed.
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
    rollout = _resample_trajectory(rollout, target.shape[0])
    rollout_positions = context.fk_batch(rollout)  # (T, 5, 3)
    mdm_positions = context.mdm_positions  # (T, 5, 3)
    return float(np.linalg.norm(rollout_positions - mdm_positions, axis=-1).mean())


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
        visualizer.render_joint_angle_comparison(
            angle_path,
            target_series=build_joint_angle_series(context, correction_traj),
            rollout_series=build_joint_angle_series(context, rollout),
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
        prompt: str = "2",
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
    ) -> None:
        self.context = context
        self.instruction = instruction
        self.summaries = summaries
        self.run_dir = run_dir
        self.prompt = prompt
        self.images = images or {}
        self.use_images = use_images
        self.model = model
        self.strict = strict
        self.mpc = mpc
        self.llm_model_factory = llm_model_factory
        self.rollout_fn = rollout_fn
        self.eval_state = eval_state
        self.save_candidate_videos = save_candidate_videos

    # -- shared helpers -----------------------------------------------------

    @property
    def model_name(self) -> str:
        """Resolved model name (config, then env, then default)."""
        return self.model or os.getenv("OPENAI_MODEL", "gpt-5.4")

    def make_llm(self) -> Any:
        """Construct the cost-generator LLM wrapper for this run's model."""
        return self.llm_model_factory(self.model_name)

    def build_prompt(self) -> tuple[str, list[str] | None]:
        """Build the prompt text and the list of image paths to attach."""
        text, attached = build_llm_cost_prompt(
            self.instruction, self.summaries, self.images, prompt=self.prompt
        )
        image_input = [str(path) for path in attached] or None
        return text, image_input

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
        return response, cost

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
    from uncertain_feedback.planners.mpc.costs.staged_costs import (  # pylint: disable=import-outside-toplevel
        StagedCostGenerator,
    )
    from uncertain_feedback.planners.mpc.costs.turns_costs import (  # pylint: disable=import-outside-toplevel
        TurnsCostGenerator,
    )

    common: dict[str, Any] = dict(
        context=context,
        instruction=instruction,
        summaries=summaries,
        run_dir=run_dir,
        prompt=cfg.prompt,
        images=images,
        use_images=cfg.use_images,
        model=cfg.model,
        strict=cfg.strict,
        mpc=mpc,
        llm_model_factory=llm_model_factory,
        rollout_fn=rollout_fn,
        eval_state=eval_state,
        save_candidate_videos=save_candidate_videos,
    )
    backend = cfg.backend
    if backend == "llm":
        return LlmCostGenerator(**common)
    if backend == "staged":
        return StagedCostGenerator(**common)
    if backend == "turns":
        return TurnsCostGenerator(max_turns=cfg.max_turns, **common)
    if backend == "agent":
        return AgentCostGenerator(codex_cmd=cfg.codex_cmd, **common)
    raise GeneratedCostValidationError(
        f"unknown cost-generator backend {backend!r}; "
        "expected one of 'llm', 'staged', 'turns', 'agent'"
    )
