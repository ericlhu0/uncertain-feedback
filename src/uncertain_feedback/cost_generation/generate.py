"""The cost-generation stage: correction context in, generated cost out.

:func:`generate_cost_for_cluster` is the whole stage in one call — roll the
pre-correction reference, assemble the full corrected path, build the prompt
summaries and images, bundle the picklable :class:`EvalState`, then run whichever
backend the config selects. Callers (the fixed-pipeline orchestrator, the demo
runner, and later the per-stage CLIs) do not touch the backends directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.cost_generation.base import (
    CostGenerator,
    _make_llm_model,
    create_cost_generator,
)
from uncertain_feedback.cost_generation.summaries import (
    build_motion_summaries,
    render_prompt_images,
)
from uncertain_feedback.evaluation_mechanism import EvalState
from uncertain_feedback.planners.mpc import ArmMPC
from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedCostContext,
    GeneratedPythonCost,
    MpcCostContext,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.rollout import (
    assemble_full_correction_traj,
    make_cost_eval_rollout,
    rollout_reference_trajectory,
)


@dataclass
class CostGenerationResult:
    """Everything one cost-generation call produced, including its prompt artifacts."""

    cost_dir: Path
    generated_context: GeneratedCostContext
    reference_traj: np.ndarray | None
    full_correction_traj: np.ndarray
    summaries: dict[str, Any]
    images: dict[str, Path]
    eval_state: EvalState
    generated_cost: GeneratedPythonCost | None
    description: str = ""
    explanation: str = ""
    interpretation: str = ""
    grounding: str = ""


def _rationale_fields(
    cost_dir: Path, cost: GeneratedPythonCost | None
) -> dict[str, str]:
    """Read the generation evidence chain the combiner needs from ``rationale.json``.

    ``interpretation``/``grounding`` are the parsed stage-1/stage-2 objects
    ``save_rationale`` wrote, re-dumped as JSON text; all fields are empty when no
    rationale was produced (e.g. a failed generation).
    """
    description = cost.description if cost is not None else ""
    fields = {
        "description": description,
        "explanation": "",
        "interpretation": "",
        "grounding": "",
    }
    path = cost_dir / "rationale.json"
    if not path.exists():
        return fields
    data = json.loads(path.read_text(encoding="utf-8"))
    final = data.get("final") or {}
    fields["description"] = description or str(final.get("description") or "")
    fields["explanation"] = str(final.get("explanation") or "")
    for key, stage in (("interpretation", "interpret"), ("grounding", "ground")):
        value = data.get(stage)
        if value:
            fields[key] = json.dumps(value, indent=2, sort_keys=True)
    return fields


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _log(message: str, *, prefix: str) -> None:
    print(f"{prefix} {message}", flush=True)


def _rejected_candidate_trajs(
    candidate_trajs: dict[int, np.ndarray],
    selected_label: int,
    undesirable_labels: frozenset[int],
) -> tuple[np.ndarray, ...]:
    if selected_label in undesirable_labels:
        raise ValueError(
            f"Selected cluster label {selected_label} cannot be undesirable."
        )
    return tuple(
        trajectory
        for label, trajectory in sorted(candidate_trajs.items())
        if label in undesirable_labels
    )


def generate_cost_for_cluster(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: ArmMPC | None,
    cfg: MpcRunConfig,
    instruction: str,
    cluster_traj: np.ndarray,
    current_q: np.ndarray,
    q_history: list[np.ndarray],
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    cost_dir: Path,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    *,
    backend: str | None = None,
    candidate_trajs: dict[int, np.ndarray] | None = None,
    highlight_label: int | None = None,
    undesirable_labels: frozenset[int] = frozenset(),
    history_window: int | None = None,
    llm_model_factory: Callable[[str], Any] = _make_llm_model,
    install: bool = False,
    save_candidate_videos: bool = False,
    corpus_dir: Path | None = None,
    log_prefix: str = "[experiment]",
) -> CostGenerationResult:
    """Build one prompt context and generate one cost for one cluster/backend."""
    cfg_backend = (
        replace(cfg, llm_cost=replace(cfg.llm_cost, backend=backend))
        if backend is not None
        else cfg
    )
    window = cfg.preference_window if history_window is None else history_window
    phase_t0 = time.perf_counter()
    _log("phase C building cost-generation context", prefix=log_prefix)
    reference_q = rollout_reference_trajectory(
        cfg_backend,
        current_q,
        context,
        base_extra_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
    )
    goal_pos = (
        np.asarray(cfg_backend.cartesian.goals[0], dtype=np.float64)
        if cfg_backend.cartesian is not None
        else None
    )
    cartesian_threshold = (
        cfg_backend.cartesian.threshold if cfg_backend.cartesian is not None else 0.05
    )
    correction_q = canonical_arm_q(cluster_traj, context)
    full_correction_q = assemble_full_correction_traj(
        cfg_backend,
        q_history,
        correction_q,
        context,
        base_extra_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
    )
    rejected_trajs: tuple[np.ndarray, ...] = ()
    prompt_candidate_trajs = candidate_trajs
    if candidate_trajs:
        selected_label = (
            highlight_label
            if highlight_label is not None
            else next(iter(candidate_trajs))
        )
        rejected_trajs = _rejected_candidate_trajs(
            candidate_trajs,
            selected_label,
            undesirable_labels,
        )
        prompt_candidate_trajs = {
            label: trajectory
            for label, trajectory in candidate_trajs.items()
            if label == selected_label or label in undesirable_labels
        }
    generated_context = build_generated_cost_context(
        context,
        current_q,
        correction_q,
        q_history,
        window=window,
        body_pos=body_pos,
        reference_traj=reference_q,
        full_correction_traj=full_correction_q,
        cartesian_goal=goal_pos,
        cartesian_threshold=cartesian_threshold,
        rejected_trajs=rejected_trajs,
    )
    summaries = build_motion_summaries(generated_context, cartesian_goal=goal_pos)
    _log(f"cost-generation context ready in {_elapsed(phase_t0)}", prefix=log_prefix)

    images: dict[str, Path] = {}
    if cfg_backend.llm_cost.use_images:
        images_t0 = time.perf_counter()
        _log(
            f"rendering cost prompt images to {cost_dir / 'images'}", prefix=log_prefix
        )
        images = render_prompt_images(
            generated_context,
            cost_dir / "images",
            prompt_candidate_trajs,
            highlight_label,
            reference_traj=reference_q,
            goal_pos=goal_pos,
        )
        _log(
            f"rendered {len(images)} prompt image(s) in {_elapsed(images_t0)}",
            prefix=log_prefix,
        )

    eval_state = EvalState(
        cfg=cfg_backend,
        current_q=current_q,
        correction_traj=correction_q,
        q_history=q_history,
        window=window,
        cost_context=context,
        base_extra_costs=base_extra_costs,
        body_pos=body_pos,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        reference_traj=reference_q,
        full_correction_traj=full_correction_q,
        cartesian_goal=goal_pos,
        cartesian_threshold=cartesian_threshold,
        rejected_trajs=rejected_trajs,
    )
    generator: CostGenerator = create_cost_generator(
        cfg_backend.llm_cost,
        generated_context,
        instruction,
        summaries=summaries,
        run_dir=cost_dir,
        images=images,
        mpc=mpc,
        llm_model_factory=llm_model_factory,
        rollout_fn=make_cost_eval_rollout(
            cfg_backend,
            current_q,
            context,
            base_extra_costs,
            body_pos,
            spine3_pos,
            spine3_aa,
        ),
        eval_state=eval_state,
        save_candidate_videos=save_candidate_videos,
        corpus_dir=corpus_dir,
    )
    cost_t0 = time.perf_counter()
    _log(
        f"starting cost generation "
        f"(backend={cfg_backend.llm_cost.backend}, model={cfg_backend.llm_cost.model}, "
        f"artifacts={cost_dir})",
        prefix=log_prefix,
    )
    generated_cost = generator.generate(install=install)
    if generated_cost is None:
        _log(
            f"cost generation returned no cost in {_elapsed(cost_t0)}",
            prefix=log_prefix,
        )
    else:
        _log(
            f"cost generation finished in {_elapsed(cost_t0)}; "
            f"description={generated_cost.description!r}",
            prefix=log_prefix,
        )
    return CostGenerationResult(
        cost_dir=cost_dir,
        generated_context=generated_context,
        reference_traj=reference_q,
        full_correction_traj=full_correction_q,
        summaries=summaries,
        images=images,
        eval_state=eval_state,
        generated_cost=generated_cost,
        **_rationale_fields(cost_dir, generated_cost),
    )
