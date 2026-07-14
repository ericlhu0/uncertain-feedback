"""Shared staged experiment pipeline for persona/backend evaluations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc import ArmMPCCartesianNoMDM, LeftArmMPCMDMUQ, SmplLeftArmMPC
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import UqClusterResult
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    CostGenerator,
    EvalState,
    GeneratedCostContext,
    GeneratedPythonCost,
    MpcCostContext,
    artifact_run_dir,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    render_prompt_images,
)
from uncertain_feedback.planners.mpc.costs.cost_generator import _make_llm_model
from uncertain_feedback.planners.run import (
    _assemble_full_correction_traj,
    _make_cost_eval_rollout,
    _rollout_reference_trajectory,
    run_planning_loop,
)
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    SimulatedUser,
    compute_violations,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory
from uncertain_feedback.utils.plot import ArmVisualizer


@dataclass
class InitialRolloutResult:
    initial_traj: np.ndarray
    trigger_step: int | None
    trigger_violation: float | None
    q_feedback: np.ndarray | None
    q_history: list[np.ndarray]


@dataclass
class UqCorrectionResult:
    correction_traj: np.ndarray
    uq_result: UqClusterResult
    cluster_oracle_scores: dict[int, float]


@dataclass
class CostGenerationResult:
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


@dataclass
class ExperimentResult:
    root_dir: Path
    summary: dict[str, Any]
    cfg: MpcRunConfig
    user: SimulatedUser
    base_extra_costs: CompositeTrajectoryCost
    initial: InitialRolloutResult
    correction: UqCorrectionResult | None
    cost_generation: CostGenerationResult | None
    generated_cost: GeneratedPythonCost | None
    goal_pos: np.ndarray | None


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _log(message: str, *, prefix: str) -> None:
    print(f"{prefix} {message}", flush=True)


def _write_summary(root_dir: Path, filename: str, summary: dict[str, Any]) -> None:
    with open(root_dir / filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def apply_persona_goals(cfg: MpcRunConfig, user_name: str) -> MpcRunConfig:
    """Return ``cfg`` with ``user`` and any per-persona goal override applied."""
    persona_goals = cfg.persona_goals.get(user_name)
    cfg = replace(cfg, user=user_name)
    if persona_goals is None:
        return cfg
    return replace(
        cfg,
        cartesian=replace(cfg.cartesian, goals=persona_goals.cartesian),
        transfer=replace(cfg.transfer, goals=persona_goals.transfer),
    )


def rollout_to_goal(
    cfg: MpcRunConfig,
    q0: np.ndarray,
    goal: np.ndarray,
    context: MpcCostContext,
    extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    *,
    progress_label: str | None = None,
    log_prefix: str = "[experiment]",
) -> np.ndarray:
    """Roll a headless Cartesian MPC from ``q0`` toward one goal."""
    planner = ArmMPCCartesianNoMDM(
        cartesian_goals=[np.asarray(goal, dtype=np.float64)],
        initial_arm_aa=q0,
        cartesian_threshold=cfg.cartesian.threshold,
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        goal_threshold=cfg.goal_threshold,
        visualize=False,
        fk=context.fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        body_pos=body_pos,
        extra_costs=extra_costs,
        seed=cfg.seed,
    )
    q0 = np.asarray(q0, dtype=np.float64).copy()

    def _progress(step: int, _q: np.ndarray, _q_history: list[np.ndarray]) -> None:
        if progress_label is not None and (step + 1) % 50 == 0:
            _log(
                f"{progress_label}: step {step + 1}/{max(1, cfg.steps)}",
                prefix=log_prefix,
            )

    result = run_planning_loop(
        planner,
        q0,
        max(1, cfg.steps),
        on_post_step=_progress if progress_label is not None else None,
        stop_on_runtime_error=True,
    )
    return np.asarray([q0, *result.q_history], dtype=np.float64)


def goal_reach(
    context: MpcCostContext,
    cfg: MpcRunConfig,
    rollout: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    """Final spine3-relative wrist distance to ``goal``."""
    arm_pos = context.fk.fk(rollout[-1], context.spine3_pos, context.spine3_aa)
    wrist_rel = arm_pos[-1] - context.spine3_pos
    distance = float(np.linalg.norm(wrist_rel - np.asarray(goal, dtype=np.float64)))
    return {
        "reached": distance < cfg.cartesian.threshold,
        "distance": distance,
        "threshold": float(cfg.cartesian.threshold),
    }


def evaluate_rollout(
    user: SimulatedUser,
    context: MpcCostContext,
    cfg: MpcRunConfig,
    rollout: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = violation_metrics(user, context, rollout)
    metrics["goal_reach"] = goal_reach(context, cfg, rollout, goal)
    metrics["steps"] = int(rollout.shape[0] - 1)
    return metrics


def save_rollout(
    rollout: np.ndarray,
    out_dir: Path,
    name: str,
    context: MpcCostContext,
    body_pos: np.ndarray | None,
    goal: np.ndarray,
    save_video: bool,
    *,
    log_prefix: str = "[experiment]",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", rollout)
    if save_video:
        video_path = out_dir / f"{name}.mp4"
        video_t0 = time.perf_counter()
        _log(f"rendering video: {video_path}", prefix=log_prefix)
        ArmVisualizer(context.fk).render_rollout_video(
            rollout,
            video_path,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=body_pos,
            cartesian_goal=np.asarray(goal, dtype=np.float64),
        )
        _log(f"rendered video: {video_path} in {_elapsed(video_t0)}", prefix=log_prefix)


def oracle_cluster_scores(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
    scale: float,
) -> dict[int, float]:
    oracle_cost = HiddenCostTerm(user=user, context=context)
    return {
        label: float(
            oracle_cost(
                np.expand_dims(
                    scale_trajectory(np.asarray(traj, dtype=np.float64), scale),
                    axis=0,
                )
            )[0]
        )
        for label, traj in cluster_means.items()
    }


def choose_oracle_cluster(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
    scale: float,
) -> tuple[int, dict[int, float]]:
    scores = oracle_cluster_scores(user, context, cluster_means, scale)
    chosen_label = min(sorted(scores), key=lambda label: scores[label])
    return chosen_label, scores


def run_initial_rollout(
    cfg: MpcRunConfig,
    user: SimulatedUser,
    initial_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    root_dir: Path,
    *,
    log_prefix: str = "[experiment]",
) -> InitialRolloutResult:
    phase_t0 = time.perf_counter()
    _log(
        f"{user.name}: phase A initial rollout "
        f"(steps={cfg.steps}, samples={cfg.n_mpc_samples})",
        prefix=log_prefix,
    )
    initial_traj = _rollout_reference_trajectory(
        cfg, initial_q, context, base_extra_costs, body_pos, spine3_pos, spine3_aa
    )
    if initial_traj is None:
        raise ValueError("Experiment requires a Cartesian-goal planner.")
    _log(
        f"{user.name}: initial rollout complete "
        f"({initial_traj.shape[0] - 1} steps) in {_elapsed(phase_t0)}",
        prefix=log_prefix,
    )
    _log(f"{user.name}: scoring initial rollout for hidden-cost trigger", prefix=log_prefix)
    trigger = first_violation_step(
        user, context, initial_traj, cfg.corrections.trigger_threshold
    )
    np.save(root_dir / "initial_rollout.npy", initial_traj)
    if trigger is None:
        return InitialRolloutResult(
            initial_traj=initial_traj,
            trigger_step=None,
            trigger_violation=None,
            q_feedback=None,
            q_history=[],
        )
    violations = compute_violations(user, context, initial_traj)
    q_feedback = initial_traj[trigger]
    q_history = [np.asarray(q, dtype=np.float64) for q in initial_traj[: trigger + 1]]
    trigger_violation = float(violations[trigger])
    _log(
        f"{user.name}: interrupts at step {trigger} "
        f"(violation {trigger_violation:.3f} rad): '{user.feedback_text}'",
        prefix=log_prefix,
    )
    return InitialRolloutResult(
        initial_traj=initial_traj,
        trigger_step=trigger,
        trigger_violation=trigger_violation,
        q_feedback=q_feedback,
        q_history=q_history,
    )


def generate_uq_correction(
    mpc: LeftArmMPCMDMUQ,
    cfg: MpcRunConfig,
    user: SimulatedUser,
    gen: MotionGenerator,
    initial_pose: np.ndarray,
    q_feedback: np.ndarray,
    context: MpcCostContext,
    root_dir: Path,
    body_pos: np.ndarray | None,
    *,
    mdm_frames: int | None,
    frozen_body: bool,
    log_prefix: str = "[experiment]",
) -> UqCorrectionResult:
    cluster_oracle_scores: dict[int, float] = {}

    def select_oracle_cluster(means: dict[int, np.ndarray]) -> int:
        nonlocal cluster_oracle_scores
        _log(f"{user.name}: oracle-scoring {len(means)} cluster mean(s)", prefix=log_prefix)
        chosen, cluster_oracle_scores = choose_oracle_cluster(
            user, context, means, cfg.uq.scale
        )
        scores = ", ".join(
            f"{label}={score:.3f}"
            for label, score in sorted(cluster_oracle_scores.items())
        )
        _log(f"{user.name}: oracle chose cluster {chosen} ({scores})", prefix=log_prefix)
        return chosen

    mdm_t0 = time.perf_counter()
    _log(
        f"{user.name}: phase B MDM/UQ correction "
        f"(samples={cfg.uq.diffusion_samples}, clusters={cfg.uq.n_clusters}, "
        f"frames={mdm_frames}, scale={cfg.uq.scale})",
        prefix=log_prefix,
    )
    current_pose = gen.build_pose_from_arm_aa(initial_pose, q_feedback)
    correction_traj = mpc.query_mdm_with_uncertainty(
        gen,
        user.feedback_text,
        start_pose=current_pose,
        current_arm_aa=q_feedback,
        default_scale=cfg.uq.scale,
        mdm_frames=mdm_frames,
        frozen_body=frozen_body,
        cluster_selector=select_oracle_cluster,
    )
    uq_result = mpc.last_uq_result
    if uq_result is None:
        raise RuntimeError("UQ clustering produced no result.")
    _log(
        f"{user.name}: MDM/UQ complete in {_elapsed(mdm_t0)}; "
        f"chosen_cluster={uq_result.chosen_label}, "
        f"correction_frames={correction_traj.shape[0]}",
        prefix=log_prefix,
    )
    if cfg.cartesian.goals:
        overlay_t0 = time.perf_counter()
        _log(f"{user.name}: rendering UQ cluster overlay", prefix=log_prefix)
        ArmVisualizer(context.fk).render_cluster_contrast_overlay(
            root_dir / "cluster_options.png",
            mdm_trajs=uq_result.cluster_means,
            highlight_label=uq_result.chosen_label,
            current_q=q_feedback,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=body_pos,
            goal_pos=np.asarray(cfg.cartesian.goals[0], dtype=np.float64),
            include_others=True,
            include_reference=False,
        )
        _log(
            f"{user.name}: wrote {root_dir / 'cluster_options.png'} "
            f"in {_elapsed(overlay_t0)}",
            prefix=log_prefix,
        )
    return UqCorrectionResult(
        correction_traj=correction_traj,
        uq_result=uq_result,
        cluster_oracle_scores=cluster_oracle_scores,
    )


def generate_cost_for_cluster(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: SmplLeftArmMPC | None,
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
    reference_traj = _rollout_reference_trajectory(
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
        if cfg_backend.cartesian.goals
        else None
    )
    full_correction_traj = _assemble_full_correction_traj(
        cfg_backend,
        q_history,
        cluster_traj,
        context,
        base_extra_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
    )
    generated_context = build_generated_cost_context(
        context,
        current_q,
        cluster_traj,
        q_history,
        window=window,
        body_pos=body_pos,
        reference_traj=reference_traj,
        full_correction_traj=full_correction_traj,
    )
    summaries = build_motion_summaries(generated_context, cartesian_goal=goal_pos)
    _log(f"cost-generation context ready in {_elapsed(phase_t0)}", prefix=log_prefix)

    images: dict[str, Path] = {}
    if cfg_backend.llm_cost.use_images:
        images_t0 = time.perf_counter()
        _log(f"rendering cost prompt images to {cost_dir / 'images'}", prefix=log_prefix)
        images = render_prompt_images(
            generated_context,
            cost_dir / "images",
            candidate_trajs,
            highlight_label,
            reference_traj=reference_traj,
            goal_pos=goal_pos,
        )
        _log(
            f"rendered {len(images)} prompt image(s) in {_elapsed(images_t0)}",
            prefix=log_prefix,
        )

    eval_state = EvalState(
        cfg=cfg_backend,
        current_q=current_q,
        correction_traj=cluster_traj,
        q_history=q_history,
        window=window,
        cost_context=context,
        base_extra_costs=base_extra_costs,
        body_pos=body_pos,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        reference_traj=reference_traj,
        full_correction_traj=full_correction_traj,
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
        rollout_fn=_make_cost_eval_rollout(
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
        _log(f"cost generation returned no cost in {_elapsed(cost_t0)}", prefix=log_prefix)
    else:
        _log(
            f"cost generation finished in {_elapsed(cost_t0)}; "
            f"description={generated_cost.description!r}",
            prefix=log_prefix,
        )
    return CostGenerationResult(
        cost_dir=cost_dir,
        generated_context=generated_context,
        reference_traj=reference_traj,
        full_correction_traj=full_correction_traj,
        summaries=summaries,
        images=images,
        eval_state=eval_state,
        generated_cost=generated_cost,
        **_rationale_fields(cost_dir, generated_cost),
    )


def evaluate_cost_conditions(  # pylint: disable=too-many-arguments
    cfg: MpcRunConfig,
    user: SimulatedUser,
    initial_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    generated_cost: GeneratedPythonCost | None,
    root_dir: Path,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    *,
    save_video: bool,
    q_history: list[np.ndarray] | None = None,
    log_prefix: str = "[experiment]",
) -> dict[str, dict[str, Any]]:
    """Evaluate base, hidden-cost oracle, and generated costs on the first goal.

    With ``q_history`` (the base rollout up to the feedback trigger), oracle and
    generated resume from the feedback pose — the preference is unknown until
    the user speaks — and are scored on the assembled prefix + continuation,
    like the tracking condition. Base always rolls from ``initial_q``.
    """
    if not cfg.cartesian.goals:
        raise ValueError("Experiment evaluation requires cartesian.goals.")
    goal_pos = np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
    _log(f"{user.name}: evaluating original goal (save_video={save_video})", prefix=log_prefix)
    conditions: dict[str, CompositeTrajectoryCost] = {
        "base": base_extra_costs,
        "oracle": CompositeTrajectoryCost(
            [*base_extra_costs.terms(), HiddenCostTerm(user=user, context=context)]
        ),
    }
    if generated_cost is not None:
        conditions["generated"] = CompositeTrajectoryCost(
            [*base_extra_costs.terms(), generated_cost]
        )
    else:
        _log(f"{user.name}: skipping generated condition because no cost was produced", prefix=log_prefix)

    results: dict[str, dict[str, Any]] = {name: {} for name in conditions}
    for cond_name, extra_costs in conditions.items():
        rollout_t0 = time.perf_counter()
        progress_label = f"{user.name} {cond_name}/goal_0"
        resume = q_history is not None and cond_name != "base"
        _log(
            f"{progress_label}: rolling"
            + (f" (resuming from feedback step {len(q_history) - 1})" if resume else ""),
            prefix=log_prefix,
        )
        rollout = rollout_to_goal(
            cfg,
            q_history[-1] if resume else initial_q,
            goal_pos,
            context,
            extra_costs,
            body_pos,
            spine3_pos,
            spine3_aa,
            progress_label=progress_label,
            log_prefix=log_prefix,
        )
        if resume:
            rollout = np.concatenate(
                [np.asarray(q_history[:-1], dtype=np.float64), rollout], axis=0
            )
        metrics = evaluate_rollout(user, context, cfg, rollout, goal_pos)
        results[cond_name]["goal_0"] = metrics
        reach = metrics["goal_reach"]
        _log(
            f"{progress_label}: done in {_elapsed(rollout_t0)} "
            f"steps={metrics['steps']} "
            f"mean_violation={metrics['mean_violation']:.3f} "
            f"max_violation={metrics['max_violation']:.3f} "
            f"goal_distance={reach['distance']:.3f} "
            f"reached={reach['reached']}",
            prefix=log_prefix,
        )
        save_rollout(
            rollout,
            root_dir / cond_name,
            "goal_0",
            context,
            body_pos,
            goal_pos,
            save_video,
            log_prefix=log_prefix,
        )

    return results


def evaluate_original_goal(  # pylint: disable=too-many-arguments
    cfg: MpcRunConfig,
    user: SimulatedUser,
    initial_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    generated_cost: GeneratedPythonCost | None,
    full_correction_traj: np.ndarray,
    root_dir: Path,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    *,
    save_video: bool,
    q_history: list[np.ndarray] | None = None,
    log_prefix: str = "[experiment]",
) -> dict[str, dict[str, Any]]:
    results = evaluate_cost_conditions(
        cfg,
        user,
        initial_q,
        context,
        base_extra_costs,
        generated_cost,
        root_dir,
        body_pos,
        spine3_pos,
        spine3_aa,
        save_video=save_video,
        q_history=q_history,
        log_prefix=log_prefix,
    )
    goal_pos = np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
    _log(f"{user.name} tracking/goal_0: scoring assembled correction trajectory", prefix=log_prefix)
    results["tracking"] = {}
    results["tracking"]["goal_0"] = evaluate_rollout(
        user, context, cfg, full_correction_traj, goal_pos
    )
    save_rollout(
        full_correction_traj,
        root_dir / "tracking",
        "goal_0",
        context,
        body_pos,
        goal_pos,
        save_video,
        log_prefix=log_prefix,
    )
    return results


def run_experiment(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: LeftArmMPCMDMUQ,
    cfg: MpcRunConfig,
    user: SimulatedUser,
    gen: MotionGenerator,
    initial_pose: np.ndarray,
    initial_q: np.ndarray,
    context: MpcCostContext,
    artifact_base_dir: Path,
    *,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    backend: str | None = None,
    mdm_frames: int | None = None,
    frozen_body: bool = False,
    save_video: bool = False,
    artifact_dir: Path | None = None,
    root_dir: Path | None = None,
    summary_filename: str = "experiment_summary.json",
    log_prefix: str = "[experiment]",
) -> ExperimentResult:
    """Run one persona/backend experiment on the original goal only."""
    cfg_backend = (
        replace(cfg, llm_cost=replace(cfg.llm_cost, backend=backend))
        if backend is not None
        else cfg
    )
    if root_dir is None:
        root_dir = artifact_run_dir(
            artifact_base_dir,
            artifact_dir if artifact_dir is not None else cfg_backend.llm_cost.artifact_dir,
        )
    root_dir.mkdir(parents=True, exist_ok=True)
    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access
    _log(f"{user.name}: artifacts -> {root_dir}", prefix=log_prefix)

    summary: dict[str, Any] = {
        "persona": user.name,
        "persona_description": user.description,
        "feedback_text": user.feedback_text,
        "backend": cfg_backend.llm_cost.backend,
        "trigger_threshold": cfg_backend.corrections.trigger_threshold,
    }

    initial = run_initial_rollout(
        cfg_backend,
        user,
        initial_q,
        context,
        base_extra_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
        root_dir,
        log_prefix=log_prefix,
    )
    summary["trigger_step"] = initial.trigger_step
    if initial.trigger_step is None or initial.q_feedback is None:
        summary["result"] = "no_violation"
        _write_summary(root_dir, summary_filename, summary)
        _log(
            f"{user.name}: initial plan never violates the hidden cost; no feedback to give",
            prefix=log_prefix,
        )
        return ExperimentResult(
            root_dir=root_dir,
            summary=summary,
            cfg=cfg_backend,
            user=user,
            base_extra_costs=base_extra_costs,
            initial=initial,
            correction=None,
            cost_generation=None,
            generated_cost=None,
            goal_pos=None,
        )

    summary["trigger_violation"] = initial.trigger_violation
    correction = generate_uq_correction(
        mpc,
        cfg_backend,
        user,
        gen,
        initial_pose,
        initial.q_feedback,
        context,
        root_dir,
        body_pos,
        mdm_frames=mdm_frames,
        frozen_body=frozen_body,
        log_prefix=log_prefix,
    )
    uq_result = correction.uq_result
    summary["chosen_cluster"] = uq_result.chosen_label
    summary["cluster_selection_method"] = "oracle_raw_cluster"
    summary["cluster_oracle_scores"] = {
        str(label): float(score)
        for label, score in sorted(correction.cluster_oracle_scores.items())
    }
    summary["cluster_violations"] = {
        str(label): float(np.mean(compute_violations(user, context, traj)))
        for label, traj in uq_result.cluster_means.items()
    }

    cost_generation = generate_cost_for_cluster(
        mpc=None,
        cfg=cfg_backend,
        instruction=user.feedback_text,
        cluster_traj=correction.correction_traj,
        current_q=initial.q_feedback,
        q_history=initial.q_history,
        context=context,
        base_extra_costs=base_extra_costs,
        cost_dir=root_dir / "cost_generation",
        body_pos=body_pos,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        backend=cfg_backend.llm_cost.backend,
        candidate_trajs=uq_result.cluster_means,
        highlight_label=uq_result.chosen_label,
        install=False,
        save_candidate_videos=save_video,
        log_prefix=log_prefix,
    )
    generated_cost = cost_generation.generated_cost
    summary["generated_cost"] = (
        {
            "description": generated_cost.description,
            "artifact_dir": str(cost_generation.cost_dir),
        }
        if generated_cost is not None
        else None
    )

    results = evaluate_original_goal(
        cfg_backend,
        user,
        initial_q,
        context,
        base_extra_costs,
        generated_cost,
        cost_generation.full_correction_traj,
        root_dir,
        body_pos,
        spine3_pos,
        spine3_aa,
        save_video=save_video,
        q_history=initial.q_history,
        log_prefix=log_prefix,
    )
    summary["results"] = results
    summary["result"] = "ok"
    _write_summary(root_dir, summary_filename, summary)
    _log(f"{user.name}: artifacts saved to {root_dir}", prefix=log_prefix)
    goal_pos = (
        np.asarray(cfg_backend.cartesian.goals[0], dtype=np.float64)
        if cfg_backend.cartesian.goals
        else None
    )
    return ExperimentResult(
        root_dir=root_dir,
        summary=summary,
        cfg=cfg_backend,
        user=user,
        base_extra_costs=base_extra_costs,
        initial=initial,
        correction=correction,
        cost_generation=cost_generation,
        generated_cost=generated_cost,
        goal_pos=goal_pos,
    )
