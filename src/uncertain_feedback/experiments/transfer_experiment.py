"""Simulated-user transfer evaluation for generated costs.

A :class:`~uncertain_feedback.simulated_users.SimulatedUser` with a hidden
comfort cost plays the care recipient: the planner rolls toward its original
goal, the user interrupts at the first hidden-cost violation, speaks their
persona's feedback line, picks the most comfortable UQ cluster, and a cost is
generated from that correction. Every evaluation condition is then rolled out
from the initial pose toward the original goal AND each held-out transfer goal,
and scored against the hidden cost the generator never saw — so the evaluation
signal is independent of the MDM correction used for generation, and the value
of the generated cost is measured by how it transfers to new goals.

Conditions:

- ``base``       — configured comfort costs only (no correction learned).
- ``tracking``   — the correction trajectory tracked directly, then comfort-only
                   continuation (``full_correction_traj``); defined only for the
                   original goal — on transfer goals there is no trajectory to
                   track, which is the point of persisting a cost instead.
- ``generated``  — comfort costs + the generated cost.
- ``oracle``     — comfort costs + the hidden cost itself (upper bound).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc import ArmMPCCartesianNoMDM, LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    EvalState,
    MpcCostContext,
    artifact_run_dir,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    render_prompt_images,
)
from uncertain_feedback.planners.run import (
    _assemble_full_correction_traj,
    _make_cost_eval_rollout,
    _rollout_reference_trajectory,
    run_planning_loop,
)
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    SimulatedUser,
    choose_cluster,
    compute_violations,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.utils.plot import ArmVisualizer


def _rollout_to_goal(
    cfg: MpcRunConfig,
    q0: np.ndarray,
    goal: np.ndarray,
    context: MpcCostContext,
    extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> np.ndarray:
    """Roll a headless Cartesian MPC from ``q0`` toward one goal; ``(T, 3, 3)``."""
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
    )
    q0 = np.asarray(q0, dtype=np.float64).copy()
    result = run_planning_loop(planner, q0, max(1, cfg.steps), stop_on_runtime_error=True)
    return np.asarray([q0, *result.q_history], dtype=np.float64)


def _goal_reach(
    context: MpcCostContext,
    cfg: MpcRunConfig,
    rollout: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    """Final spine3-relative wrist distance to ``goal`` vs the Cartesian threshold."""
    arm_pos = context.fk.fk(rollout[-1], context.spine3_pos, context.spine3_aa)
    wrist_rel = arm_pos[-1] - context.spine3_pos
    distance = float(np.linalg.norm(wrist_rel - np.asarray(goal, dtype=np.float64)))
    return {
        "reached": distance < cfg.cartesian.threshold,
        "distance": distance,
        "threshold": float(cfg.cartesian.threshold),
    }


def _evaluate_rollout(
    user: SimulatedUser,
    context: MpcCostContext,
    cfg: MpcRunConfig,
    rollout: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = violation_metrics(user, context, rollout)
    metrics["goal_reach"] = _goal_reach(context, cfg, rollout, goal)
    metrics["steps"] = int(rollout.shape[0] - 1)
    return metrics


def _save_rollout(
    rollout: np.ndarray,
    out_dir: Path,
    name: str,
    context: MpcCostContext,
    body_pos: np.ndarray | None,
    goal: np.ndarray,
    save_video: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", rollout)
    if save_video:
        ArmVisualizer(context.fk).render_rollout_video(
            rollout,
            out_dir / f"{name}.mp4",
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=body_pos,
            cartesian_goal=np.asarray(goal, dtype=np.float64),
        )


def _write_summary(root_dir: Path, summary: dict[str, Any]) -> None:
    with open(root_dir / "transfer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_transfer_experiment(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    mpc: LeftArmMPCMDMUQ,
    cfg: MpcRunConfig,
    user: SimulatedUser,
    gen: MotionGenerator,
    initial_pose: np.ndarray,
    initial_q: np.ndarray,
    context: MpcCostContext,
    artifact_base_dir: Path,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    mdm_frames: int | None = None,
    frozen_body: bool = False,
    save_video: bool = False,
) -> dict[str, Any]:
    """Run the full simulated-user protocol and write ``transfer_summary.json``."""
    root_dir = artifact_run_dir(artifact_base_dir, Path("transfer_artifacts"))
    root_dir.mkdir(parents=True, exist_ok=True)
    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access

    summary: dict[str, Any] = {
        "persona": user.name,
        "persona_description": user.description,
        "feedback_text": user.feedback_text,
        "trigger_threshold": cfg.transfer.trigger_threshold,
    }

    # Phase A: initial comfort-only rollout toward the original goal; the user
    # interrupts at the first hidden-cost violation.
    initial_traj = _rollout_reference_trajectory(
        cfg, initial_q, context, base_extra_costs, body_pos, spine3_pos, spine3_aa
    )
    if initial_traj is None:
        raise ValueError("Transfer experiment requires a Cartesian-goal planner.")
    trigger = first_violation_step(
        user, context, initial_traj, cfg.transfer.trigger_threshold
    )
    summary["trigger_step"] = trigger
    np.save(root_dir / "initial_rollout.npy", initial_traj)
    if trigger is None:
        summary["result"] = "no_violation"
        _write_summary(root_dir, summary)
        print(
            f"[transfer] {user.name}: initial plan never violates the hidden "
            "cost — no feedback to give."
        )
        return summary
    violations = compute_violations(user, context, initial_traj)
    summary["trigger_violation"] = float(violations[trigger])
    q_feedback = initial_traj[trigger]
    q_history = [np.asarray(q, dtype=np.float64) for q in initial_traj[: trigger + 1]]
    print(
        f"[transfer] {user.name} interrupts at step {trigger} "
        f"(violation {violations[trigger]:.3f} rad): '{user.feedback_text}'"
    )

    # Phase B: correction — the user speaks, MDM generates, the user picks the
    # most comfortable cluster.
    current_pose = gen.build_pose_from_arm_aa(initial_pose, q_feedback)
    correction_traj = mpc.query_mdm_with_uncertainty(
        gen,
        user.feedback_text,
        start_pose=current_pose,
        current_arm_aa=q_feedback,
        default_scale=cfg.uq.scale,
        mdm_frames=mdm_frames,
        frozen_body=frozen_body,
        cluster_selector=lambda means: choose_cluster(user, context, means),
    )
    uq_result = mpc.last_uq_result
    assert uq_result is not None
    summary["chosen_cluster"] = uq_result.chosen_label
    summary["cluster_violations"] = {
        str(label): float(np.mean(compute_violations(user, context, traj)))
        for label, traj in uq_result.cluster_means.items()
    }

    # Phase C: cost generation from the correction (identical inputs to a live
    # run; the hidden cost is not visible here).
    reference_traj = _rollout_reference_trajectory(
        cfg, q_feedback, context, base_extra_costs, body_pos, spine3_pos, spine3_aa
    )
    goal_pos = np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
    full_correction_traj = _assemble_full_correction_traj(
        cfg, q_history, correction_traj, context, base_extra_costs,
        body_pos, spine3_pos, spine3_aa,
    )
    generated_context = build_generated_cost_context(
        context, q_feedback, correction_traj, q_history,
        window=cfg.preference_window, body_pos=body_pos,
        reference_traj=reference_traj,
        full_correction_traj=full_correction_traj,
    )
    summaries = build_motion_summaries(generated_context, cartesian_goal=goal_pos)
    cost_dir = root_dir / "cost_generation"
    images: dict[str, Path] = {}
    if cfg.llm_cost.use_images:
        images = render_prompt_images(
            generated_context, cost_dir / "images",
            uq_result.cluster_means, uq_result.chosen_label,
            reference_traj=reference_traj, goal_pos=goal_pos,
        )
    eval_state = EvalState(
        cfg=cfg,
        current_q=q_feedback,
        correction_traj=correction_traj,
        q_history=q_history,
        window=cfg.preference_window,
        cost_context=context,
        base_extra_costs=base_extra_costs,
        body_pos=body_pos,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        reference_traj=reference_traj,
        full_correction_traj=full_correction_traj,
    )
    generator = create_cost_generator(
        cfg.llm_cost, generated_context, user.feedback_text,
        summaries=summaries, run_dir=cost_dir, images=images, mpc=None,
        rollout_fn=_make_cost_eval_rollout(
            cfg, q_feedback, context, base_extra_costs,
            body_pos, spine3_pos, spine3_aa,
        ),
        eval_state=eval_state,
    )
    generated_cost = generator.generate(install=False)
    summary["generated_cost"] = (
        {"description": generated_cost.description, "artifact_dir": str(cost_dir)}
        if generated_cost is not None
        else None
    )

    # Phase D: roll every condition out to the original goal and each held-out
    # transfer goal, scoring against the hidden cost.
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

    goals: list[tuple[str, np.ndarray]] = [("goal_0", goal_pos)]
    goals += [
        (f"transfer_{i}", np.asarray(g, dtype=np.float64))
        for i, g in enumerate(cfg.transfer.goals)
    ]
    results: dict[str, dict[str, Any]] = {name: {} for name in conditions}
    results["tracking"] = {}
    for goal_name, goal in goals:
        for cond_name, extra_costs in conditions.items():
            rollout = _rollout_to_goal(
                cfg, initial_q, goal, context, extra_costs,
                body_pos, spine3_pos, spine3_aa,
            )
            results[cond_name][goal_name] = _evaluate_rollout(
                user, context, cfg, rollout, goal
            )
            _save_rollout(
                rollout, root_dir / cond_name, goal_name, context,
                body_pos, goal, save_video,
            )
    # Tracking is only defined for the original goal: the executed path is the
    # pre-correction history + tracked correction + comfort-only continuation.
    results["tracking"]["goal_0"] = _evaluate_rollout(
        user, context, cfg, full_correction_traj, goal_pos
    )
    _save_rollout(
        full_correction_traj, root_dir / "tracking", "goal_0", context,
        body_pos, goal_pos, save_video,
    )
    for goal_name, _goal in goals[1:]:
        results["tracking"][goal_name] = {"same_as_base": True}

    summary["results"] = results
    summary["result"] = "ok"
    _write_summary(root_dir, summary)
    print(f"[transfer] artifacts saved to: {root_dir}")
    return summary
