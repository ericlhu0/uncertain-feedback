"""Simulated-user transfer evaluation for generated costs.

The original-goal feedback loop is delegated to
``experiment_pipeline.run_experiment``. This module adds only the held-out
``transfer.goals`` rollouts that measure whether the generated cost generalizes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.experiments.experiment_pipeline import (
    choose_oracle_cluster,
    evaluate_rollout,
    rollout_to_goal,
    run_experiment,
    save_rollout,
)
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost, MpcCostContext
from uncertain_feedback.simulated_users import HiddenCostTerm, SimulatedUser


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _log(message: str) -> None:
    print(f"[transfer] {message}", flush=True)


def _write_summary(root_dir: Path, summary: dict[str, Any]) -> None:
    with open(root_dir / "transfer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _choose_oracle_cluster(
    user: SimulatedUser,
    context: MpcCostContext,
    cluster_means: dict[int, np.ndarray],
    scale: float,
) -> tuple[int, dict[int, float]]:
    """Compatibility wrapper for tests and older imports."""
    return choose_oracle_cluster(user, context, cluster_means, scale)


def run_transfer_experiment(  # pylint: disable=too-many-arguments,too-many-locals
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
    """Run the original-goal experiment, then evaluate held-out transfer goals."""
    result = run_experiment(
        mpc,
        cfg,
        user,
        gen,
        initial_pose,
        initial_q,
        context,
        artifact_base_dir,
        body_pos=body_pos,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        mdm_frames=mdm_frames,
        frozen_body=frozen_body,
        save_video=save_video,
        artifact_dir=Path("transfer_artifacts"),
        summary_filename="transfer_summary.json",
        log_prefix="[transfer]",
    )
    summary = result.summary
    if summary.get("result") != "ok":
        return summary

    transfer_goals = [
        (f"transfer_{i}", np.asarray(g, dtype=np.float64))
        for i, g in enumerate(cfg.transfer.goals)
    ]
    if not transfer_goals:
        _write_summary(result.root_dir, summary)
        return summary

    cfg = result.cfg
    conditions: dict[str, CompositeTrajectoryCost] = {
        "base": result.base_extra_costs,
        "oracle": CompositeTrajectoryCost(
            [*result.base_extra_costs.terms(), HiddenCostTerm(user=user, context=context)]
        ),
    }
    if result.generated_cost is not None:
        conditions["generated"] = CompositeTrajectoryCost(
            [*result.base_extra_costs.terms(), result.generated_cost]
        )
    else:
        _log(f"{user.name}: skipping generated transfer condition; no cost was produced")

    results = summary.setdefault("results", {})
    for cond_name in conditions:
        results.setdefault(cond_name, {})
    results.setdefault("tracking", {})

    _log(
        f"{user.name}: evaluating {len(transfer_goals)} held-out transfer goal(s) "
        f"(save_video={save_video})"
    )
    for goal_name, goal in transfer_goals:
        goal_text = np.array2string(goal, precision=3, suppress_small=True)
        _log(f"{user.name}: evaluating {goal_name} target={goal_text}")
        for cond_name, extra_costs in conditions.items():
            rollout_t0 = time.perf_counter()
            progress_label = f"{user.name} {cond_name}/{goal_name}"
            _log(f"{progress_label}: rolling")
            rollout = rollout_to_goal(
                cfg,
                initial_q,
                goal,
                context,
                extra_costs,
                body_pos,
                spine3_pos,
                spine3_aa,
                progress_label=progress_label,
                log_prefix="[transfer]",
            )
            metrics = evaluate_rollout(user, context, cfg, rollout, goal)
            results[cond_name][goal_name] = metrics
            reach = metrics["goal_reach"]
            _log(
                f"{progress_label}: done in {_elapsed(rollout_t0)} "
                f"steps={metrics['steps']} "
                f"mean_violation={metrics['mean_violation']:.3f} "
                f"max_violation={metrics['max_violation']:.3f} "
                f"goal_distance={reach['distance']:.3f} "
                f"reached={reach['reached']}"
            )
            save_rollout(
                rollout,
                result.root_dir / cond_name,
                goal_name,
                context,
                body_pos,
                goal,
                save_video,
                log_prefix="[transfer]",
            )
        results["tracking"][goal_name] = {"same_as_base": True}

    summary["results"] = results
    _write_summary(result.root_dir, summary)
    _log(f"{user.name}: transfer artifacts saved to {result.root_dir}")
    return summary
