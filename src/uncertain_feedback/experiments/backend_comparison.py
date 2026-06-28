"""Per-backend cost-generation comparison experiment.

Given a single MDM correction (the chosen UQ cluster mean), generate one cost
with each backend (``llm`` / ``turns`` / ``agent``) and score them all on the
**same axis**: the real rollout-vs-MDM FK-position L2 metric
(:func:`evaluate_candidate_cost`). Because ``llm`` never self-evaluates, the
experiment re-scores every backend's *final* cost itself rather than reading the
backends' internal ``score.json`` — that is the only way the three land on a
comparable axis. Writes per-backend artifacts plus a ``backend_comparison.json``
ranking.

This is the backend-axis counterpart to
:mod:`uncertain_feedback.experiments.cluster_comparison`, which instead holds the
backend fixed and varies the UQ cluster.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uncertain_feedback.planners.mpc import SmplLeftArmMPC
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    MpcCostContext,
    artifact_run_dir,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    evaluate_candidate_cost,
    render_prompt_images,
)
from uncertain_feedback.experiments.cluster_comparison import (
    _read_json_if_exists,
    _render_rollout_video,
    _run_cluster_comparison_rollout,
    _run_initial_state_rollout,
)
from uncertain_feedback.planners.run import (
    _make_cost_eval_rollout,
    _rollout_reference_trajectory,
)


def _write_backend_summary(root_dir: Path, summary: dict[str, Any]) -> None:
    with open(root_dir / "backend_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _finite_or_none(score: float | None) -> float | None:
    """JSON-safe score: drop non-finite (failed / no-rollout) values to ``null``."""
    if score is None or not math.isfinite(score):
        return None
    return score


def run_backend_comparison(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: SmplLeftArmMPC,
    cfg: MpcRunConfig,
    instruction: str,
    correction_traj: np.ndarray,
    current_q: np.ndarray,
    q_history: list[np.ndarray],
    context: MpcCostContext,
    artifact_base_dir: Path,
    history_window: int,
    rollout_steps: int,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    initial_q: np.ndarray | None = None,
    backends: Sequence[str] = ("llm", "turns", "agent"),
    save_video: bool = False,
) -> None:
    """Generate one cost per backend for a single correction and score them uniformly.

    Every backend's final cost is re-scored with the same
    :func:`evaluate_candidate_cost` rollout closure, so ``llm`` (which never
    self-evaluates) is comparable to ``turns`` / ``agent``. Writes a
    ``backend_comparison.json`` with a ranking (ascending score; lower is closer
    to the correction). Backends that fail to produce a cost are recorded and
    skipped rather than aborting the run.
    """
    root_dir = artifact_run_dir(artifact_base_dir, cfg.llm_cost.artifact_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access
    rollout_steps = max(1, rollout_steps)

    # Correction-independent across backends: compute once and share.
    reference_traj = _rollout_reference_trajectory(
        cfg, current_q, context, base_extra_costs, body_pos, spine3_pos, spine3_aa,
    )
    goal_pos = (
        np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
        if reference_traj is not None and cfg.cartesian.goals
        else None
    )
    generated_context = build_generated_cost_context(
        context, current_q, correction_traj, q_history, window=history_window,
        body_pos=body_pos, reference_traj=reference_traj,
    )
    summaries = build_motion_summaries(generated_context, cartesian_goal=goal_pos)
    images: dict[str, Path] = {}
    if cfg.llm_cost.use_images:
        images = render_prompt_images(
            generated_context, root_dir / "images",
            reference_traj=reference_traj, goal_pos=goal_pos,
        )

    rollout_fn = _make_cost_eval_rollout(
        cfg, current_q, context, base_extra_costs, body_pos, spine3_pos, spine3_aa,
    )

    cartesian_goal = (
        np.asarray(cfg.cartesian.goals[0]) if cfg.cartesian.goals else None
    )

    summary: dict[str, Any] = {
        "instruction": instruction,
        "backends": list(backends),
        "note": (
            "Scores are single draws from the non-deterministic sampling MPC: "
            "re-rolling the same cost varies by ~5%. When backends' scores are "
            "within that noise band the ranking order is not meaningful — treat "
            "them as tied. For a reliable ordering, average several rollouts per "
            "cost."
        ),
        "results": {},
        "ranking": [],
    }

    for backend in backends:
        backend_dir = root_dir / backend
        cfg_backend = dataclasses.replace(
            cfg, llm_cost=dataclasses.replace(cfg.llm_cost, backend=backend)
        )
        generator = create_cost_generator(
            cfg_backend.llm_cost, generated_context, instruction,
            summaries=summaries, run_dir=backend_dir, images=images, mpc=mpc,
            rollout_fn=rollout_fn,
        )
        generated = generator.generate(install=False)
        validation = _read_json_if_exists(backend_dir / "validation.json")
        params = _read_json_if_exists(backend_dir / "params.json")
        entry: dict[str, Any] = {
            "artifact_path": str(backend_dir),
            "generated": generated is not None,
            "score": None,
            "validation": validation,
            "description": (
                params.get("description") if isinstance(params, dict) else None
            ),
            "explanation": (
                params.get("explanation") if isinstance(params, dict) else None
            ),
            "recipient_explanation": (
                params.get("recipient_explanation")
                if isinstance(params, dict)
                else None
            ),
            "params": params.get("params") if isinstance(params, dict) else None,
            "rollout_metrics": None,
        }
        if generated is None:
            summary["results"][backend] = entry
            _write_backend_summary(root_dir, summary)
            continue

        score = evaluate_candidate_cost(generated_context, generated, rollout_fn)
        entry["score"] = _finite_or_none(score)
        print(f"[backend-compare] {backend}: score={score:.4f}")

        if save_video:
            rollout, metrics, colors = _run_cluster_comparison_rollout(
                cfg, current_q, correction_traj, context, base_extra_costs,
                generated, rollout_steps, body_pos, spine3_pos, spine3_aa,
            )
            np.save(backend_dir / "rollout.npy", rollout)
            entry["rollout_metrics"] = metrics
            n_frames = correction_traj.shape[0]
            cutoff = max(1, round(n_frames * cfg.trajectory_fraction))
            _render_rollout_video(
                rollout, context.fk, spine3_pos, spine3_aa,
                backend_dir / "rollout.mp4",
                body_pos=body_pos, cartesian_goal=cartesian_goal,
                frame_colors=colors, mdm_goal_q=correction_traj[cutoff - 1],
            )
            if initial_q is not None and cfg.planner in (
                "arm_mpc_cartesian", "arm_mpc_cartesian_no_mdm"
            ):
                is_rollout, is_metrics = _run_initial_state_rollout(
                    cfg, initial_q, context, base_extra_costs, generated,
                    rollout_steps, body_pos, spine3_pos, spine3_aa,
                )
                np.save(backend_dir / "initial_state_rollout.npy", is_rollout)
                entry["initial_state_metrics"] = is_metrics
                _render_rollout_video(
                    is_rollout, context.fk, spine3_pos, spine3_aa,
                    backend_dir / "initial_state_rollout.mp4",
                    body_pos=body_pos, cartesian_goal=cartesian_goal,
                )

        summary["results"][backend] = entry
        _write_backend_summary(root_dir, summary)

    summary["ranking"] = sorted(
        (
            {"backend": b, "score": summary["results"][b]["score"]}
            for b in summary["results"]
        ),
        key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0),
    )
    _write_backend_summary(root_dir, summary)
    print(f"[backend-compare] comparison artifacts saved to: {root_dir}")
