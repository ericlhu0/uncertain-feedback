"""Per-cluster LLM-cost comparison experiment.

Given a UQ clustering of an MDM correction, generate one LLM-authored cost per
cluster, roll each one out headlessly with sampling MPC, and write per-cluster
metrics plus a ``comparison_summary.json``. This is the "multiple runs"
machinery that used to live inside ``planners/run.py``; it now reuses the single
stepping primitive :func:`uncertain_feedback.planners.run.run_planning_loop` so a
cluster rollout steps exactly like the live single run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from uncertain_feedback.planners.mpc import (
    ArmMPCCartesianNoMDM,
    LeftArmMPCCartesian,
    LeftArmMPCMDM,
    SmplLeftArmFK,
    SmplLeftArmMPC,
)
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import UqClusterResult
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    MpcCostContext,
)
from uncertain_feedback.utils.plot import ArmVisualizer
from uncertain_feedback.planners.mpc.costs import (
    artifact_run_dir,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    render_prompt_images,
)
from uncertain_feedback.planners.mpc.costs.cost_generator import _make_llm_model
from uncertain_feedback.planners.run import (
    _append_extra_cost,
    _rollout_reference_trajectory,
    run_planning_loop,
)


def _planner_frame_color(planner: SmplLeftArmMPC) -> str:
    """Return the arm color for the current planner state (MDM vs Cartesian)."""
    if isinstance(planner, LeftArmMPCCartesian) and not planner.mdm_tracking_complete:
        return ArmVisualizer.MDM_COLOR
    return ArmVisualizer.TARGET_COLOR


def _build_cluster_rollout_planner(  # pylint: disable=too-many-arguments
    cfg: MpcRunConfig,
    current_q: np.ndarray,
    cluster_traj: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    generated_cost: GeneratedPythonCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> SmplLeftArmMPC:
    """Build an isolated headless planner for one cluster comparison rollout."""
    extra_costs = _append_extra_cost(base_extra_costs, generated_cost)
    common: dict[str, Any] = {
        "horizon": cfg.horizon,
        "n_mpc_samples": cfg.n_mpc_samples,
        "max_angle_delta": cfg.max_angle_delta,
        "goal_threshold": cfg.goal_threshold,
        "visualize": False,
        "fk": context.fk,
        "spine3_pos": spine3_pos,
        "spine3_aa": spine3_aa,
        "body_pos": body_pos,
        "extra_costs": extra_costs,
    }
    if cfg.planner == "arm_mpc_cartesian":
        planner: SmplLeftArmMPC = LeftArmMPCCartesian(
            cartesian_goals=[
                np.asarray(g, dtype=np.float64) for g in cfg.cartesian.goals
            ],
            initial_arm_aa=current_q,
            cartesian_threshold=cfg.cartesian.threshold,
            **common,
            advance_threshold=cfg.advance_threshold,
            max_playback_delta=cfg.max_playback_delta,
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )
    else:
        planner = LeftArmMPCMDM(
            **common,
            goals=[],
            advance_threshold=cfg.advance_threshold,
            max_playback_delta=cfg.max_playback_delta,
            trajectory_fraction=cfg.trajectory_fraction,
        )

    mdm_planner = cast(LeftArmMPCMDM, planner)
    n_frames = cluster_traj.shape[0]
    cutoff = max(1, round(n_frames * mdm_planner.trajectory_fraction))
    mdm_planner.set_mdm_goal(cluster_traj[cutoff - 1])
    mdm_planner.push_trajectory(cluster_traj[:cutoff])
    return planner


def _rollout_metrics(
    rollout: np.ndarray,
    mpc: SmplLeftArmMPC,
    context: MpcCostContext,
) -> dict[str, Any]:
    deltas = np.diff(rollout, axis=0)
    path_length = float(
        np.linalg.norm(deltas.reshape(deltas.shape[0], -1), axis=1).sum()
    )
    final_q = rollout[-1]
    positions = context.fk.fk(final_q, context.spine3_pos, context.spine3_aa)
    metrics: dict[str, Any] = {
        "path_length_joint_l2": path_length,
        "final_q_norm": float(np.linalg.norm(final_q)),
        "final_wrist_position": positions[-1].tolist(),
    }
    if mpc.current_goal is not None:
        metrics["final_goal_distance"] = float(
            np.linalg.norm(final_q - mpc.current_goal)
        )
    return metrics


def _render_rollout_video(
    rollout: np.ndarray,
    fk: SmplLeftArmFK,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    save_path: Path,
    body_pos: np.ndarray | None = None,
    cartesian_goal: np.ndarray | None = None,
    frame_colors: list[str] | None = None,
    mdm_goal_q: np.ndarray | None = None,
    fps: int = 20,
) -> None:
    """Render a ``(steps+1, 3, 3)`` rollout to a video via the shared visualizer.

    Uses the same ``ArmVisualizer`` 6-panel layout as the live window, so a saved
    rollout looks identical to watching it live — only the ``save_video`` flag in
    :func:`run_cluster_comparison` decides whether this is called.
    """
    ArmVisualizer(fk).render_rollout_video(
        rollout, save_path,
        spine3_pos=spine3_pos, spine3_aa=spine3_aa,
        body_pos=body_pos, cartesian_goal=cartesian_goal,
        frame_colors=frame_colors, mdm_goal_q=mdm_goal_q, fps=fps,
    )


def _run_cluster_comparison_rollout(  # pylint: disable=too-many-arguments
    cfg: MpcRunConfig,
    current_q: np.ndarray,
    cluster_traj: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    generated_cost: GeneratedPythonCost,
    rollout_steps: int,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Run a headless MPC rollout with one cluster's generated cost."""
    comparison_mpc = _build_cluster_rollout_planner(
        cfg, current_q, cluster_traj, context, base_extra_costs,
        generated_cost, body_pos, spine3_pos, spine3_aa,
    )
    frame_colors = [_planner_frame_color(comparison_mpc)]

    def _collect_color(_step: int, _q: np.ndarray, _hist: list[np.ndarray]) -> None:
        frame_colors.append(_planner_frame_color(comparison_mpc))

    result = run_planning_loop(
        comparison_mpc, current_q, rollout_steps,
        on_post_step=_collect_color, stop_on_runtime_error=True,
    )
    q0 = np.asarray(current_q, dtype=np.float64).copy()
    rollout = np.asarray([q0, *result.q_history], dtype=np.float64)
    metrics = _rollout_metrics(rollout, comparison_mpc, context)
    metrics["steps_requested"] = rollout_steps
    metrics["steps_completed"] = int(max(0, rollout.shape[0] - 1))
    if result.error is not None:
        metrics["error"] = result.error
    return rollout, metrics, frame_colors


def _run_initial_state_rollout(  # pylint: disable=too-many-arguments
    cfg: MpcRunConfig,
    initial_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    generated_cost: GeneratedPythonCost,
    rollout_steps: int,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run MPC from the true initial state with one cluster's cost and the Cartesian goal."""
    extra_costs = _append_extra_cost(base_extra_costs, generated_cost)
    planner = ArmMPCCartesianNoMDM(
        cartesian_goals=[np.asarray(g, dtype=np.float64) for g in cfg.cartesian.goals],
        initial_arm_aa=initial_q,
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
    result = run_planning_loop(
        planner, initial_q, rollout_steps, stop_on_runtime_error=True
    )
    q0 = np.asarray(initial_q, dtype=np.float64).copy()
    rollout = np.asarray([q0, *result.q_history], dtype=np.float64)
    metrics = _rollout_metrics(rollout, planner, context)
    metrics["steps_requested"] = rollout_steps
    metrics["steps_completed"] = int(max(0, rollout.shape[0] - 1))
    if result.error is not None:
        metrics["error"] = result.error
    return rollout, metrics


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_comparison_summary(root_dir: Path, summary: dict[str, Any]) -> None:
    with open(root_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_cluster_comparison(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: SmplLeftArmMPC,
    cfg: MpcRunConfig,
    instruction: str,
    uq_result: UqClusterResult,
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
    llm_model_factory: Callable[[str], Any] = _make_llm_model,
    install: bool = True,
    root_dir: Path | None = None,
    save_video: bool = False,
) -> GeneratedPythonCost | None:
    """Generate one LLM cost per UQ cluster and run headless comparison rollouts.

    Runs synchronously: every cluster's cost is generated, rolled out for
    ``rollout_steps`` steps, scored, and (optionally) rendered to a video.
    Returns the chosen cluster's cost, installed on ``mpc`` when ``install``.
    """
    if root_dir is None:
        root_dir = artifact_run_dir(artifact_base_dir, cfg.llm_cost.artifact_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "selected_cluster.txt").write_text(
        f"{uq_result.chosen_label}\n", encoding="utf-8"
    )

    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access
    rollout_steps = max(1, rollout_steps)

    # The original-goal reference is cluster-independent — compute it once (no MDM
    # correction, no generated cost) and attach it to every cluster's generation so
    # each cost can be kept from blocking the goal the arm was driving toward.
    reference_traj = _rollout_reference_trajectory(
        cfg, current_q, context, base_extra_costs, body_pos, spine3_pos, spine3_aa,
    )
    goal_pos = (
        np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
        if reference_traj is not None and cfg.cartesian.goals
        else None
    )

    summary: dict[str, Any] = {
        "selected_cluster": uq_result.chosen_label,
        "cluster_ids": sorted(int(label) for label in uq_result.cluster_means),
        "clusters": {},
    }
    selected_cost: GeneratedPythonCost | None = None
    generated_costs_by_label: dict[int, GeneratedPythonCost] = {}

    # Phase 1: generate every cluster's LLM cost (sequential). Each generation grounds
    # on all cluster means (overlay highlights the active cluster); the prompt template
    # decides whether to use the other paths.
    costs_ready: list[tuple[int, np.ndarray, GeneratedPythonCost, Path, dict[str, Any]]] = []
    for label in summary["cluster_ids"]:
        cluster_traj = uq_result.cluster_means[int(label)]
        cluster_dir = root_dir / f"cluster_{label}"
        install_selected = int(label) == uq_result.chosen_label
        generated_context = build_generated_cost_context(
            context, current_q, cluster_traj, q_history, window=history_window,
            body_pos=body_pos, reference_traj=reference_traj,
        )
        summaries = build_motion_summaries(generated_context, cartesian_goal=goal_pos)
        images: dict[str, Path] = {}
        if cfg.llm_cost.use_images:
            images = render_prompt_images(
                generated_context, cluster_dir / "images",
                uq_result.cluster_means, int(label),
                reference_traj=reference_traj, goal_pos=goal_pos,
            )
        generator = create_cost_generator(
            cfg.llm_cost, generated_context, instruction,
            summaries=summaries, run_dir=cluster_dir, images=images, mpc=mpc,
            llm_model_factory=llm_model_factory,
        )
        generated = generator.generate(install=install_selected and install)
        validation = _read_json_if_exists(cluster_dir / "validation.json")
        params = _read_json_if_exists(cluster_dir / "params.json")
        entry: dict[str, Any] = {
            "artifact_path": str(cluster_dir),
            "rollout_path": None,
            "metrics_path": None,
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
            summary["clusters"][str(label)] = entry
            _write_comparison_summary(root_dir, summary)
            continue
        if install_selected:
            selected_cost = generated
        generated_costs_by_label[int(label)] = generated
        costs_ready.append((label, cluster_traj, generated, cluster_dir, entry))
        summary["clusters"][str(label)] = entry
        _write_comparison_summary(root_dir, summary)

    # Phase 2: roll out each cluster's cost and (optionally) render the video.
    cartesian_goal = np.asarray(cfg.cartesian.goals[0]) if cfg.cartesian.goals else None
    for label, traj, gen_cost, cdir, entry in costs_ready:
        rollout, metrics, colors = _run_cluster_comparison_rollout(
            cfg, current_q, traj, context, base_extra_costs, gen_cost,
            rollout_steps, body_pos, spine3_pos, spine3_aa,
        )
        np.save(cdir / "rollout.npy", rollout)
        with open(cdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        entry["rollout_path"] = str(cdir / "rollout.npy")
        entry["metrics_path"] = str(cdir / "metrics.json")
        entry["rollout_metrics"] = metrics
        if save_video:
            n_frames = traj.shape[0]
            cutoff = max(1, round(n_frames * cfg.trajectory_fraction))
            _render_rollout_video(
                rollout, context.fk, spine3_pos, spine3_aa,
                cdir / "rollout.mp4",
                body_pos=body_pos, cartesian_goal=cartesian_goal,
                frame_colors=colors, mdm_goal_q=traj[cutoff - 1],
            )
        summary["clusters"][str(label)] = entry
        _write_comparison_summary(root_dir, summary)

    if initial_q is not None and cfg.planner in (
        "arm_mpc_cartesian", "arm_mpc_cartesian_no_mdm"
    ):
        for label in summary["cluster_ids"]:
            gen_cost = generated_costs_by_label.get(int(label))
            if gen_cost is None:
                continue
            cdir = root_dir / f"cluster_{label}"
            cluster_entry = summary["clusters"].get(str(label), {})
            is_rollout, is_metrics = _run_initial_state_rollout(
                cfg, initial_q, context, base_extra_costs, gen_cost,
                rollout_steps, body_pos, spine3_pos, spine3_aa,
            )
            np.save(cdir / "initial_state_rollout.npy", is_rollout)
            with open(cdir / "initial_state_metrics.json", "w", encoding="utf-8") as f:
                json.dump(is_metrics, f, indent=2, sort_keys=True)
            if save_video:
                _render_rollout_video(
                    is_rollout, context.fk, spine3_pos, spine3_aa,
                    cdir / "initial_state_rollout.mp4",
                    body_pos=body_pos, cartesian_goal=cartesian_goal,
                )
            cluster_entry["initial_state_rollout_path"] = str(
                cdir / "initial_state_rollout.npy"
            )
            cluster_entry["initial_state_metrics"] = is_metrics
            summary["clusters"][str(label)] = cluster_entry
            _write_comparison_summary(root_dir, summary)

    print(f"[llm-cost] cluster comparison artifacts saved to: {root_dir}")
    return selected_cost
