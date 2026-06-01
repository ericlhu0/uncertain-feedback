"""Unified entry point for running arm MPC motion planning.

Usage examples::

    # UQ planner with live GUI (default pose + default text prompt)
    python -m uncertain_feedback.planners.run --mpc-config mpc.yaml --live

    # Save a compact video without watching
    python -m uncertain_feedback.planners.run --mpc-config mpc.yaml --text "wave left arm" --save out.mp4

    # Custom starting pose and arm override
    python -m uncertain_feedback.planners.run --mpc-config mpc.yaml --pose my_pose.pt --arm my_arm.npy --live

    # Plain MPC (no MDM)
    python -m uncertain_feedback.planners.run --mpc-config plain_mpc.yaml --live
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.planners.mpc import (
    ArmMPCCartesianNoMDM,
    ArmVisualizer,
    LeftArmMPCCartesian,
    LeftArmMPCMDM,
    LeftArmMPCMDMUQ,
    SmplLeftArmFK,
    SmplLeftArmMPC,
)
from uncertain_feedback.planners.mpc.arm_mpc_mdm_uq import UqClusterResult
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    LearnablePreferenceCost,
    MpcCostContext,
    build_extra_costs,
    replace_cost_in_composite,
    update_preference_cost,
)
from uncertain_feedback.planners.mpc.llm_costs import (
    GeneratedPythonCost,
    _IMAGE_DESCRIPTION_PROMPT,
    build_generated_cost_context,
    build_llm_cost_prompt,
    build_motion_summaries,
    parse_llm_cost_response,
    render_prompt_images,
)

_MDM_PLANNERS = {"arm_mpc_mdm", "arm_mpc_mdm_uq", "arm_mpc_cartesian"}


@dataclass
class _InitialPoseState:
    """Initial controlled-arm state plus the whole-body context it came from."""

    arm_aa: np.ndarray
    fixed_collar_aa: np.ndarray
    body_pos: np.ndarray | None = None
    spine3_pos: np.ndarray | None = None
    spine3_aa: np.ndarray | None = None
    hml_pose: np.ndarray | None = None

    @classmethod
    def tpose(cls) -> "_InitialPoseState":
        return cls(arm_aa=np.zeros((3, 3)), fixed_collar_aa=np.zeros(3))


def _make_motion_generator(model_path: Path | None) -> Any:
    from uncertain_feedback.motion_generators.mdm.mdm_api import (  # pylint: disable=import-outside-toplevel
        MdmMotionGenerator,
    )

    return MdmMotionGenerator(model_path=model_path)


def _load_initial_pose_state(
    args: argparse.Namespace,
    uses_mdm: bool,
    config_pose: Path | None = None,
    motion_generator_factory: Callable[[Path | None], Any] | None = None,
) -> tuple[Any | None, _InitialPoseState]:
    """Load the optional HML pose used to initialize all planner variants.

    MDM-backed planners keep the historical default sitting pose. Non-MDM
    planners keep their T-pose default unless the user explicitly passes
    ``--pose`` or sets ``pose`` in the YAML config.
    """
    pose_path = args.pose if args.pose is not None else config_pose
    if uses_mdm and pose_path is None:
        pose_path = MDM_ROOT / "demo_pose.pt"
    if pose_path is None:
        return None, _InitialPoseState.tpose()

    factory = motion_generator_factory or _make_motion_generator
    gen = factory(args.model_path)
    hml_pose = gen.load_hml_pose(pose_path)
    arm_aa, body_pos, spine3_aa, fixed_collar_aa = gen.decode_pose_with_collar(hml_pose)
    return gen, _InitialPoseState(
        arm_aa=np.asarray(arm_aa, dtype=np.float64),
        fixed_collar_aa=np.asarray(fixed_collar_aa, dtype=np.float64),
        body_pos=np.asarray(body_pos, dtype=np.float64),
        spine3_pos=np.asarray(body_pos[9], dtype=np.float64),
        spine3_aa=np.asarray(spine3_aa, dtype=np.float64),
        hml_pose=np.asarray(hml_pose, dtype=np.float64),
    )


def _apply_arm_override(state: _InitialPoseState, arm_path: Path | None) -> None:
    if arm_path is None:
        return

    arm_override = np.load(arm_path)
    state.arm_aa = np.asarray(arm_override, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run arm MPC motion planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mpc-config",
        type=Path,
        required=True,
        dest="mpc_config",
        help="Required YAML file with MPC planner, controller settings, and costs.",
    )

    # --- Model ---
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        dest="model_path",
        help="Path to MDM weights .pt file. Defaults to the base humanml model.",
    )

    # --- Pose input ---
    p.add_argument(
        "--pose",
        type=Path,
        default=None,
        help=(
            "Override the YAML pose path with a body pose .pt file (HML263 format). "
            f"MDM-backed planners default to {MDM_ROOT}/demo_pose.pt; "
            "non-MDM planners use T-pose unless YAML pose or this override is set."
        ),
    )
    p.add_argument(
        "--arm",
        type=Path,
        default=None,
        help="Optional .npy file with (3, 3) shoulder/elbow/wrist axis-angles.",
    )

    # --- Visualization ---
    p.add_argument(
        "--live",
        action="store_true",
        help="Show interactive matplotlib window while running",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save video to this path (.mp4 or .gif). Uses compact 1-panel layout.",
    )
    p.add_argument("--fps", type=int, default=20, help="FPS for saved video")

    # --- MDM args (arm_mpc_mdm, arm_mpc_mdm_uq) ---
    p.add_argument(
        "--text",
        type=str,
        default="move my arm up",
        help="Natural-language MDM motion description (mdm/uq planners only)",
    )
    p.add_argument(
        "--text-time",
        type=int,
        default=0,
        dest="text_time",
        help="MPC step at which MDM generation is triggered",
    )
    p.add_argument(
        "--save-motion",
        type=Path,
        default=None,
        dest="save_motion",
        help="Save the raw MDM full-body motion video to this path (arm_mpc_mdm only)",
    )
    p.add_argument(
        "--mdm-frames",
        type=int,
        default=None,
        dest="mdm_frames",
        help="Exact number of MDM frames to generate (1-196). Default is 120.",
    )
    p.add_argument(
        "--frozen-body",
        action="store_true",
        dest="frozen_body",
        help="Freeze non-left-arm body features during MDM generation.",
    )
    p.add_argument(
        "--preference-output",
        type=Path,
        default="learned.yaml",
        dest="preference_output",
        help=(
            "Where to save a YAML copy with learned preference costs. "
            "Defaults to <mpc-config stem>_learned.yaml next to the input config."
        ),
    )
    return p


def _restore_interactive_backend() -> None:
    if plt.get_backend().lower() == "agg":
        for backend in ("Qt5Agg", "TkAgg", "Qt6Agg", "WXAgg", "MacOSX"):
            try:
                plt.switch_backend(backend)
                break
            except Exception:  # pylint: disable=broad-exception-caught
                continue


def _get_vis(mpc: SmplLeftArmMPC) -> ArmVisualizer | None:
    return mpc.get_visualizer()


def _iter_learnable_costs(
    composite: CompositeTrajectoryCost,
) -> list[LearnablePreferenceCost]:
    """Return all configured preference costs that support learned bounds."""
    costs: list[LearnablePreferenceCost] = []
    for term in composite.terms():
        if isinstance(term, LearnablePreferenceCost):
            costs.append(term)
    return costs


def _apply_preference_update(
    mpc: SmplLeftArmMPC,
    mdm_traj: np.ndarray,
    q_history: list[np.ndarray],
    context: MpcCostContext,
    alpha: float,
    window: int,
) -> list[LearnablePreferenceCost]:
    """Update configured preference bounds from MDM/MPC discrepancy."""
    _ = context
    costs = _iter_learnable_costs(mpc._extra_costs)  # pylint: disable=protected-access
    if not costs:
        return []
    if not q_history:
        return []
    recent_q = np.array(q_history[-window:])
    updated_costs: list[LearnablePreferenceCost] = []
    extra_costs = mpc._extra_costs  # pylint: disable=protected-access

    for cost in costs:
        mpc_values = cost.feature_values(recent_q)
        mdm_values = cost.feature_values(mdm_traj)
        mdm_lo, mdm_hi = np.percentile(mdm_values, [5.0, 95.0])
        mpc_mean = float(mpc_values.mean())
        mdm_mean = float(mdm_values.mean())
        if np.isclose(mpc_mean, mdm_mean):
            side = "none"
        elif mpc_mean < mdm_mean:
            side = "min"
        else:
            side = "max"
        updated = update_preference_cost(
            cost,
            mdm_values,
            mpc_values,
            alpha=alpha,
        )
        print(
            f"[preference] {cost.cost_name} bounds updated: "
            f"[{cost.min_value:.3f}, {cost.max_value:.3f}] -> "
            f"[{updated.min_value:.3f}, {updated.max_value:.3f}]  "
            f"(side={side} mdm_range=[{mdm_lo:.3f}, {mdm_hi:.3f}] "
            f"mpc_mean={mpc_mean:.3f}, mdm_mean={mdm_mean:.3f})"
        )
        extra_costs = replace_cost_in_composite(extra_costs, updated)
        updated_costs.append(updated)

    mpc.set_extra_costs(extra_costs)
    return updated_costs


def _default_preference_output_path(config_path: Path) -> Path:
    """Return the default learned-preference YAML path for an input config."""
    return config_path.with_name(f"{config_path.stem}_learned{config_path.suffix}")


def _make_llm_model(model_name: str) -> Any:
    """Build the cost-generator LLM wrapper."""
    from uncertain_feedback.llm import (
        OpenAIModel,
    )  # pylint: disable=import-outside-toplevel

    return OpenAIModel(
        model=model_name,
        system_prompt=(
            "You generate safe, vectorized Python MPC trajectory cost functions. "
            "Return only the requested JSON object."
        ),
        temperature=0.2,
        max_tokens=1800,
    )


def _llm_artifact_run_dir(base_dir: Path, artifact_dir: Path) -> Path:
    """Return a unique artifact directory for one LLM-cost generation."""
    root = artifact_dir if artifact_dir.is_absolute() else base_dir / artifact_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def _append_extra_cost(
    composite: CompositeTrajectoryCost,
    cost: GeneratedPythonCost,
) -> CompositeTrajectoryCost:
    """Return a composite with an additional generated cost term."""
    return CompositeTrajectoryCost([*composite.terms(), cost])


def _apply_llm_generated_cost(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: SmplLeftArmMPC,
    instruction: str,
    mdm_traj: np.ndarray,
    current_q: np.ndarray,
    q_history: list[np.ndarray],
    context: MpcCostContext,
    llm_cfg: Any,
    artifact_base_dir: Path,
    history_window: int,
    llm_model_factory: Callable[[str], Any] = _make_llm_model,
    run_dir: Path | None = None,
    install: bool = True,
) -> GeneratedPythonCost | None:
    """Generate, validate, save, and install an LLM-generated MPC cost."""
    if run_dir is None:
        run_dir = _llm_artifact_run_dir(artifact_base_dir, llm_cfg.artifact_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir = run_dir / "images"
    validation: dict[str, Any] = {"ok": False}

    try:
        generated_context = build_generated_cost_context(
            context,
            current_q,
            mdm_traj,
            q_history,
            window=history_window,
        )
        summaries = build_motion_summaries(generated_context)
        image_paths: list[Path] = []
        if llm_cfg.use_images:
            image_paths = render_prompt_images(generated_context, image_dir)
        with open(run_dir / "summaries.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, sort_keys=True)

        model_name = llm_cfg.model or os.getenv("OPENAI_MODEL", "gpt-5.4")
        llm = llm_model_factory(model_name)
        image_input = [str(path) for path in image_paths] or None
        image_description = ""
        if image_input:
            image_description = llm.get_full_output(
                _IMAGE_DESCRIPTION_PROMPT, image_input=image_input
            )
            (run_dir / "image_description.txt").write_text(
                image_description, encoding="utf-8"
            )
            print(f"[llm-cost] image description: {image_description}")
        prompt = build_llm_cost_prompt(instruction, summaries, image_paths, image_description)
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        raw_response = llm.get_full_output(prompt, image_input=image_input)
        (run_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        response = parse_llm_cost_response(raw_response)
        generated_cost = GeneratedPythonCost(
            code=response.code,
            params=response.params,
            description=response.description,
            context=generated_context,
        )
        validation_q = np.repeat(
            current_q[np.newaxis, np.newaxis],
            repeats=2,
            axis=1,
        )
        generated_cost(validation_q)

        (run_dir / "cost.py").write_text(response.code, encoding="utf-8")
        with open(run_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": response.description,
                    "params": response.params,
                    "model": model_name,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        if install:
            mpc.set_extra_costs(
                _append_extra_cost(
                    mpc._extra_costs,  # pylint: disable=protected-access
                    generated_cost,
                )
            )
        validation = {"ok": True, "artifact_dir": str(run_dir)}
        action = "installed" if install else "generated"
        print(f"[llm-cost] {action} generated cost: {response.description}")
        print(f"[llm-cost] artifacts saved to: {run_dir}")
        return generated_cost
    except Exception as exc:  # pylint: disable=broad-exception-caught
        validation = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "artifact_dir": str(run_dir),
        }
        action = "install" if install else "generate"
        print(f"[llm-cost] failed to {action} generated cost: {exc}")
        if llm_cfg.strict:
            raise
        return None
    finally:
        with open(run_dir / "validation.json", "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2, sort_keys=True)


def _run_llm_cluster_experiment(  # pylint: disable=too-many-arguments,too-many-locals
    mpc: SmplLeftArmMPC,
    cfg: MpcRunConfig,
    instruction: str,
    uq_result: UqClusterResult,
    current_q: np.ndarray,
    q_history: list[np.ndarray],
    context: MpcCostContext,
    artifact_base_dir: Path,
    history_window: int,
    remaining_steps: int,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    llm_model_factory: Callable[[str], Any] = _make_llm_model,
) -> GeneratedPythonCost | None:
    """Generate one cost per UQ cluster and run headless comparison rollouts."""
    cluster_cfg = cfg.llm_cost.cluster_experiment
    root_dir = _llm_artifact_run_dir(artifact_base_dir, cfg.llm_cost.artifact_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "selected_cluster.txt").write_text(
        f"{uq_result.chosen_label}\n", encoding="utf-8"
    )

    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access
    rollout_steps = (
        max(1, remaining_steps)
        if cluster_cfg.rollout_steps is None
        else cluster_cfg.rollout_steps
    )
    summary: dict[str, Any] = {
        "selected_cluster": uq_result.chosen_label,
        "cluster_ids": sorted(int(label) for label in uq_result.cluster_means),
        "clusters": {},
    }
    selected_cost: GeneratedPythonCost | None = None

    for label in summary["cluster_ids"]:
        cluster_traj = uq_result.cluster_means[int(label)]
        cluster_dir = root_dir / f"cluster_{label}"
        install_selected = int(label) == uq_result.chosen_label
        generated = _apply_llm_generated_cost(
            mpc,
            instruction,
            cluster_traj,
            current_q,
            q_history,
            context,
            cfg.llm_cost,
            artifact_base_dir,
            history_window,
            llm_model_factory=llm_model_factory,
            run_dir=cluster_dir,
            install=install_selected,
        )
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
            "params": params.get("params") if isinstance(params, dict) else None,
            "rollout_metrics": None,
        }
        if generated is None:
            summary["clusters"][str(label)] = entry
            _write_comparison_summary(root_dir, summary)
            continue

        if install_selected:
            selected_cost = generated

        rollout, metrics = _run_cluster_comparison_rollout(
            cfg,
            current_q,
            cluster_traj,
            context,
            base_extra_costs,
            generated,
            rollout_steps,
            body_pos,
            spine3_pos,
            spine3_aa,
        )
        rollout_path = cluster_dir / "rollout.npy"
        metrics_path = cluster_dir / "metrics.json"
        np.save(rollout_path, rollout)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        entry["rollout_path"] = str(rollout_path)
        entry["metrics_path"] = str(metrics_path)
        entry["rollout_metrics"] = metrics
        summary["clusters"][str(label)] = entry
        _write_comparison_summary(root_dir, summary)

    print(f"[llm-cost] cluster comparison artifacts saved to: {root_dir}")
    return selected_cost


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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a headless MPC rollout with one cluster's generated cost."""
    comparison_mpc = _build_cluster_rollout_planner(
        cfg,
        current_q,
        cluster_traj,
        context,
        base_extra_costs,
        generated_cost,
        body_pos,
        spine3_pos,
        spine3_aa,
    )
    q = np.asarray(current_q, dtype=np.float64).copy()
    rollout_frames = [q.copy()]
    error: str | None = None
    for _ in range(rollout_steps):
        try:
            q = comparison_mpc.step(q)
        except RuntimeError as exc:
            error = str(exc)
            break
        rollout_frames.append(q.copy())

    rollout = np.asarray(rollout_frames, dtype=np.float64)
    metrics = _rollout_metrics(rollout, comparison_mpc, context)
    metrics["steps_requested"] = rollout_steps
    metrics["steps_completed"] = int(max(0, rollout.shape[0] - 1))
    if error is not None:
        metrics["error"] = error
    return rollout, metrics


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
    """Build an isolated planner for one cluster comparison rollout."""
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
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )
    else:
        planner = LeftArmMPCMDM(
            **common,
            goals=[],
            advance_threshold=cfg.advance_threshold,
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


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_comparison_summary(root_dir: Path, summary: dict[str, Any]) -> None:
    with open(root_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _save_learned_preference_yaml(
    input_path: Path,
    output_path: Path,
    learned_costs: LearnablePreferenceCost | list[LearnablePreferenceCost],
) -> None:
    """Save a config copy with learned preference parameters."""
    with open(input_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    data = raw if isinstance(raw, dict) else {}

    costs = data.get("costs")
    if not isinstance(costs, dict):
        costs = {}
        data["costs"] = costs

    normalized_costs = (
        [learned_costs]
        if isinstance(learned_costs, LearnablePreferenceCost)
        else learned_costs
    )
    for learned_cost in normalized_costs:
        cost_data = costs.get(learned_cost.cost_name)
        if not isinstance(cost_data, dict):
            cost_data = {}
            costs[learned_cost.cost_name] = cost_data
        cost_data["min"] = float(learned_cost.min_value)
        cost_data["max"] = float(learned_cost.max_value)
        cost_data["weight"] = float(learned_cost.weight)
        cost_data["progress_weight"] = float(learned_cost.progress_weight)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[preference] saved learned preference YAML: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    uses_mdm = cfg.planner in _MDM_PLANNERS
    visualize = args.live or (args.save is not None)
    # Compact (1-panel) rendering when saving without live view — faster to render and encode
    compact = (args.save is not None) and not args.live

    # --- Load pose ---
    gen, initial_state = _load_initial_pose_state(args, uses_mdm, cfg.pose)
    _apply_arm_override(initial_state, args.arm)
    initial_pose = initial_state.hml_pose
    arm_aa = initial_state.arm_aa
    body_pos = initial_state.body_pos
    spine3_pos = initial_state.spine3_pos
    spine3_aa = initial_state.spine3_aa

    fk = SmplLeftArmFK()
    fk.collar_aa = initial_state.fixed_collar_aa
    cost_context = MpcCostContext(
        fk=fk,
        spine3_pos=np.asarray(
            spine3_pos if spine3_pos is not None else fk.tpose_spine3_pos,
            dtype=np.float64,
        ),
        spine3_aa=np.asarray(
            spine3_aa if spine3_aa is not None else np.zeros(3), dtype=np.float64
        ),
    )
    extra_costs = build_extra_costs(cfg.costs, cost_context)

    # Default goal: arm raised from the initial pose
    default_goal = arm_aa.copy() + np.array(
        [[0.0, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    # --- Build planner ---
    common: dict = dict(
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        goal_threshold=cfg.goal_threshold,
        visualize=visualize,
        fk=fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        body_pos=body_pos,
        extra_costs=extra_costs,
    )

    mpc: SmplLeftArmMPC
    if cfg.planner == "arm_mpc":
        mpc = SmplLeftArmMPC(
            **common,
            goals=[default_goal],
        )
    elif cfg.planner == "arm_mpc_mdm":
        mpc = LeftArmMPCMDM(
            **common,
            goals=[default_goal],
            advance_threshold=cfg.advance_threshold,
            trajectory_fraction=cfg.trajectory_fraction,
        )
    elif cfg.planner == "arm_mpc_cartesian_no_mdm":
        if not cfg.cartesian.goals:
            raise ValueError(
                "cartesian.goals is required when planner is arm_mpc_cartesian_no_mdm."
            )
        _spine3_ref = spine3_pos if spine3_pos is not None else fk.tpose_spine3_pos
        init_wrist_rel = fk.fk(arm_aa, spine3_pos, spine3_aa)[-1] - _spine3_ref
        print(f"Initial wrist position (spine3-relative): {init_wrist_rel}")
        mpc = ArmMPCCartesianNoMDM(
            cartesian_goals=[np.array(g) for g in cfg.cartesian.goals],
            initial_arm_aa=arm_aa,
            cartesian_threshold=cfg.cartesian.threshold,
            **common,
        )
    elif cfg.planner == "arm_mpc_cartesian":
        if not cfg.cartesian.goals:
            raise ValueError(
                "cartesian.goals is required when planner is arm_mpc_cartesian."
            )
        _spine3_ref = spine3_pos if spine3_pos is not None else fk.tpose_spine3_pos
        init_wrist_rel = fk.fk(arm_aa, spine3_pos, spine3_aa)[-1] - _spine3_ref
        print(f"Initial wrist position (spine3-relative): {init_wrist_rel}")
        mpc = LeftArmMPCCartesian(
            cartesian_goals=[np.array(g) for g in cfg.cartesian.goals],
            initial_arm_aa=arm_aa,
            cartesian_threshold=cfg.cartesian.threshold,
            **common,
            advance_threshold=cfg.advance_threshold,
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )
    else:
        mpc = LeftArmMPCMDMUQ(
            **common,
            goals=[default_goal],
            advance_threshold=cfg.advance_threshold,
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )

    mpc.set_visualization_mode(capture=args.save is not None, compact=compact)

    # --- MPC loop ---
    from tqdm import tqdm  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    q = arm_aa.copy()
    q_history: list[np.ndarray] = []
    mdm_triggered = False
    pre_mdm_vis: ArmVisualizer | None = None
    preference_output_path = args.preference_output or _default_preference_output_path(
        args.mpc_config
    )

    for step in tqdm(range(cfg.steps), desc="MPC", unit="step"):
        # Trigger MDM generation at the configured step
        if uses_mdm and args.text and step == args.text_time and not mdm_triggered:
            mdm_triggered = True
            assert gen is not None and initial_pose is not None

            # Close vis before the blocking MDM computation to avoid GUI freeze
            pre_mdm_vis = mpc.close_visualizer()

            current_pose = gen.build_pose_from_arm_aa(initial_pose, q)

            if cfg.planner == "arm_mpc_mdm":
                traj = gen.generate_left_arm_trajectory(
                    args.text,
                    start_pose=current_pose,
                    save_path=str(args.save_motion) if args.save_motion else None,
                    num_frames=args.mdm_frames,
                    frozen_body=args.frozen_body,
                    spine3_aa=spine3_aa,
                )
                if cfg.preference_learning:
                    learned_costs = _apply_preference_update(
                        mpc,
                        traj,
                        q_history,
                        cost_context,
                        alpha=cfg.preference_alpha,
                        window=cfg.preference_window,
                    )
                    if learned_costs:
                        _save_learned_preference_yaml(
                            args.mpc_config, preference_output_path, learned_costs
                        )
                else:
                    print(
                        "[preference] learning disabled; generated trajectory "
                        "will not update elbow bounds"
                    )
                if cfg.llm_cost.enabled:
                    _apply_llm_generated_cost(
                        mpc,
                        args.text,
                        traj,
                        q,
                        q_history,
                        cost_context,
                        cfg.llm_cost,
                        artifact_base_dir,
                        cfg.preference_window,
                    )
                mdm_mpc = cast(LeftArmMPCMDM, mpc)
                n_frames = traj.shape[0]
                cutoff = max(1, round(n_frames * mdm_mpc.trajectory_fraction))
                mdm_mpc.set_mdm_goal(traj[cutoff - 1])
                mdm_mpc.push_trajectory(traj[:cutoff])
            else:
                uq_mpc = cast(LeftArmMPCMDMUQ, mpc)
                traj = uq_mpc.query_mdm_with_uncertainty(
                    gen,
                    args.text,
                    start_pose=current_pose,
                    current_arm_aa=q,
                    auto_cluster=cfg.uq.auto_cluster,
                    mdm_frames=args.mdm_frames,
                    frozen_body=args.frozen_body,
                )
                if cfg.preference_learning:
                    learned_costs = _apply_preference_update(
                        mpc,
                        traj,
                        q_history,
                        cost_context,
                        alpha=cfg.preference_alpha,
                        window=cfg.preference_window,
                    )
                    if learned_costs:
                        _save_learned_preference_yaml(
                            args.mpc_config, preference_output_path, learned_costs
                        )
                else:
                    print(
                        "[preference] learning disabled; generated trajectory "
                        "will not update elbow bounds"
                    )
                if cfg.llm_cost.enabled:
                    uq_result = uq_mpc.last_uq_result
                    if (
                        cfg.llm_cost.cluster_experiment.enabled
                        and uq_result is not None
                    ):
                        _run_llm_cluster_experiment(
                            mpc,
                            cfg,
                            args.text,
                            uq_result,
                            q,
                            q_history,
                            cost_context,
                            artifact_base_dir,
                            cfg.preference_window,
                            remaining_steps=max(1, cfg.steps - step),
                            body_pos=body_pos,
                            spine3_pos=spine3_pos,
                            spine3_aa=spine3_aa,
                        )
                    else:
                        _apply_llm_generated_cost(
                            mpc,
                            args.text,
                            traj,
                            q,
                            q_history,
                            cost_context,
                            cfg.llm_cost,
                            artifact_base_dir,
                            cfg.preference_window,
                        )

            # MDM generation can switch matplotlib to the Agg backend; restore it
            _restore_interactive_backend()

        q = mpc.step(q)
        q_history.append(q)

    # --- Save video ---
    if args.save and visualize:
        vis = _get_vis(mpc)
        if vis is not None:
            if (
                pre_mdm_vis is not None and pre_mdm_vis._frame_bufs
            ):  # pylint: disable=protected-access
                vis.prepend_frames(
                    pre_mdm_vis._frame_bufs
                )  # pylint: disable=protected-access
            vis.finish_live(str(args.save), fps=args.fps)

    if args.live:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
