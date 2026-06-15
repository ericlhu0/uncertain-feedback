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
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.planners.mpc import (
    ArmMPCCartesianNoMDM,
    LeftArmMPCCartesian,
    LeftArmMPCMDM,
    LeftArmMPCMDMUQ,
    SmplLeftArmFK,
    SmplLeftArmMPC,
)
from uncertain_feedback.utils.plot import ArmVisualizer
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
        default=None,
        dest="text_time",
        help="MPC step at which MDM generation is triggered (overrides YAML text_time)",
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
    body_pos: np.ndarray | None = None,
    llm_model_factory: Callable[[str], Any] = _make_llm_model,
    run_dir: Path | None = None,
    install: bool = True,
    pre_rendered_image_paths: list[Path] | None = None,
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
            body_pos=body_pos,
        )
        summaries = build_motion_summaries(generated_context)
        image_paths: list[Path] = []
        if pre_rendered_image_paths is not None:
            image_paths = pre_rendered_image_paths
        elif llm_cfg.use_images:
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
        prompt = build_llm_cost_prompt(instruction, summaries, image_paths)
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
            repeats=11,
            axis=1,
        )
        generated_cost(validation_q)

        (run_dir / "cost.py").write_text(response.code, encoding="utf-8")
        if response.explanation:
            (run_dir / "explanation.txt").write_text(
                response.explanation,
                encoding="utf-8",
            )
            print(f"[llm-cost] explanation: {response.explanation}")
        if response.recipient_explanation:
            (run_dir / "recipient_explanation.txt").write_text(
                response.recipient_explanation,
                encoding="utf-8",
            )
            print(
                "[llm-cost] recipient explanation: "
                f"{response.recipient_explanation}"
            )
        with open(run_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": response.description,
                    "explanation": response.explanation,
                    "recipient_explanation": response.recipient_explanation,
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


# --- Run setup + unified planning loop --------------------------------------


@dataclass
class RunSetup:
    """Planner, kinematics, and cost context for one run.

    Shared by the single-run entry point (``main``) and the experiment runner so
    both reach an identical planner/context before stepping.
    """

    mpc: SmplLeftArmMPC
    gen: Any | None
    fk: SmplLeftArmFK
    cost_context: MpcCostContext
    body_pos: np.ndarray | None
    spine3_pos: np.ndarray | None
    spine3_aa: np.ndarray | None
    arm_aa: np.ndarray
    initial_pose: np.ndarray | None
    uses_mdm: bool
    visualize: bool
    compact: bool


def build_run(args: argparse.Namespace, cfg: MpcRunConfig) -> RunSetup:
    """Load the pose, kinematics, costs, and planner for a single run."""
    uses_mdm = cfg.planner in _MDM_PLANNERS
    visualize = args.live or (args.save is not None)
    # Compact (1-panel) rendering when saving without live view.
    compact = (args.save is not None) and not args.live

    gen, initial_state = _load_initial_pose_state(args, uses_mdm, cfg.pose)
    _apply_arm_override(initial_state, args.arm)
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
        mpc = SmplLeftArmMPC(**common, goals=[default_goal])
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

    return RunSetup(
        mpc=mpc,
        gen=gen,
        fk=fk,
        cost_context=cost_context,
        body_pos=body_pos,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        arm_aa=arm_aa,
        initial_pose=initial_state.hml_pose,
        uses_mdm=uses_mdm,
        visualize=visualize,
        compact=compact,
    )


@dataclass
class LoopResult:
    """Joint configs visited by :func:`run_planning_loop`."""

    q_history: list[np.ndarray]
    error: str | None = None


StepHook = Callable[[int, np.ndarray, list[np.ndarray]], None]


def run_planning_loop(
    mpc: SmplLeftArmMPC,
    q0: np.ndarray,
    n_steps: int,
    *,
    on_pre_step: StepHook | None = None,
    on_post_step: StepHook | None = None,
    stop_on_runtime_error: bool = False,
    progress: bool = False,
    progress_desc: str = "MPC",
) -> LoopResult:
    """Step ``mpc`` forward ``n_steps``, returning the visited joint configs.

    This is the single stepping primitive shared by the live single run and the
    headless per-cluster experiment rollouts. The planner drives its own
    live/captured visualization (per the ``visualize``/``capture`` flags it was
    built with), so a live and a saved rollout share one rendering path.

    ``on_pre_step(step, q, q_history)`` runs before each ``mpc.step`` (the single
    run uses it to trigger MDM/LLM generation at ``text_time``); ``on_post_step``
    runs after (deferred LLM install, or per-step bookkeeping like frame colors).
    ``q_history`` holds the configs visited so far. With ``stop_on_runtime_error``
    a ``RuntimeError`` from ``mpc.step`` ends the loop and is recorded on the
    result instead of propagating.
    """
    q = np.asarray(q0, dtype=np.float64).copy()
    q_history: list[np.ndarray] = []
    iterator: Iterable[int] = range(n_steps)
    if progress:
        from tqdm import tqdm  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

        iterator = tqdm(iterator, desc=progress_desc, unit="step")
    error: str | None = None
    for step in iterator:
        if on_pre_step is not None:
            on_pre_step(step, q, q_history)
        try:
            q = mpc.step(q)
        except RuntimeError as exc:
            if not stop_on_runtime_error:
                raise
            error = str(exc)
            break
        q_history.append(q.copy())
        if on_post_step is not None:
            on_post_step(step, q, q_history)
    return LoopResult(q_history=q_history, error=error)


def _start_llm_cost_thread(  # pylint: disable=too-many-arguments
    mpc: SmplLeftArmMPC,
    args: argparse.Namespace,
    cfg: MpcRunConfig,
    cost_context: MpcCostContext,
    artifact_base_dir: Path,
    traj: np.ndarray,
    current_q: np.ndarray,
    q_history: list[np.ndarray],
    body_pos: np.ndarray | None,
    result_out: list[GeneratedPythonCost | None],
) -> threading.Thread:
    """Generate the single LLM correction cost in a daemon thread.

    Images are pre-rendered on the main thread (matplotlib is not thread-safe).
    The generated cost is appended to ``result_out`` (not installed); the caller
    installs it once the correction trajectory completes.
    """
    run_dir = _llm_artifact_run_dir(artifact_base_dir, cfg.llm_cost.artifact_dir)
    pre_imgs: list[Path] = []
    if cfg.llm_cost.use_images:
        run_dir.mkdir(parents=True, exist_ok=True)
        img_ctx = build_generated_cost_context(
            cost_context, current_q, traj, list(q_history),
            window=cfg.preference_window, body_pos=body_pos,
        )
        pre_imgs = render_prompt_images(img_ctx, run_dir / "images")

    traj_snap = traj.copy()
    q_snap = current_q.copy()
    qh_snap = list(q_history)
    pre = pre_imgs if pre_imgs else None

    def _thread_fn() -> None:
        result = _apply_llm_generated_cost(
            mpc, args.text, traj_snap, q_snap, qh_snap,
            cost_context, cfg.llm_cost, artifact_base_dir,
            cfg.preference_window, body_pos=body_pos,
            install=False, pre_rendered_image_paths=pre, run_dir=run_dir,
        )
        result_out.append(result)

    thread = threading.Thread(target=_thread_fn, daemon=True)
    thread.start()
    return thread


def main() -> None:
    args = build_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    setup = build_run(args, cfg)
    mpc = setup.mpc
    gen = setup.gen
    cost_context = setup.cost_context
    body_pos = setup.body_pos
    spine3_aa = setup.spine3_aa
    initial_pose = setup.initial_pose

    preference_output_path = args.preference_output or _default_preference_output_path(
        args.mpc_config
    )
    effective_text_time = args.text_time if args.text_time is not None else cfg.text_time

    mdm_triggered = False
    pre_mdm_vis: ArmVisualizer | None = None
    pending_llm_thread: threading.Thread | None = None
    pending_llm_result: list[GeneratedPythonCost | None] = []

    def _trigger_correction(q: np.ndarray, q_history: list[np.ndarray]) -> None:
        """Generate the MDM/UQ correction (and optional LLM cost) at text_time."""
        nonlocal pre_mdm_vis, pending_llm_thread
        assert gen is not None and initial_pose is not None

        # Close vis before the blocking MDM computation to avoid GUI freeze
        pre_mdm_vis = mpc.close_visualizer()
        current_pose = gen.build_pose_from_arm_aa(initial_pose, q)
        mdm_frames = args.mdm_frames if args.mdm_frames is not None else cfg.mdm_frames

        if cfg.planner == "arm_mpc_mdm":
            traj = gen.generate_left_arm_trajectory(
                args.text,
                start_pose=current_pose,
                save_path=str(args.save_motion) if args.save_motion else None,
                num_frames=mdm_frames,
                frozen_body=args.frozen_body,
                spine3_aa=spine3_aa,
            )
        else:
            uq_mpc = cast(LeftArmMPCMDMUQ, mpc)
            traj = uq_mpc.query_mdm_with_uncertainty(
                gen,
                args.text,
                start_pose=current_pose,
                current_arm_aa=q,
                auto_cluster=cfg.uq.auto_cluster,
                mdm_frames=mdm_frames,
                frozen_body=args.frozen_body,
            )

        if cfg.preference_learning:
            learned_costs = _apply_preference_update(
                mpc, traj, q_history, cost_context,
                alpha=cfg.preference_alpha, window=cfg.preference_window,
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

        # arm_mpc_mdm must enqueue the correction manually; the UQ planners
        # already enqueue the chosen cluster mean inside query_mdm_with_uncertainty.
        if cfg.planner == "arm_mpc_mdm":
            mdm_mpc = cast(LeftArmMPCMDM, mpc)
            n_frames = traj.shape[0]
            cutoff = max(1, round(n_frames * mdm_mpc.trajectory_fraction))
            mdm_mpc.set_mdm_goal(traj[cutoff - 1])
            mdm_mpc.push_trajectory(traj[:cutoff])
            llm_traj = traj[:cutoff]
        else:
            llm_traj = traj

        if cfg.llm_cost.enabled:
            pending_llm_thread = _start_llm_cost_thread(
                mpc, args, cfg, cost_context, artifact_base_dir,
                llm_traj, q, list(q_history), body_pos, pending_llm_result,
            )

        # MDM generation can switch matplotlib to the Agg backend; restore it
        _restore_interactive_backend()

    def _on_pre_step(step: int, q: np.ndarray, q_history: list[np.ndarray]) -> None:
        nonlocal mdm_triggered
        if (
            setup.uses_mdm
            and args.text
            and step == effective_text_time
            and not mdm_triggered
        ):
            mdm_triggered = True
            _trigger_correction(q, q_history)

    def _on_post_step(_step: int, _q: np.ndarray, _q_history: list[np.ndarray]) -> None:
        nonlocal pending_llm_thread
        # Install the LLM cost once the correction trajectory has finished.
        if (
            pending_llm_thread is not None
            and isinstance(mpc, LeftArmMPCMDM)
            and mpc.mdm_tracking_complete
        ):
            print("[llm-cost] correction trajectory done — waiting for LLM query...")
            pending_llm_thread.join()
            pending_llm_thread = None
            if pending_llm_result and pending_llm_result[0] is not None:
                generated_cost = pending_llm_result.pop(0)
                mpc.set_extra_costs(
                    _append_extra_cost(mpc._extra_costs, generated_cost)  # pylint: disable=protected-access
                )
                print("[llm-cost] installed after correction trajectory completed")

    run_planning_loop(
        mpc,
        setup.arm_aa.copy(),
        cfg.steps,
        on_pre_step=_on_pre_step,
        on_post_step=_on_post_step,
        progress=True,
        progress_desc="MPC",
    )

    # --- Save video ---
    if args.save and setup.visualize:
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
