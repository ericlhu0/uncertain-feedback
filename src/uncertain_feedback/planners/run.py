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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.motion_generators.base import MotionGenerator
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
    EvalState,
    GeneratedPythonCost,
    LearnablePreferenceCost,
    MpcCostContext,
    artifact_run_dir,
    build_extra_costs,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    render_prompt_images,
    replace_cost_in_composite,
    update_preference_cost,
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


def _load_initial_pose_state(
    args: argparse.Namespace,
    uses_mdm: bool,
    config_pose: Path | None = None,
    motion_generator: str = "mdm",
    motion_generator_factory: Callable[[Path | None], MotionGenerator] | None = None,
    num_denoising_steps: int | None = None,
) -> tuple[MotionGenerator | None, _InitialPoseState]:
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

    factory = motion_generator_factory or (
        lambda mp: make_motion_generator(motion_generator, mp, num_denoising_steps)
    )
    gen = factory(args.model_path)
    hml_pose = gen.load_pose(pose_path)
    arm_aa, body_pos, spine3_aa, fixed_collar_aa = gen.decode_pose(hml_pose)
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
    arm_override = np.asarray(arm_override, dtype=np.float64)
    if arm_override.shape == (4, 3):
        state.fixed_collar_aa = arm_override[0].copy()
        state.arm_aa = arm_override[1:].copy()
    elif arm_override.shape == (3, 3):
        state.arm_aa = arm_override.copy()
    else:
        raise ValueError(
            "--arm must contain shape (3, 3) for "
            "[left_shoulder, left_elbow, left_wrist], or legacy shape "
            f"(4, 3) with left_collar first; got {arm_override.shape}"
        )


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
        help=(
            "Optional .npy file with (3, 3) shoulder/elbow/wrist axis-angles. "
            "Legacy (4, 3) files are interpreted as collar, shoulder, elbow, wrist."
        ),
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


def _append_extra_cost(
    composite: CompositeTrajectoryCost,
    cost: GeneratedPythonCost,
) -> CompositeTrajectoryCost:
    """Return a composite with an additional generated cost term."""
    return CompositeTrajectoryCost([*composite.terms(), cost])


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

    gen, initial_state = _load_initial_pose_state(
        args,
        uses_mdm,
        cfg.pose,
        motion_generator=cfg.motion_generator,
        num_denoising_steps=cfg.num_denoising_steps,
    )
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
            max_playback_delta=cfg.max_playback_delta,
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
            max_playback_delta=cfg.max_playback_delta,
            trajectory_fraction=cfg.trajectory_fraction,
            n_diffusion_samples=cfg.uq.diffusion_samples,
            n_clusters=cfg.uq.n_clusters,
        )
    else:
        mpc = LeftArmMPCMDMUQ(
            **common,
            goals=[default_goal],
            advance_threshold=cfg.advance_threshold,
            max_playback_delta=cfg.max_playback_delta,
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
    reached_goal: bool = False


StepHook = Callable[[int, np.ndarray, list[np.ndarray]], None]


def run_planning_loop(
    mpc: SmplLeftArmMPC,
    q0: np.ndarray,
    n_steps: int,
    *,
    on_pre_step: StepHook | None = None,
    on_post_step: StepHook | None = None,
    stop_on_runtime_error: bool = False,
    stop_at_goal: bool = True,
    progress: bool = False,
    progress_desc: str = "MPC",
) -> LoopResult:
    """Step ``mpc`` forward up to ``n_steps``, returning the visited joint configs.

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

    With ``stop_at_goal`` (the default), the loop ends as soon as the planner
    reports it has reached its final goal (``mpc.goal_reached``) and any MDM
    correction has finished playing (``mpc.mdm_ready_to_terminate``), rather than
    always running the full ``n_steps`` and idling at the goal. ``n_steps`` is
    therefore an upper bound. ``LoopResult.reached_goal`` records whether the loop
    stopped this way.
    """
    q = np.asarray(q0, dtype=np.float64).copy()
    q_history: list[np.ndarray] = []
    iterator: Iterable[int] = range(n_steps)
    if progress:
        from tqdm import tqdm  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

        iterator = tqdm(iterator, desc=progress_desc, unit="step")
    error: str | None = None
    reached_goal = False
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
        # Stop once the goal is reached (and any correction has finished), so the
        # rollout doesn't idle at the goal for the remaining step budget.
        if stop_at_goal and mpc.mdm_ready_to_terminate and mpc.goal_reached(q):
            reached_goal = True
            break
    return LoopResult(q_history=q_history, error=error, reached_goal=reached_goal)


def _rollout_reference_trajectory(
    cfg: MpcRunConfig,
    current_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> np.ndarray | None:
    """Roll the MPC toward its original Cartesian goal, ignoring the correction.

    Builds a headless :class:`ArmMPCCartesianNoMDM` from ``current_q`` carrying only
    the configured comfort costs (no MDM correction, no LLM-generated cost) and steps
    it toward ``cfg.cartesian.goals`` so the cost generator can see what the arm was
    driving toward before the correction — and avoid blocking it. With no MDM phase
    (``mdm_ready_to_terminate`` is always ``True``) the loop stops as soon as the wrist
    reaches the goal, so the trajectory ends at the goal rather than idling there for the
    full ``cfg.steps``. Returns ``(T, 3, 3)``, or ``None`` for planners without a
    persistent Cartesian goal.
    """
    if cfg.planner not in ("arm_mpc_cartesian", "arm_mpc_cartesian_no_mdm"):
        return None
    if not cfg.cartesian.goals:
        return None

    planner = ArmMPCCartesianNoMDM(
        cartesian_goals=[np.asarray(g, dtype=np.float64) for g in cfg.cartesian.goals],
        initial_arm_aa=current_q,
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
        extra_costs=base_extra_costs,
    )
    q0 = np.asarray(current_q, dtype=np.float64).copy()
    result = run_planning_loop(
        planner, q0, max(1, cfg.steps), stop_on_runtime_error=True
    )
    return np.asarray([q0, *result.q_history], dtype=np.float64)


def _assemble_full_correction_traj(
    cfg: MpcRunConfig,
    q_history: list[np.ndarray],
    correction_traj: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> np.ndarray:
    """Assemble the entire corrected path: history → correction → goal continuation.

    This is the target shown (green) in the cost-feedback comparison so the cost
    generator sees the whole intended trajectory, not just the MDM correction
    segment. The three segments are the executed pre-correction history, the MDM
    correction itself, and a comfort-only goal-seeking continuation rolled from the
    correction's endpoint (so the arm still reaches the goal afterwards). The
    continuation is empty for planners without a Cartesian goal, leaving just
    history + correction. The duplicated seam frame at the correction endpoint is
    dropped.
    """
    correction_traj = np.asarray(correction_traj, dtype=np.float64)
    segments: list[np.ndarray] = []
    if q_history:
        segments.append(np.asarray(q_history, dtype=np.float64))
    segments.append(correction_traj)
    post = _rollout_reference_trajectory(
        cfg, correction_traj[-1], context, base_extra_costs,
        body_pos, spine3_pos, spine3_aa,
    )
    if post is not None and len(post) > 1:
        segments.append(post[1:])
    return np.concatenate(segments, axis=0)


def _make_cost_eval_rollout(
    cfg: MpcRunConfig,
    current_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> Callable[[GeneratedPythonCost], np.ndarray | None]:
    """Return a closure rolling the goal-seeking MPC with a candidate cost installed.

    The returned function appends the candidate generated cost to the comfort costs
    and rolls toward the original Cartesian goal (reusing
    :func:`_rollout_reference_trajectory`), yielding the ``(T, 3, 3)`` trajectory the
    cost evaluator compares against the MDM correction. Returns ``None`` for planners
    without a persistent Cartesian goal. Each call builds a fresh headless planner, so
    the live MPC's goals/warm-start are untouched.
    """

    def rollout(cost: GeneratedPythonCost) -> np.ndarray | None:
        extra = _append_extra_cost(base_extra_costs, cost)
        return _rollout_reference_trajectory(
            cfg, current_q, context, extra, body_pos, spine3_pos, spine3_aa
        )

    return rollout


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
    pending_cost: GeneratedPythonCost | None = None

    def _trigger_correction(q: np.ndarray, q_history: list[np.ndarray]) -> None:
        """Generate the MDM/UQ correction (and optional LLM cost) at text_time."""
        nonlocal pre_mdm_vis, pending_cost
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
            # Ground the cost in all candidate paths when the planner clustered the
            # correction, so the overlay shows every cluster mean with the chosen one
            # highlighted. Non-UQ planners pass a single path.
            candidate_trajs: dict[int, np.ndarray] | None = None
            highlight_label: int | None = None
            uqr = getattr(mpc, "last_uq_result", None)
            if uqr is not None:
                candidate_trajs = uqr.cluster_means
                highlight_label = uqr.chosen_label

            # Roll the original-goal MPC out (no correction, no generated cost) so
            # the cost generator sees what the arm was driving toward and avoids
            # blocking it. mpc._extra_costs is still the pre-correction comfort set —
            # the generated cost is held in pending_cost, not yet installed.
            reference_traj = _rollout_reference_trajectory(
                cfg, q, cost_context, mpc._extra_costs,  # pylint: disable=protected-access
                body_pos, setup.spine3_pos, spine3_aa,
            )
            goal_pos = (
                np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
                if reference_traj is not None and cfg.cartesian.goals
                else None
            )

            # Build the shared context/summaries/images once, then let the
            # configured backend (llm / turns / agent) generate the cost. All three
            # are constructed and called identically; only the factory branches.
            full_correction_traj = _assemble_full_correction_traj(
                cfg, list(q_history), llm_traj, cost_context, mpc._extra_costs,  # pylint: disable=protected-access
                body_pos, setup.spine3_pos, spine3_aa,
            )
            generated_context = build_generated_cost_context(
                cost_context, q, llm_traj, list(q_history),
                window=cfg.preference_window, body_pos=body_pos,
                reference_traj=reference_traj,
                full_correction_traj=full_correction_traj,
            )
            summaries = build_motion_summaries(
                generated_context, cartesian_goal=goal_pos
            )
            run_dir = artifact_run_dir(artifact_base_dir, cfg.llm_cost.artifact_dir)
            images: dict[str, Path] = {}
            if cfg.llm_cost.use_images:
                images = render_prompt_images(
                    generated_context, run_dir / "images",
                    candidate_trajs, highlight_label,
                    reference_traj=reference_traj, goal_pos=goal_pos,
                )

            # Generate synchronously here (the live viz is already closed for the
            # MDM compute, so this adds no extra freeze). The cost is held and
            # installed once the correction trajectory finishes (see _on_post_step).
            cost_eval_rollout = _make_cost_eval_rollout(
                cfg, q, cost_context, mpc._extra_costs,  # pylint: disable=protected-access
                body_pos, setup.spine3_pos, spine3_aa,
            )
            eval_state = EvalState(
                cfg=cfg,
                current_q=q,
                correction_traj=llm_traj,
                q_history=list(q_history),
                window=cfg.preference_window,
                cost_context=cost_context,
                base_extra_costs=mpc._extra_costs,  # pylint: disable=protected-access
                body_pos=body_pos,
                spine3_pos=setup.spine3_pos,
                spine3_aa=spine3_aa,
                reference_traj=reference_traj,
                full_correction_traj=full_correction_traj,
            )
            generator = create_cost_generator(
                cfg.llm_cost, generated_context, args.text,
                summaries=summaries, run_dir=run_dir, images=images, mpc=mpc,
                rollout_fn=cost_eval_rollout, eval_state=eval_state,
            )
            pending_cost = generator.generate(install=False)

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
        nonlocal pending_cost
        # Install the LLM cost once the correction trajectory has finished.
        if (
            pending_cost is not None
            and isinstance(mpc, LeftArmMPCMDM)
            and mpc.mdm_tracking_complete
        ):
            mpc.set_extra_costs(
                _append_extra_cost(mpc._extra_costs, pending_cost)  # pylint: disable=protected-access
            )
            pending_cost = None
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
