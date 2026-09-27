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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.cost_generation import (
    CombineCostGenerator,
    CostRound,
    artifact_run_dir,
    build_motion_summaries,
    create_cost_generator,
    render_prompt_images,
)
from uncertain_feedback.envs import make_env
from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.robot_preview import RobotPlanPreviewEnv
from uncertain_feedback.evaluation_mechanism import EvalState
from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.steering import build_steering_spec
from uncertain_feedback.planners.correction_session import (
    CorrectionRoundResult,
    CorrectionSession,
    CorrectionTrajectoryResult,
    TriggerReason,
)
from uncertain_feedback.planners.interactive import OperatorPause
from uncertain_feedback.planners.mpc import ArmMPC
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    LearnablePreferenceCost,
    build_extra_costs,
    build_generated_cost_context,
    replace_cost_in_composite,
    replace_generated_costs,
    update_preference_cost,
)
from uncertain_feedback.planners.mpc.goal_spaces import goal_point
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    anchor_q_trajectory,
)
from uncertain_feedback.planners.mpc.rollout import (
    assemble_full_correction_traj,
    rollout_reference_trajectory,
    run_planning_loop,
)
from uncertain_feedback.simulated_users import (
    SimulatedUser,
    choose_cluster,
    get_persona,
)
from uncertain_feedback.uncertainty.clustering import make_clusterer
from uncertain_feedback.utils.plot import ArmVisualizer


def build_parser() -> argparse.ArgumentParser:
    """Build the shared argument parser for the MPC run entry points."""
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
            "Optional .npy file with (3, 3) shoulder/elbow/wrist axis-angles "
            "replacing the start arm."
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
    p.add_argument(
        "--env-video",
        type=Path,
        default=None,
        dest="env_video",
        help="Save the execution env's rendering of the run (.mp4 or .gif)",
    )

    # --- MDM args (feedback-configured runs) ---
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help=(
            "Natural-language MDM motion description (mdm/uq planners only). "
            "Defaults to the configured user's feedback line when that user "
            "has hidden bounds, else 'move my arm up'."
        ),
    )
    p.add_argument(
        "--text-time",
        type=int,
        default=None,
        dest="text_time",
        help="MPC step at which MDM generation is triggered (overrides YAML text_time)",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Pause when the operator presses enter and read that round's "
            "correction from stdin, instead of injecting --text at text_time "
            "(which is then ignored). For live runs with a real person."
        ),
    )
    p.add_argument(
        "--save-motion",
        type=Path,
        default=None,
        dest="save_motion",
        help="Save the raw MDM full-body motion video to this path (direct-MDM runs only)",
    )
    p.add_argument(
        "--mdm-frames",
        type=int,
        default=None,
        dest="mdm_frames",
        help=(
            "Exact number of MDM frames to return (1-189: the pinned prefix is "
            "sampled on top, within MDM's 196-frame limit). Default is 120."
        ),
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


def resolve_feedback_text(args_text: str | None, user: SimulatedUser) -> str:
    """Return the MDM instruction: explicit --text > restricted user's line > default."""
    if args_text:
        return args_text
    if user.bounds and user.feedback_text:
        return user.feedback_text
    return "move my arm up"


def _restore_interactive_backend() -> None:
    if plt.get_backend().lower() == "agg":
        for backend in ("Qt5Agg", "TkAgg", "Qt6Agg", "WXAgg", "MacOSX"):
            try:
                plt.switch_backend(backend)
                break
            except Exception:  # pylint: disable=broad-exception-caught
                continue


def _get_vis(mpc: ArmMPC) -> ArmVisualizer | None:
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
    mpc: ArmMPC,
    mdm_traj: np.ndarray,
    human: Human,
    alpha: float,
    window: int,
) -> list[LearnablePreferenceCost]:
    """Update configured preference bounds from MDM/MPC discrepancy."""
    costs = _iter_learnable_costs(mpc._extra_costs)  # pylint: disable=protected-access
    if not costs:
        return []
    recent_aa = human.arm_aa_from_q(human.history[-window:])
    mdm_aa = human.arm_aa_from_q(mdm_traj)
    updated_costs: list[LearnablePreferenceCost] = []
    extra_costs = mpc._extra_costs  # pylint: disable=protected-access

    for cost in costs:
        mpc_values = cost.feature_values(recent_aa)
        mdm_values = cost.feature_values(mdm_aa)
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
    """Planner, person, and costs for one run.

    Shared by the single-run entry point (``main``) and the experiment runner so
    both reach an identical planner before stepping. ``human`` is the person at
    the start, as the env measured them.
    """

    mpc: ArmMPC
    gen: MotionGenerator | None
    human: Human
    uses_mdm: bool
    visualize: bool
    compact: bool
    user: SimulatedUser
    env: ExecutionEnv
    extra_costs: CompositeTrajectoryCost


def build_run(
    args: argparse.Namespace,
    cfg: MpcRunConfig,
    motion_generator_factory: Callable[[Path | None], MotionGenerator] | None = None,
) -> RunSetup:
    """Load the person, costs, and planner for a single run."""
    uses_mdm = cfg.feedback is not None
    visualize = args.live or (args.save is not None)
    # Compact (1-panel) rendering when saving without live view.
    compact = (args.save is not None) and not args.live

    pose_path = args.pose if args.pose is not None else cfg.pose
    if uses_mdm and pose_path is None:
        pose_path = MDM_ROOT / "demo_pose.pt"
    arm = np.load(args.arm) if args.arm is not None else cfg.arm
    human = Human(pose=pose_path, arm=arm)
    gen: MotionGenerator | None = None
    if uses_mdm:
        factory = motion_generator_factory or (
            lambda model_path: make_motion_generator(
                cfg.motion_generator, model_path, seed=cfg.seed
            )
        )
        gen = factory(args.model_path)

    env = make_env(cfg.env, **cfg.env_params)
    # Envs that measure the person (env: real) report where the arm actually
    # is, their segment lengths, and where they sit, which moves the torso
    # anchor off the config's. Goals and costs are spine3-relative, so plan
    # against the person the env reports.
    human = env.measure(human)

    user = get_persona(cfg.user)
    extra_costs = build_extra_costs(cfg.costs, human)
    if user.joint_limits:
        extra_costs = CompositeTrajectoryCost([*extra_costs.terms(), user.limit_cost()])

    if (cfg.robot_actions is not None or cfg.constraints) and cfg.env not in (
        "real",
        "sim_mannequin",
    ):
        raise ValueError(
            "robot_actions and constraints need an env with a robot "
            f"('real' or 'sim_mannequin'), got '{cfg.env}'."
        )

    if cfg.cartesian is not None:
        print(
            f"Initial wrist position (spine3-relative): {human.wrist_from_q(human.q)}"
        )

    mpc = ArmMPC(
        human,
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=visualize,
        extra_costs=extra_costs,
        seed=cfg.seed,
        env=env,
        cartesian=cfg.cartesian,
        feedback=cfg.feedback,
        constraints=cfg.constraints,
        robot_actions=cfg.robot_actions,
        # Without this ArmMPC falls back to its own XyzPositionClusterer and the
        # config's `clusterer:` is silently dropped — the demo runner and
        # evaluation/approaches/system.py both resolve the name.
        clusterer=(
            None
            if cfg.feedback is None or cfg.feedback.uq is None
            else make_clusterer(
                cfg.feedback.uq.clusterer, cfg.feedback.uq.n_clusters, fk=human.fk
            )
        ),
    )

    mpc.set_visualization_mode(capture=args.save is not None, compact=compact)
    # Show the goal to envs that can draw it (env: real's live view). A Cartesian
    # goal pins the wrist only, so the pose shown is the nearest configuration
    # reaching it — solved against the anchor the env reported, since the goal is
    # relative to that.
    shown_goal = (
        goal_point(cfg.cartesian.goals[0]) if cfg.cartesian is not None else None
    )
    if shown_goal is not None:
        env.show_goal(human.q_from_wrist(shown_goal))
    elif cfg.cartesian is None:
        # Default goal display: arm raised from the initial pose. Shoulder
        # slot, not clavicle — the planner's actions cannot move the girdle.
        default_goal = human.q
        default_goal[4] += 0.7
        env.show_goal(default_goal)

    return RunSetup(
        mpc=mpc,
        gen=gen,
        human=human,
        uses_mdm=uses_mdm,
        visualize=visualize,
        compact=compact,
        user=user,
        env=env,
        extra_costs=extra_costs,
    )


def _preview_env(setup: RunSetup) -> RobotPlanPreviewEnv:
    """Kinematic double of the run's robot env, frozen at the measured state.

    Snapshots the robot chain, measured grasp, joint state, limits, and step
    cap, and delegates exact IK back to the env itself, so a rollout against it
    is the solve about to run. The env's grasp must already be measured — the
    preview captures it before calling the plan.
    """
    env = setup.env
    human = setup.human
    q0 = human.q
    return RobotPlanPreviewEnv(
        fk=human.fk,
        chain=env.robot_fk(),
        grasp=env.current_grasp(q0),
        robot_q=env.current_robot_q(),
        joint_limits=env.robot_joint_limits(),
        q_ref=q0,
        spine3_pos=human.spine3_pos,
        spine3_aa=human.spine3_aa,
        ik_env=env,
        max_joint_delta=env.robot_max_joint_delta(),
    )


def _rollout_robot_reference_trajectory(
    cfg: MpcRunConfig,
    setup: RunSetup,
    on_step: Callable[[np.ndarray, np.ndarray | None], None],
) -> None:
    """Roll the robot-action planner offline against a kinematic stand-in.

    The rollout is the actual robot-space solve about to run — the previewed
    robot and arm stay consistent by construction instead of the robot chasing
    a human-space plan through IK. Each planned step is reported through
    ``on_step(q, robot_q)`` as its solve finishes, so the env can draw the
    rollout while it is being planned.
    """
    preview_env = _preview_env(setup)
    planner = ArmMPC(
        setup.human,
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=False,
        extra_costs=setup.extra_costs,
        seed=cfg.seed,
        env=preview_env,
        cartesian=cfg.cartesian,
        robot_actions=cfg.robot_actions,
    )

    def report(_step: int, human: Human) -> None:
        on_step(human.q, preview_env.robot_trajectory[-1])

    run_planning_loop(
        planner,
        max(1, cfg.steps),
        on_post_step=report,
        stop_on_runtime_error=True,
    )


def _rollout_gated_reference_trajectory(
    cfg: MpcRunConfig,
    setup: RunSetup,
    on_step: Callable[[np.ndarray, np.ndarray | None], None],
) -> None:
    """Roll the IK-gated planner offline, gate included.

    The gate *is* the planner: previewing the ungated planner instead shows a
    trajectory the run will not take, and — since the gate refuses exactly the
    frames whose grasp the robot cannot hold — shows it diverging at the poses
    the run is built to refuse. The stand-in delegates exact IK to the env, so
    the reachability enforced here is the one execution will enforce, and the
    robot reported to ``on_step`` is the one the gate reasoned against rather
    than a second IK chasing the arm.
    """
    preview_env = _preview_env(setup)
    planner = ArmMPC(
        setup.human,
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=False,
        extra_costs=setup.extra_costs,
        seed=cfg.seed,
        env=preview_env,
        cartesian=cfg.cartesian,
        constraints=cfg.constraints,
    )

    def report(_step: int, human: Human) -> None:
        on_step(human.q, preview_env.robot_trajectory[-1])

    run_planning_loop(
        planner,
        max(1, cfg.steps),
        on_post_step=report,
        stop_on_runtime_error=True,
    )


def _rollout_human_reference_trajectory(
    cfg: MpcRunConfig,
    setup: RunSetup,
    on_step: Callable[[np.ndarray, np.ndarray | None], None],
) -> None:
    """Roll a plain human-action Cartesian planner offline, kinematically."""
    rollout_reference_trajectory(cfg, setup.human, setup.extra_costs, on_step=on_step)


_PreviewRollout = Callable[
    [MpcRunConfig, RunSetup, Callable[[np.ndarray, np.ndarray | None], None]], None
]


def _select_preview_rollout(cfg: MpcRunConfig) -> _PreviewRollout | None:
    """Pick the offline rollout that runs the planner about to run live.

    A preview is only worth watching if it carries the same feasibility
    constraints and action space as the run: previewing the constrained
    planner unconstrained once drew the run walking into exactly the poses
    the constraints exist to discard. The constraint > robot > human
    precedence is deliberate (the loader forbids constraints + robot_actions
    together today, but the ordering keeps this correct if that changes).
    Feedback corrections do not exist until the user has spoken, so only the
    goal-space phase can be previewed at all — ``None`` without one.
    """
    if cfg.cartesian is None:
        return None
    if cfg.constraints:
        return _rollout_gated_reference_trajectory
    if cfg.robot_actions is not None:
        return _rollout_robot_reference_trajectory
    return _rollout_human_reference_trajectory


def preview_planned_trajectory(cfg: MpcRunConfig, setup: RunSetup) -> bool:
    """Show the env the plan before it is executed; ``False`` means abort the run.

    The plan is a headless rollout of the same planner, goals, and costs from the
    run's start configuration — for ``env: real`` that start configuration and
    every goal are the *measured* ones, so what the env is handed is the
    trajectory it is about to run on the person (see :meth:`RealEnv.preview`).
    Kinematic execution means it assumes perfect tracking; the real loop will
    drift from it.

    Planners whose trajectory is not known before the run have nothing to
    preview: an MDM correction only exists once the user has spoken, and the
    pre-correction phase of the plain joint-goal planners is not what the run is
    for. The rollout is deferred to the env because it costs a full MPC solve per
    step, and only envs that show it need it — each planned step streams through
    the env's ``on_step`` callback as its solve finishes, so the env draws the
    rollout live while it is being planned. Which rollout runs is
    :func:`_select_preview_rollout`: the planner previewed is the planner that
    will run, including whatever feasibility constraints it carries and the
    robot state those constraints reason against, so a plan the run would
    refuse is never drawn as one it would take. The planners with a robot
    report their own robot joints per step, so the animation shows the planned
    robot motion rather than an IK chase.
    """
    rollout = _select_preview_rollout(cfg)
    if rollout is None:
        return True

    def plan(on_step: Callable[[np.ndarray, np.ndarray | None], None]) -> None:
        rollout(cfg, setup, on_step)

    return setup.env.preview(plan)


def run_repeated_correction_session(
    args: argparse.Namespace,
    cfg: MpcRunConfig,
    setup: RunSetup,
    artifact_base_dir: Path,
    preference_output_path: Path,
    *,
    trajectory_index: int = 0,
    prior_rounds: tuple[CostRound, ...] = (),
    prior_unified_cost: GeneratedPythonCost | None = None,
) -> tuple[CorrectionTrajectoryResult, list[ArmVisualizer], tuple[CostRound, ...]]:
    """Run one feedback-configured planner with repeated interruptions."""
    mpc = setup.mpc
    feedback_cfg = cfg.feedback
    assert feedback_cfg is not None
    assert setup.gen is not None
    gen = setup.gen
    feedback_text = resolve_feedback_text(args.text, setup.user)
    # The operator's stdin watcher must be the only stdin reader. RealEnv's
    # "press Enter to start tracking" confirmation fires when the grasp is
    # first established — normally at the first MPC step, after the watcher
    # thread has taken stdin, and the two then race for the typed line and the
    # run freezes at the prompt. Establish the grasp (and consume that prompt)
    # first.
    if args.interactive and cfg.env == "real":
        setup.env.current_grasp(setup.human.q)
    # Interactive runs take the words from whoever is being moved, so the
    # scripted step trigger would only inject a correction nobody asked for.
    operator = OperatorPause() if args.interactive else None
    effective_text_time = (
        None
        if operator is not None
        else (args.text_time if args.text_time is not None else feedback_cfg.text_time)
    )
    configured_base_costs = replace_generated_costs(
        mpc._extra_costs, None  # pylint: disable=protected-access
    )
    artifact_root = (
        artifact_run_dir(artifact_base_dir, cfg.llm_cost.artifact_dir)
        / f"trajectory_{trajectory_index:02d}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    closed_visualizers: list[ArmVisualizer] = []
    runtime_rounds: list[
        tuple[CostRound, EvalState, Any, dict[str, Any], dict[str, Path]]
    ] = []

    def handle_correction(
        step: int,
        human: Human,
        reason: TriggerReason,
        violation: float | None,
        local_index: int,
    ) -> CorrectionRoundResult:
        nonlocal feedback_text
        if operator is not None:
            feedback_text = operator.feedback(step)
        old_suffix = mpc.remaining_mdm_trajectory(human.q)
        closed = mpc.close_visualizer()
        if closed is not None:
            closed_visualizers.append(closed)
        round_index = len(prior_rounds) + local_index
        round_dir = artifact_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if old_suffix is not None:
            np.save(round_dir / "interrupted_reference.npy", old_suffix)

        # MDM is conditioned on the arm's recent trajectory (prefix=True): the
        # pinned prefix ends at the current configuration and is stripped from
        # the generated motion by the generator.
        mdm_frames = (
            args.mdm_frames if args.mdm_frames is not None else feedback_cfg.frames
        )
        save_path: str | None = None
        if args.save_motion is not None:
            path = Path(args.save_motion)
            save_path = str(
                path.with_name(f"{path.stem}_round_{round_index:02d}{path.suffix}")
            )

        if feedback_cfg.uq is None:
            positions = gen.generate_positions(
                feedback_text,
                human,
                prefix=True,
                num_frames=mdm_frames,
                frozen_body=args.frozen_body,
                save_path=save_path,
            )[0]
            traj = human.ik_q_from_positions(positions[:, LEFT_ARM_CHAIN_INDICES])
            if feedback_cfg.anchor_correction:
                traj = anchor_q_trajectory(traj, human.q)
            cutoff = max(1, round(len(traj) * mpc.trajectory_fraction))
            llm_traj = traj[:cutoff]
            mpc.set_mdm_goal(llm_traj[-1])
            mpc.push_trajectory(llm_traj)
        else:
            selector = (
                (lambda means: choose_cluster(setup.user, human, means))
                if feedback_cfg.uq.user_cluster and setup.user.bounds
                else None
            )
            spec = build_steering_spec(
                gen, setup.user, feedback_cfg.uq.steering, cfg.seed
            )
            if spec is not None:
                print(f"steering: mode {spec.config.mode}")
            traj = mpc.query_mdm_with_uncertainty(
                gen,
                feedback_text,
                prefix=True,
                auto_cluster=feedback_cfg.uq.auto_cluster,
                default_scale=feedback_cfg.uq.scale,
                mdm_frames=mdm_frames,
                frozen_body=args.frozen_body,
                cluster_selector=selector,
                steering=spec,
            )
            cutoff = max(1, round(len(traj) * mpc.trajectory_fraction))
            llm_traj = traj[:cutoff]
        np.save(round_dir / "correction.npy", llm_traj)
        uq_result = mpc.last_uq_result
        if uq_result is not None:
            # Anchored the way the chosen mean already is, so the candidates a
            # rendering overlays all leave the arm's actual pose instead of each
            # teleporting off it from its own raw frame 0.
            np.savez(
                round_dir / "cluster_means.npz",
                chosen_label=uq_result.chosen_label,
                **{
                    f"cluster_{label:02d}": (
                        human.arm_aa_from_q(
                            anchor_q_trajectory(human.q_from_arm_aa(mean), human.q)
                        )
                        if feedback_cfg.anchor_correction
                        else mean
                    )
                    for label, mean in uq_result.cluster_means.items()
                },
            )

        if cfg.preference_learning:
            learned = _apply_preference_update(
                mpc,
                llm_traj,
                human,
                alpha=cfg.preference_alpha,
                window=cfg.preference_window,
            )
            if learned:
                _save_learned_preference_yaml(
                    args.mpc_config, preference_output_path, learned
                )

        generated: GeneratedPythonCost | None = None
        cost_round: CostRound | None = None
        if cfg.llm_cost.enabled:
            candidate_trajs: dict[int, np.ndarray] | None = None
            highlight_label: int | None = None
            rejected_trajs: tuple[np.ndarray, ...] = ()
            uqr = getattr(mpc, "last_uq_result", None)
            if uqr is not None:
                candidate_trajs = {
                    uqr.chosen_label: uqr.cluster_means[uqr.chosen_label]
                }
                highlight_label = uqr.chosen_label
            reference_q = old_suffix
            if reference_q is None:
                reference = rollout_reference_trajectory(
                    cfg, human, configured_base_costs
                )
                reference_q = None if reference is None else reference.history
            goal_pos = (
                goal_point(cfg.cartesian.goals[0])
                if cfg.cartesian is not None
                else None
            )
            cartesian_threshold = (
                cfg.cartesian.threshold if cfg.cartesian is not None else 0.01
            )
            full_correction_q = assemble_full_correction_traj(
                cfg, human, llm_traj, configured_base_costs
            )
            context = build_generated_cost_context(
                human,
                llm_traj,
                window=cfg.preference_window,
                reference_traj=reference_q,
                full_correction_traj=full_correction_q,
                cartesian_goal=goal_pos,
                cartesian_threshold=cartesian_threshold,
                rejected_trajs=rejected_trajs,
            )
            summaries = build_motion_summaries(context, cartesian_goal=goal_pos)
            images: dict[str, Path] = {}
            if cfg.llm_cost.use_images:
                images = render_prompt_images(
                    context,
                    round_dir / "images",
                    candidate_trajs,
                    highlight_label,
                    reference_traj=reference_q,
                    goal_pos=goal_pos,
                )
            eval_state = EvalState(
                cfg=cfg,
                human=human,
                correction_traj=llm_traj,
                window=cfg.preference_window,
                base_extra_costs=configured_base_costs,
                reference_traj=reference_q,
                full_correction_traj=full_correction_q,
                cartesian_goal=goal_pos,
                cartesian_threshold=cartesian_threshold,
                rejected_trajs=rejected_trajs,
            )
            state_path = round_dir / "state.pkl"
            eval_state.save(state_path)
            generator = create_cost_generator(
                cfg.llm_cost,
                context,
                feedback_text,
                summaries=summaries,
                run_dir=round_dir / "cost_generation",
                images=images,
                mpc=None,
                rollout_fn=eval_state.make_rollout_fn(),
                eval_state=eval_state,
            )
            generated = generator.generate(install=False)
            if generated is not None:
                mpc.set_extra_costs(
                    _append_extra_cost(
                        mpc._extra_costs, generated  # pylint: disable=protected-access
                    )
                )
                goal = (
                    (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
                    if goal_pos is not None
                    else None
                )
                cost_round = CostRound(
                    index=round_index,
                    goal=goal,
                    feedback_text=feedback_text,
                    trigger_step=step,
                    round_dir=round_dir.resolve(),
                    state_path=state_path.resolve(),
                    cost_code=generated.code,
                    params=generated.params,
                    summaries=summaries,
                    image_paths=tuple(path.resolve() for path in images.values()),
                    trajectory_index=trajectory_index,
                    trigger_reason=reason,
                    trigger_violation=violation,
                )
                runtime_rounds.append(
                    (cost_round, eval_state, context, summaries, images)
                )
                print(f"[llm-cost] stacked correction cost {round_index}")
        _restore_interactive_backend()
        return CorrectionRoundResult(
            round_index=round_index,
            trajectory_index=trajectory_index,
            trigger_step=step,
            trigger_reason=reason,
            trigger_violation=violation,
            feedback_text=feedback_text,
            correction_traj=llm_traj,
            generated_cost=generated,
            cost_round=cost_round,
            artifact_dir=round_dir,
        )

    def finish(
        rounds: Sequence[CorrectionRoundResult],
    ) -> GeneratedPythonCost | None:
        all_cost_rounds = [
            *prior_rounds,
            *(round_.cost_round for round_ in rounds if round_.cost_round),
        ]
        history_path = artifact_root / "history.json"
        history_path.write_text(
            json.dumps([round_.to_json() for round_ in all_cost_rounds], indent=2),
            encoding="utf-8",
        )
        if not runtime_rounds:
            return prior_unified_cost
        if len(all_cost_rounds) == 1:
            unified = runtime_rounds[-1][0]
            cost = next(r.generated_cost for r in rounds if r.cost_round is unified)
            assert cost is not None
            mpc.set_extra_costs(
                replace_generated_costs(
                    mpc._extra_costs, cost  # pylint: disable=protected-access
                )
            )
            return cost
        _, eval_state, context, summaries, images = runtime_rounds[-1]
        combinator = CombineCostGenerator(
            context=context,
            instruction=feedback_text,
            summaries=summaries,
            run_dir=artifact_root / f"combine_after_trajectory_{trajectory_index:02d}",
            images=images,
            use_images=cfg.llm_cost.use_images,
            model=cfg.llm_cost.model,
            strict=cfg.llm_cost.strict,
            mpc=None,
            rollout_fn=eval_state.make_rollout_fn(),
            eval_state=eval_state,
            codex_cmd=cfg.llm_cost.codex_cmd,
            rounds=all_cost_rounds,
        )
        combined = combinator.generate(install=False)
        if combined is not None:
            mpc.set_extra_costs(
                replace_generated_costs(
                    mpc._extra_costs, combined  # pylint: disable=protected-access
                )
            )
            return combined
        print("[combine] failed; retaining stacked generated costs")
        return prior_unified_cost

    session = CorrectionSession(
        mpc=mpc,
        user=setup.user,
        feedback_text=feedback_text,
        trigger_threshold=cfg.corrections.trigger_threshold,
        text_time=effective_text_time,
        artifact_dir=artifact_root,
        handle_correction=handle_correction,
        finish=finish,
        trajectory_index=trajectory_index,
        prior_rounds=prior_rounds,
        prior_unified_cost=prior_unified_cost,
        operator_requested=operator.requested if operator is not None else None,
    )
    result = session.run_trajectory(cfg.steps, progress=True, progress_desc="MPC")
    np.save(artifact_root / "executed_trajectory.npy", result.loop_result.human.history)
    summary = {
        "trajectory_index": trajectory_index,
        "correction_count": len(result.rounds),
        "reached_goal": result.loop_result.reached_goal,
        "error": result.loop_result.error,
        "corrections": [
            {
                "round_index": round_.round_index,
                "trigger_step": round_.trigger_step,
                "trigger_reason": round_.trigger_reason,
                "trigger_violation": round_.trigger_violation,
            }
            for round_ in result.rounds
        ],
    }
    (artifact_root / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    cost_rounds = tuple(
        [*prior_rounds, *(r.cost_round for r in result.rounds if r.cost_round)]
    )
    return result, closed_visualizers, cost_rounds


def main() -> None:
    """Parse CLI arguments and run the configured MPC planner."""
    args = build_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    # Every video is written after the MDM loader has os.chdir()ed into its
    # submodule, so a relative path would land there (or fail) instead of here.
    for dest in ("save", "env_video", "save_motion"):
        if getattr(args, dest) is not None:
            setattr(args, dest, Path(getattr(args, dest)).resolve())
    cfg = load_mpc_config(args.mpc_config)
    # Checked before build_run, which on env: real already talks to the hardware.
    if args.interactive and cfg.feedback is None:
        raise ValueError(
            "--interactive needs a feedback: section to turn the typed "
            "feedback into a correction."
        )
    setup = build_run(args, cfg)
    preference_output_path = args.preference_output or _default_preference_output_path(
        args.mpc_config
    )

    if not preview_planned_trajectory(cfg, setup):
        print("aborted at the plan preview; nothing was executed")
        return

    closed_visualizers: list[ArmVisualizer] = []
    if setup.uses_mdm:
        _, closed_visualizers, _ = run_repeated_correction_session(
            args, cfg, setup, artifact_base_dir, preference_output_path
        )
    else:
        run_planning_loop(
            setup.mpc,
            cfg.steps,
            progress=True,
            progress_desc="MPC",
        )

    if args.save and setup.visualize:
        vis = _get_vis(setup.mpc)
        if vis is not None:
            prior_frames = [
                frame
                for closed in closed_visualizers
                for frame in closed._frame_bufs  # pylint: disable=protected-access
            ]
            if prior_frames:
                vis.prepend_frames(prior_frames)
            vis.finish_live(str(args.save), fps=args.fps)

    if args.env_video is not None:
        setup.env.save_video(args.env_video, fps=args.fps)

    if args.live:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
