"""Headless per-backend cost-generation comparison experiment.

Run as::

    uv run python src/uncertain_feedback/experiments/run_backend_experiment.py \
        --mpc-config <yaml> [--backends llm turns agent] [--rollout-steps N] \
        [--save-video]

Drives a single ``arm_mpc_cartesian`` planner up to ``text_time`` to obtain the
UQ clustering, then generates a cost for the **chosen** cluster with each backend
(``llm`` / ``turns`` / ``agent``) and scores them on the same rollout-vs-MDM L2
axis (see
:func:`uncertain_feedback.experiments.backend_comparison.run_backend_comparison`).

The ``agent`` backend's ``codex_cmd`` is read from the base config, so use a
config whose ``llm_cost.codex_cmd`` works on this host.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from uncertain_feedback.experiments.backend_comparison import run_backend_comparison
from uncertain_feedback.experiments.experiment_pipeline import apply_persona_goals
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.planners.run import (
    build_parser,
    build_run,
    resolve_feedback_text,
    run_planning_loop,
)
from uncertain_feedback.simulated_users import PERSONAS, choose_cluster, get_persona


def _experiment_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = "Run a per-backend cost-generation comparison experiment"
    p.add_argument(
        "--backends",
        nargs="+",
        default=["llm", "turns", "agent"],
        help="Cost-generation backends to compare (default: llm turns agent).",
    )
    p.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=sorted(PERSONAS),
        help=(
            "Simulated-user persona providing hidden cost and feedback. Defaults "
            "to the config's `user:` persona."
        ),
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        dest="rollout_steps",
        help="MPC steps per --save-video rollout. Defaults to steps - text_time.",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        dest="save_video",
        help="Render an MP4 per backend rollout (same visualization as a live run).",
    )
    return p


def main() -> None:
    """Parse CLI arguments and run the per-backend cost-generation comparison."""
    args = _experiment_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    # The evaluator rolls toward the Cartesian goal via _rollout_reference_trajectory,
    # which only exists for arm_mpc_cartesian; any other planner scores every backend
    # inf and the comparison is meaningless, so gate explicitly.
    if cfg.planner != "arm_mpc_cartesian":
        raise ValueError(
            "Backend comparison requires planner: arm_mpc_cartesian (it needs a "
            f"persistent Cartesian goal for scoring); got {cfg.planner!r}."
        )
    if not cfg.llm_cost.enabled:
        raise ValueError("Experiments require llm_cost.enabled: true in the config.")

    persona_name = args.persona if args.persona is not None else cfg.user
    user = get_persona(persona_name)
    cfg = apply_persona_goals(cfg, user.name)

    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Experiment config must provide an MDM pose to generate from.")
    mpc = cast(LeftArmMPCMDMUQ, setup.mpc)

    effective_text_time = (
        args.text_time if args.text_time is not None else cfg.text_time
    )

    q = setup.q0.copy()
    q_history: list = []
    if effective_text_time > 0:
        result = run_planning_loop(mpc, q, effective_text_time)
        q_history = result.q_history
        if q_history:
            q = q_history[-1]

    mdm_frames = args.mdm_frames if args.mdm_frames is not None else cfg.mdm_frames
    feedback_text = resolve_feedback_text(args.text, setup.user)
    cluster_selector = (
        (lambda means: choose_cluster(setup.user, setup.cost_context, means))
        if cfg.uq.user_cluster and setup.user.bounds
        else None
    )
    current_pose = setup.gen.build_pose_from_arm_aa(
        setup.initial_pose,
        q_to_arm_aa(q, setup.fk.elbow_hinge_axis),
    )
    mpc.query_mdm_with_uncertainty(  # pylint: disable=no-member
        setup.gen,
        feedback_text,
        start_pose=current_pose,
        current_q=q,
        auto_cluster=cfg.uq.auto_cluster,
        mdm_frames=mdm_frames,
        frozen_body=args.frozen_body,
        cluster_selector=cluster_selector,
    )
    uq_result = mpc.last_uq_result  # pylint: disable=no-member
    if uq_result is None:
        raise RuntimeError("UQ clustering produced no result.")
    correction_traj = uq_result.cluster_means[uq_result.chosen_label]

    rollout_steps = (
        args.rollout_steps
        if args.rollout_steps is not None
        else max(1, cfg.steps - effective_text_time)
    )

    run_backend_comparison(
        mpc,
        cfg,
        feedback_text,
        correction_traj,
        q,
        q_history,
        setup.cost_context,
        artifact_base_dir,
        cfg.preference_window,
        rollout_steps,
        body_pos=setup.body_pos,
        spine3_pos=setup.spine3_pos,
        spine3_aa=setup.spine3_aa,
        initial_q=setup.q0,
        backends=args.backends,
        save_video=args.save_video,
        user=user,
    )


if __name__ == "__main__":
    main()
