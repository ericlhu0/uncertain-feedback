"""Headless per-cluster LLM-cost comparison experiment.

Run as::

    uv run python src/uncertain_feedback/experiments/run_experiment.py \
        --mpc-config <yaml> [--rollout-steps N] [--save-video]

This drives a single planner up to ``text_time`` to obtain the UQ clustering,
then generates and rolls out one LLM cost per cluster (see
:func:`uncertain_feedback.experiments.cluster_comparison.run_cluster_comparison`).
Pass ``--save-video`` to render each rollout — it uses the same ``ArmVisualizer``
layout as a live run, so the only difference is whether the file is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.run import build_parser, build_run, run_planning_loop
from uncertain_feedback.experiments.cluster_comparison import run_cluster_comparison

_UQ_PLANNERS = {"arm_mpc_mdm_uq", "arm_mpc_cartesian"}


def _experiment_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = "Run a per-cluster LLM-cost comparison experiment"
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        dest="rollout_steps",
        help="MPC steps to roll out per cluster. Defaults to steps - text_time.",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        dest="save_video",
        help="Render an MP4 per cluster rollout (same visualization as a live run).",
    )
    return p


def main() -> None:
    args = _experiment_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    if cfg.planner not in _UQ_PLANNERS:
        raise ValueError(
            "Experiments require a UQ planner (arm_mpc_mdm_uq or arm_mpc_cartesian); "
            f"got {cfg.planner!r}."
        )
    if not cfg.llm_cost.enabled:
        raise ValueError("Experiments require llm_cost.enabled: true in the config.")

    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Experiment config must provide an MDM pose to generate from.")
    mpc = cast(LeftArmMPCMDMUQ, setup.mpc)

    effective_text_time = args.text_time if args.text_time is not None else cfg.text_time

    # Step up to the generation point, mirroring the single run's pre-text_time loop.
    q = setup.arm_aa.copy()
    q_history: list = []
    if effective_text_time > 0:
        result = run_planning_loop(mpc, q, effective_text_time)
        q_history = result.q_history
        if q_history:
            q = q_history[-1]

    mdm_frames = args.mdm_frames if args.mdm_frames is not None else cfg.mdm_frames
    current_pose = setup.gen.build_pose_from_arm_aa(setup.initial_pose, q)
    mpc.query_mdm_with_uncertainty(
        setup.gen,
        args.text,
        start_pose=current_pose,
        current_arm_aa=q,
        auto_cluster=cfg.uq.auto_cluster,
        mdm_frames=mdm_frames,
        frozen_body=args.frozen_body,
    )
    uq_result = mpc.last_uq_result
    if uq_result is None:
        raise RuntimeError("UQ clustering produced no result.")

    rollout_steps = (
        args.rollout_steps
        if args.rollout_steps is not None
        else max(1, cfg.steps - effective_text_time)
    )

    run_cluster_comparison(
        mpc,
        cfg,
        args.text,
        uq_result,
        q,
        q_history,
        setup.cost_context,
        artifact_base_dir,
        cfg.preference_window,
        rollout_steps,
        body_pos=setup.body_pos,
        spine3_pos=setup.spine3_pos,
        spine3_aa=setup.spine3_aa,
        initial_q=setup.arm_aa,
        install=False,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
