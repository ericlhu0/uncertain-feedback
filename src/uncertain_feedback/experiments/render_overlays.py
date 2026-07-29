"""Render the all-paths cost-prompt overlay for each UQ cluster — no LLM, no rollouts.

Runs the planner up to ``text_time`` to obtain the MDM/UQ clustering, then writes one
overlay per cluster (each highlighting that cluster among the faded others) using the same
:func:`render_prompt_images` the LLM-cost generator attaches to its prompt. Useful for
eyeballing what the model actually sees.

Run as::

    uv run python src/uncertain_feedback/experiments/render_overlays.py \
        --mpc-config <yaml> [--out-dir overlays]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    build_generated_cost_context,
    render_prompt_images,
)
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.planners.run import (
    _rollout_reference_trajectory,
    build_parser,
    build_run,
    run_planning_loop,
)


def main() -> None:
    """Parse CLI arguments and write one cost-prompt overlay per UQ cluster."""
    parser = build_parser()
    parser.add_argument(
        "--out-dir",
        default="overlays",
        dest="out_dir",
        help="Directory for the per-cluster overlay PNGs.",
    )
    args = parser.parse_args()
    cfg = load_mpc_config(args.mpc_config)
    feedback_cfg = cfg.feedback
    if feedback_cfg is None or feedback_cfg.uq is None:
        raise ValueError("Overlay rendering requires a feedback: section with uq:.")

    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Config must provide an MDM pose to generate from.")
    mpc = setup.mpc

    effective_text_time = (
        args.text_time if args.text_time is not None else feedback_cfg.text_time
    )
    q = setup.q0.copy()
    q_history: list = []
    if effective_text_time > 0:
        result = run_planning_loop(mpc, q, effective_text_time)
        q_history = result.q_history
        if q_history:
            q = q_history[-1]

    mdm_frames = args.mdm_frames if args.mdm_frames is not None else feedback_cfg.frames
    current_pose = setup.gen.build_pose_from_arm_aa(
        setup.initial_pose,
        q_to_arm_aa(q, setup.fk.elbow_hinge_axis),
    )
    mpc.query_mdm_with_uncertainty(
        setup.gen,
        args.text,
        start_pose=current_pose,
        current_q=q,
        auto_cluster=feedback_cfg.uq.auto_cluster,
        mdm_frames=mdm_frames,
        frozen_body=args.frozen_body,
    )
    uqr = mpc.last_uq_result
    if uqr is None:
        raise RuntimeError("UQ clustering produced no result.")

    # Match what the LLM sees: the same always-on original-goal reference + goal marker.
    reference_q = _rollout_reference_trajectory(
        cfg,
        q,
        setup.cost_context,
        mpc._extra_costs,  # pylint: disable=protected-access
        setup.body_pos,
        setup.spine3_pos,
        setup.spine3_aa,
    )
    goal_pos = (
        np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
        if reference_q is not None and cfg.cartesian is not None
        else None
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(uqr.cluster_means):
        context = build_generated_cost_context(
            setup.cost_context,
            q,
            uqr.cluster_means[label],
            q_history,
            window=cfg.preference_window,
            body_pos=setup.body_pos,
            reference_traj=reference_q,
        )
        images = render_prompt_images(
            context,
            out_dir / f"cluster_{label}",
            candidate_trajs=uqr.cluster_means,
            highlight_label=label,
            reference_traj=reference_q,
            goal_pos=goal_pos,
        )
        for key, path in images.items():
            print(f"[overlay] cluster {label} {key} -> {path}")
    print(
        f"[overlay] wrote overlays for {len(uqr.cluster_means)} clusters to {out_dir}"
    )


if __name__ == "__main__":
    main()
