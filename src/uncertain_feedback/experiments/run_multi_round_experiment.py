"""CLI for multi-goal feedback and pose-dependent cost combination."""

from __future__ import annotations

import argparse
from pathlib import Path
from uncertain_feedback.experiments.experiment_pipeline import (
    apply_persona_goals,
    require_correction_planner,
)
from uncertain_feedback.experiments.multi_round_experiment import (
    run_multi_round_experiment,
)
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.run import build_parser, build_run
from uncertain_feedback.simulated_users import PERSONAS, get_persona


def _parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Run a multi-round pose-dependent cost experiment"
    parser.add_argument(
        "--persona",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(PERSONAS),
        help="Personas to run; defaults to the config's user.",
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="Run every persona with hidden bounds.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Render an MP4 for each final condition rollout.",
    )
    return parser


def main() -> None:
    """Run the requested personas while reusing one loaded motion generator."""
    args = _parser().parse_args()
    cfg = load_mpc_config(args.mpc_config)
    require_correction_planner(cfg, "Multi-round experiments")
    if not cfg.llm_cost.enabled:
        raise ValueError("Multi-round experiments require llm_cost.enabled: true.")
    if args.all_personas:
        persona_names = [name for name, user in PERSONAS.items() if user.bounds]
    elif args.persona is not None:
        persona_names = args.persona
    else:
        persona_names = [cfg.user]

    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Multi-round experiment config must provide an MDM pose.")
    mpc = setup.mpc
    mdm_frames = (
        args.mdm_frames if args.mdm_frames is not None else cfg.feedback.frames
    )
    for name in persona_names:
        user = get_persona(name)
        if not user.bounds:
            raise ValueError(f"Persona {name!r} has no hidden feedback bounds.")
        persona_cfg = apply_persona_goals(cfg, name)
        if len(persona_cfg.cartesian.goals) < 2:
            print(
                f"[multi-round] warning: {name} has fewer than two goals; "
                "no combination will be possible",
                flush=True,
            )
        run_multi_round_experiment(
            mpc,
            persona_cfg,
            user,
            setup.gen,
            setup.initial_pose,
            setup.q0,
            setup.cost_context,
            Path.cwd().resolve(),
            body_pos=setup.body_pos,
            spine3_pos=setup.spine3_pos,
            spine3_aa=setup.spine3_aa,
            mdm_frames=mdm_frames,
            frozen_body=args.frozen_body,
            save_video=args.save_video,
        )


if __name__ == "__main__":
    main()
