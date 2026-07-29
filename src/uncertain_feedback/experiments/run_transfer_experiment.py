"""Headless simulated-user transfer experiment.

Run as::

    uv run python src/uncertain_feedback/experiments/run_transfer_experiment.py \
        --mpc-config <yaml> [--persona P [P ...] | --all-personas] [--save-video]

The persona defaults to the config's ``user:`` key; ``--persona`` takes one or
more persona names, and ``--all-personas`` runs every persona with hidden
bounds. Each persona gets its own timestamped artifact dir, reusing a single
loaded MDM setup.
The persona's hidden comfort cost decides when feedback is given (first
violation of the initial plan), what is said (the persona's feedback line — the
``--text`` flag is ignored), and which UQ cluster is picked by oracle-scoring
the scaled raw cluster means. Conditions (base / tracking / generated / oracle)
are rolled out to the original goal and every ``transfer.goals`` entry, scored
against the hidden cost, and written to
``transfer_artifacts/<timestamp>/transfer_summary.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uncertain_feedback.experiments.experiment_pipeline import (
    apply_persona_goals,
    require_correction_planner,
)
from uncertain_feedback.experiments.transfer_experiment import run_transfer_experiment
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.run import build_parser, build_run
from uncertain_feedback.simulated_users import PERSONAS, get_persona


def _transfer_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = "Run a simulated-user transfer experiment"
    p.add_argument(
        "--persona",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(PERSONAS),
        help=(
            "One or more simulated-user personas to run, each providing the "
            "hidden cost and feedback. Defaults to the config's `user:` persona."
        ),
    )
    p.add_argument(
        "--all-personas",
        action="store_true",
        dest="all_personas",
        help="Run every persona with hidden bounds (ignores --persona).",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        dest="save_video",
        help="Render an MP4 per condition rollout.",
    )
    return p


def main() -> None:
    """Parse CLI arguments and run the simulated-user transfer experiment."""
    args = _transfer_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    require_correction_planner(cfg, "Transfer experiments")
    if not cfg.llm_cost.enabled:
        raise ValueError("Transfer experiments require llm_cost.enabled: true.")

    if args.all_personas:
        persona_names = [name for name, user in PERSONAS.items() if user.bounds]
    elif args.persona is not None:
        persona_names = args.persona
    else:
        persona_names = [cfg.user]

    # The initial pose and MDM generator are persona-independent — only the goals
    # and hidden cost change — so build the run once and reuse it per persona.
    print(
        f"[transfer] building shared setup for {len(persona_names)} persona(s): "
        f"{', '.join(persona_names)}",
        flush=True,
    )
    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Transfer experiment config must provide an MDM pose.")
    mpc = setup.mpc
    mdm_frames = (
        args.mdm_frames if args.mdm_frames is not None else cfg.feedback.frames
    )
    print("[transfer] shared setup ready", flush=True)

    for idx, name in enumerate(persona_names, start=1):
        user = get_persona(name)
        if not user.bounds:
            raise ValueError(
                f"Transfer experiments need a user with hidden bounds; {user.name!r} "
                "has none. Set `user:` in the config or pass --persona."
            )
        persona_cfg = apply_persona_goals(cfg, user.name)
        if not persona_cfg.transfer.goals:
            print(
                "[transfer] warning: no transfer.goals configured — only the original goal is evaluated."
            )
        print(
            f"[transfer] ===== persona {idx}/{len(persona_names)}: {user.name} "
            f"(transfer_goals={len(persona_cfg.transfer.goals)}, "
            f"save_video={args.save_video}) =====",
            flush=True,
        )
        summary = run_transfer_experiment(
            mpc,
            persona_cfg,
            user,
            setup.gen,
            setup.initial_pose,
            setup.q0,
            setup.cost_context,
            artifact_base_dir,
            body_pos=setup.body_pos,
            spine3_pos=setup.spine3_pos,
            spine3_aa=setup.spine3_aa,
            mdm_frames=mdm_frames,
            frozen_body=args.frozen_body,
            save_video=args.save_video,
        )
        print(
            f"[transfer] ===== finished persona {idx}/{len(persona_names)}: "
            f"{user.name} result={summary.get('result')} =====",
            flush=True,
        )


if __name__ == "__main__":
    main()
