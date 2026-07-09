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
``--text`` flag is ignored), and which UQ cluster is picked. Conditions
(base / tracking / generated / oracle) are rolled out to the original goal and
every ``transfer.goals`` entry, scored against the hidden cost, and written to
``transfer_artifacts/<timestamp>/transfer_summary.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import cast

from uncertain_feedback.experiments.transfer_experiment import run_transfer_experiment
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
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


def _apply_persona_goals(cfg: MpcRunConfig, user_name: str) -> MpcRunConfig:
    """Swap in the persona's goal geometry when the config defines an override."""
    persona_goals = cfg.persona_goals.get(user_name)
    if persona_goals is None:
        return cfg
    return replace(
        cfg,
        cartesian=replace(cfg.cartesian, goals=persona_goals.cartesian),
        transfer=replace(cfg.transfer, goals=persona_goals.transfer),
    )


def main() -> None:
    args = _transfer_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    if cfg.planner != "arm_mpc_cartesian":
        raise ValueError(
            f"Transfer experiments require planner: arm_mpc_cartesian; got {cfg.planner!r}."
        )
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
    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Transfer experiment config must provide an MDM pose.")
    mpc = cast(LeftArmMPCMDMUQ, setup.mpc)
    mdm_frames = args.mdm_frames if args.mdm_frames is not None else cfg.mdm_frames

    for name in persona_names:
        user = get_persona(name)
        if not user.bounds:
            raise ValueError(
                f"Transfer experiments need a user with hidden bounds; {user.name!r} "
                "has none. Set `user:` in the config or pass --persona."
            )
        persona_cfg = _apply_persona_goals(cfg, user.name)
        if not persona_cfg.transfer.goals:
            print("[transfer] warning: no transfer.goals configured — only the original goal is evaluated.")
        if len(persona_names) > 1:
            print(f"[transfer] ===== persona: {user.name} =====")
        run_transfer_experiment(
            mpc,
            persona_cfg,
            user,
            setup.gen,
            setup.initial_pose,
            setup.arm_aa,
            setup.cost_context,
            artifact_base_dir,
            body_pos=setup.body_pos,
            spine3_pos=setup.spine3_pos,
            spine3_aa=setup.spine3_aa,
            mdm_frames=mdm_frames,
            frozen_body=args.frozen_body,
            save_video=args.save_video,
        )


if __name__ == "__main__":
    main()
