"""Headless simulated-user transfer experiment.

Run as::

    uv run python src/uncertain_feedback/experiments/run_transfer_experiment.py \
        --mpc-config <yaml> [--persona adhesive_capsulitis] [--save-video]

The persona defaults to the config's ``user:`` key; ``--persona`` overrides it.
The persona's hidden comfort cost decides when feedback is given (first
violation of the initial plan), what is said (the persona's feedback line — the
``--text`` flag is ignored), and which UQ cluster is picked. Conditions
(base / tracking / generated / oracle) are rolled out to the original goal and
every ``transfer.goals`` entry, scored against the hidden cost, and written to
``transfer_artifacts/<timestamp>/transfer_summary.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from uncertain_feedback.experiments.transfer_experiment import run_transfer_experiment
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.run import build_parser, build_run
from uncertain_feedback.simulated_users import PERSONAS, get_persona


def _transfer_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = "Run a simulated-user transfer experiment"
    p.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=sorted(PERSONAS),
        help=(
            "Simulated-user persona providing the hidden cost and feedback. "
            "Defaults to the config's `user:` persona."
        ),
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        dest="save_video",
        help="Render an MP4 per condition rollout.",
    )
    return p


def main() -> None:
    args = _transfer_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)
    user = get_persona(args.persona if args.persona is not None else cfg.user)
    if not user.bounds:
        raise ValueError(
            f"Transfer experiments need a user with hidden bounds; {user.name!r} "
            "has none. Set `user:` in the config or pass --persona."
        )

    if cfg.planner != "arm_mpc_cartesian":
        raise ValueError(
            f"Transfer experiments require planner: arm_mpc_cartesian; got {cfg.planner!r}."
        )
    if not cfg.llm_cost.enabled:
        raise ValueError("Transfer experiments require llm_cost.enabled: true.")
    if not cfg.cartesian.goals:
        raise ValueError("Transfer experiments require cartesian.goals.")
    if not cfg.transfer.goals:
        print("[transfer] warning: no transfer.goals configured — only the original goal is evaluated.")

    setup = build_run(args, cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Transfer experiment config must provide an MDM pose.")
    mpc = cast(LeftArmMPCMDMUQ, setup.mpc)
    mdm_frames = args.mdm_frames if args.mdm_frames is not None else cfg.mdm_frames

    run_transfer_experiment(
        mpc,
        cfg,
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
