"""Headless single persona/backend experiment.

Run as::

    uv run python src/uncertain_feedback/experiments/run_experiment.py \
        --mpc-config <yaml> [--persona P] [--backend llm|turns|agent] [--save-video]

The persona defaults to the config's ``user:`` key, and the backend defaults to
``llm_cost.backend``. The experiment runs the simulated user's original-goal
feedback loop only: initial rollout, hidden-cost trigger, MDM/UQ candidates,
oracle cluster selection, one generated cost, and original-goal evaluation.
Held-out transfer goals are handled by ``run_transfer_experiment.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from uncertain_feedback.experiments.experiment_pipeline import (
    apply_persona_goals,
    run_experiment,
)
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import COST_BACKENDS, load_mpc_config
from uncertain_feedback.planners.run import build_parser, build_run
from uncertain_feedback.simulated_users import PERSONAS, get_persona


def _experiment_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = "Run one simulated-user persona/backend experiment"
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
        "--backend",
        type=str,
        default=None,
        choices=sorted(COST_BACKENDS),
        help="Cost-generation backend. Defaults to llm_cost.backend in the config.",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        dest="save_video",
        help="Render MP4s for original-goal condition rollouts.",
    )
    return p


def main() -> None:
    """Parse CLI arguments and run one persona/backend experiment."""
    args = _experiment_parser().parse_args()
    artifact_base_dir = Path.cwd().resolve()
    cfg = load_mpc_config(args.mpc_config)

    if cfg.planner != "arm_mpc_cartesian":
        raise ValueError(
            "Experiments require planner: arm_mpc_cartesian; " f"got {cfg.planner!r}."
        )
    if not cfg.llm_cost.enabled:
        raise ValueError("Experiments require llm_cost.enabled: true.")

    persona_name = args.persona if args.persona is not None else cfg.user
    user = get_persona(persona_name)
    if not user.bounds:
        raise ValueError(
            f"Experiments need a user with hidden bounds; {user.name!r} has none."
        )
    persona_cfg = apply_persona_goals(cfg, user.name)
    backend = args.backend if args.backend is not None else persona_cfg.llm_cost.backend

    setup = build_run(args, persona_cfg)
    if setup.gen is None or setup.initial_pose is None:
        raise ValueError("Experiment config must provide an MDM pose.")
    mpc = cast(LeftArmMPCMDMUQ, setup.mpc)
    mdm_frames = (
        args.mdm_frames if args.mdm_frames is not None else persona_cfg.mdm_frames
    )

    run_experiment(
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
        backend=backend,
        mdm_frames=mdm_frames,
        frozen_body=args.frozen_body,
        save_video=args.save_video,
        artifact_dir=Path("experiment_artifacts"),
        log_prefix="[experiment]",
    )


if __name__ == "__main__":
    main()
