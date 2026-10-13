"""Sample care-task start poses per persona or generated user that provoke feedback."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.benchmarks.task_starts import generate_task_starts
from uncertain_feedback.simulated_users import PERSONAS


def main() -> None:
    """Generate task starts from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpc-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--personas",
        nargs="+",
        default=[name for name, user in PERSONAS.items() if user.bounds],
    )
    parser.add_argument("--per-persona", type=int, default=5)
    parser.add_argument(
        "--sampled-bounds",
        type=int,
        default=0,
        help="Number of generated users, alternating constant/coupled bounds, "
        "one start each; replaces --personas",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=50)
    args = parser.parse_args()
    generate_task_starts(
        args.mpc_config,
        args.out_dir,
        args.personas,
        args.per_persona,
        args.seed,
        args.max_attempts,
        sampled_bounds=args.sampled_bounds,
    )


if __name__ == "__main__":
    main()
