"""Evaluate hidden limits in simulation: none, soft hidden limit, hand-written constraint.

    uv run python evaluation/run_speed_eval.py --out-dir outputs/speed_eval \\
        [--correction <round dir>/trajectories.npz]

Per persona, care task and sampled start, the arm follows the correction
(``correction_q`` in the ``--correction`` npz, re-anchored on the start; omit
it to skip playback) and then reaches for the task's goal under each arm of
``evaluation/benchmarks/speed.py``. The default personas limit speed; the path
personas in ``CONSTRAINTS`` run the same comparison for a path limit. Limits
act through the planner's cost only; playback is exempt. No LLM and no motion
generator runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.benchmarks.speed import CONSTRAINTS, SUMMARY_COLUMNS, run_speed_eval


def main() -> None:
    """Run the speed evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--personas",
        nargs="+",
        default=["moved_too_fast", "spastic_elbow_flexors"],
        choices=list(CONSTRAINTS),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["elbow_extension", "bathing", "transfer"],
        help="care-task config names under evaluation/conf/care_tasks/",
    )
    parser.add_argument(
        "--starts", type=int, default=3, help="kept per persona and task"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--correction", type=Path, default=None)
    parser.add_argument(
        "--weight", type=float, default=10.0, help="weight of the hidden_soft arm"
    )
    parser.add_argument(
        "--limit-weight",
        type=float,
        default=1000.0,
        help="weight of the constraint arm's hard penalty (llm_cost.limit_weight)",
    )
    parser.add_argument(
        "--pullback-weight",
        type=float,
        default=10.0,
        help="the constraint arm's cost for staying past a path limit "
        "(llm_cost.pullback_weight)",
    )
    args = parser.parse_args()
    correction = (
        None if args.correction is None else np.load(args.correction)["correction_q"]
    )
    table = run_speed_eval(
        args.out_dir.resolve(),
        args.personas,
        args.tasks,
        args.starts,
        args.seed,
        args.max_attempts,
        correction,
        args.weight,
        args.limit_weight,
        args.pullback_weight,
    )
    print(
        table.groupby(["persona", "arm"], sort=False)[SUMMARY_COLUMNS]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
