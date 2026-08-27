"""Generate hand-labelable correction clips over randomly sampled reaches.

    uv run python \
        src/uncertain_feedback/data_collection/dataset_auto_correction/generate.py \
        [--config evaluation/conf/mpc_demo_low1.yaml] [--out_dir <dir>]

Each run samples its own start arm configuration and Cartesian goal, so the
config's ``cartesian.goals`` is unused and only the clavicle of its ``arm:``
survives (the planner never actuates that slot); everything else in it — body
pose, MPC settings, ``cartesian.threshold``, ``corrections.trigger_threshold``
and ``llm_cost.model`` — still applies.

Writes into ``data/dataset_auto_correction/clips`` unless ``--out_dir`` says
otherwise — the same default ``label.py`` captions, so neither needs a flag.

Writes ``manifest.json`` with an empty ``caption`` per run; fill those in with
``label.py``, then feed the directory to ``build_dataset.py``. With the default
``--n_runs 0`` only the base artifacts are written and the labeling UI plans runs
on demand; pass ``--n_runs N`` to sample a batch up front.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uncertain_feedback.data_collection.common.paths import DEFAULT_CLIP_SET
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    CorrectionClipConfig,
    generate_correction_clips,
)


def main() -> None:
    """Parse arguments and generate the clip set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="evaluation/conf/mpc_demo_low1.yaml",
        help=(
            "Planner YAML defining the body pose, MPC settings, "
            "cartesian.threshold, corrections.trigger_threshold and "
            "llm_cost.model (the labeling UI's Draft caption button). Its "
            "cartesian.goals is NOT used and only the clavicle of its arm: is — "
            "each run samples its own start arm and goal."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_CLIP_SET),
        help=f"Directory for the clip set (default: {DEFAULT_CLIP_SET}).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=0,
        help=(
            "Runs to sample up front (default: 0 — write only the base "
            "artifacts and let the labeling UI plan runs on demand)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trigger_window",
        type=int,
        nargs=2,
        default=(12, 100),
        metavar=("LOW", "HIGH"),
        help=(
            "Accepted range for the naive rollout's first-violation step. Counts "
            "naive frames, so it scales with --max_angle_delta: (12, 100) suits "
            "the ~165-frame reaches at 0.00125, (6, 50) the ~85-frame ones at "
            "0.0025. Scenarios whose reach is shorter than LOW are redrawn."
        ),
    )
    parser.add_argument(
        "--max_angle_delta",
        type=float,
        default=0.00125,
        help=(
            "Action-sampling spread overriding the config's, and the one knob "
            "for how big and fast a clip's motion is: a clip is a fixed frame "
            "budget, so halving this halves both speed and distance covered "
            "(0.0025 -> 0.351 m of wrist path per clip, 0.00125 -> 0.186 m). "
            "Scale --trigger_window with it."
        ),
    )
    parser.add_argument(
        "--margin_range",
        type=float,
        nargs=2,
        default=(0.05, 0.20),
        metavar=("LOW", "HIGH"),
        help="Radians the sampled bound sits past the naive feature value.",
    )
    parser.add_argument(
        "--min_goal_distance",
        type=float,
        default=0.25,
        help=(
            "Metres a sampled goal must sit from the sampled start wrist for the "
            "scenario to be kept. Scales with --max_angle_delta like "
            "--trigger_window does: 0.25 m is ~60 naive frames at 0.00125."
        ),
    )
    parser.add_argument(
        "--correction_frames",
        type=int,
        nargs=2,
        default=(42, 56),
        metavar=("LOW", "HIGH"),
        help="Frames of corrected continuation kept per clip, before the prefix.",
    )
    args = parser.parse_args()

    # The MDM loader chdir()s into its submodule and never restores, so every
    # path must be absolute before the rig is built.
    generate_correction_clips(
        CorrectionClipConfig(
            config_path=Path(args.config).resolve(),
            out_dir=Path(args.out_dir).resolve(),
            n_runs=args.n_runs,
            seed=args.seed,
            trigger_window=tuple(args.trigger_window),
            margin_range=tuple(args.margin_range),
            correction_frames=tuple(args.correction_frames),
            max_angle_delta=args.max_angle_delta,
            min_goal_distance=args.min_goal_distance,
        )
    )


if __name__ == "__main__":
    main()
