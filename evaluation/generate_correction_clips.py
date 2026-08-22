"""Generate hand-labelable correction clips around one base MPC trajectory.

    uv run python evaluation/generate_correction_clips.py \
        --config evaluation/conf/mpc_demo_low1.yaml \
        --out_dir outputs/correction_clips \
        --n_runs 32

Writes ``manifest.json`` with an empty ``caption`` per run; fill those in while
watching ``run_*/clip.mp4``, then feed the directory to
``src/uncertain_feedback/data_collection/build_correction_dataset.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.correction_clips import CorrectionClipConfig, generate_correction_clips


def main() -> None:
    """Parse arguments and generate the clip set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="evaluation/conf/mpc_demo_low1.yaml",
        help=(
            "Planner YAML defining the start pose, goal and MPC settings. The "
            "The low1 default starts the arm resting low: a 165-frame reach at "
            "the default --max_angle_delta 0.00125."
        ),
    )
    parser.add_argument("--out_dir", required=True, help="Directory for the clip set.")
    parser.add_argument("--n_runs", type=int, default=32)
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
            "the 165-frame reach at 0.00125, (6, 50) the 85-frame reach at 0.0025."
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
        )
    )


if __name__ == "__main__":
    main()
