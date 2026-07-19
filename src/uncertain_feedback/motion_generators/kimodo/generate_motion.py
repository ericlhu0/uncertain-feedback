"""Generate and render a kimodo motion without running MPC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uncertain_feedback.consts import KIMODO_MODEL, KIMODO_START_POSE_PATH
from uncertain_feedback.motion_generators.kimodo.kimodo_api import KimodoMotionGenerator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a kimodo motion video")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-npz", type=Path, default=Path("kimodo_motion.npz"))
    parser.add_argument("--output-video", type=Path, default=Path("kimodo_motion.mp4"))
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model-name", default=KIMODO_MODEL)
    parser.add_argument("--start-pose", type=Path, default=KIMODO_START_POSE_PATH)
    parser.add_argument("--frozen-body", action="store_true")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only save the generated SMPL joint positions NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments, generate a kimodo motion, and render it."""
    args = _parse_args()

    gen = KimodoMotionGenerator(model_path=args.model_name)
    start_pose = gen.load_pose(args.start_pose)
    positions = gen.generate_left_arm_position_samples(
        args.text,
        start_pose=start_pose,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        frozen_body=args.frozen_body,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, positions=positions)
    print(f"[kimodo-motion] saved {args.output_npz}")

    if not args.no_video:
        from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
            ArmVisualizer,
        )

        ArmVisualizer().render_body_trajectory_video(
            positions[0],
            args.output_video,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
