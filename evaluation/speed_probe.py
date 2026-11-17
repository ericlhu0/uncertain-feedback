"""Sample MDM for a few prompts from one person; report wrist pace, render videos.

Each sample is measured as the planner would execute it: re-anchored onto the
person's current state (``anchor_q_trajectory``), which drops the pinned frame
and with it the frame-0 seam jump. Pace is the spine3-relative wrist's peak
speed over 5-frame steps (m/frame) and the frames it takes to cover half its
largest displacement; a fixed-length window's *mean* speed cannot tell a fast
move that stops early from a slow one. Every prompt is sampled from the same
seed, so prompts compare sample-for-sample. ``--history`` conditions on the last frames of a
``(T, 7)`` arm-state array the person has just executed, as the planner does;
without it the prefix is the configured pose held still, as in the demo runner.
With ``--video_dir`` sample ``--video_sample`` of each prompt is rendered at
MDM's 20 fps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    anchor_q_trajectory,
)
from uncertain_feedback.utils.plot import ArmVisualizer


def wrist_pace(wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample peak speed and frames to half displacement of ``(N, T, 3)`` wrists."""
    peak = (np.linalg.norm(wrist[:, 5:] - wrist[:, :-5], axis=-1) / 5).max(axis=1)
    disp = np.linalg.norm(wrist - wrist[:, :1], axis=-1)
    half = (disp >= 0.5 * disp.max(axis=1, keepdims=True)).argmax(axis=1)
    return peak, half


def main() -> None:
    """Parse arguments, sample every prompt, print pace and render videos."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument(
        "--mpc_config", type=Path, default=Path("evaluation/conf/mpc_demo_low1.yaml")
    )
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--video_sample", type=int, default=0)
    args = parser.parse_args()

    cfg = load_mpc_config(args.mpc_config)
    human = Human(pose=cfg.pose, arm=cfg.arm)
    if args.history is not None:
        human = human.step(np.load(args.history))
    gen = MdmMotionGenerator(model_path=args.model_path, seed=args.seed, lock_seed=True)
    # Resolved up front: loading MDM changes the working directory.
    video_dir = None if args.video_dir is None else args.video_dir.resolve()
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(args.prompts):
        positions = gen.generate_positions(
            prompt,
            human,
            prefix=args.history is not None,
            num_samples=args.num_samples,
            num_frames=args.frames,
        )
        q = human.ik_q_from_positions(positions[:, :, LEFT_ARM_CHAIN_INDICES])
        executed = np.stack([anchor_q_trajectory(sample, human.q) for sample in q])
        peak, half = wrist_pace(human.wrist_from_q(executed))
        print(
            f"PACE {prompt!r}: peak {peak.mean():.4f} +/- "
            f"{peak.std() / np.sqrt(len(peak)):.4f} m/frame, "
            f"half-displacement at {half.mean():.1f} frames",
            flush=True,
        )
        if video_dir is not None:
            ArmVisualizer(human.fk).render_rollout_video(
                human.arm_aa_from_q(executed[args.video_sample]),
                video_dir / f"prompt_{i:02d}.mp4",
                spine3_pos=human.spine3_pos,
                spine3_aa=human.spine3_aa,
                body_pos=human.posture,
                fps=20,
            )


if __name__ == "__main__":
    main()
