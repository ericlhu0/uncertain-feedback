"""Snapshot the live real-world environment to a file for offline development.

Records both channels ``env: real`` senses through — the OptiTrack stream and
the Gen3's measured joint state — so later runs can be driven from the file
instead of the lab (see
:mod:`uncertain_feedback.envs.real_recording`)::

    uv run python src/uncertain_feedback/envs/record_real.py \
        --host 192.168.2.243 --controller-host 127.0.0.1 \
        --require-bodies 1 2 3 4 5 6 \
        --seconds 30 --out real_recordings/lab.npz

Reading the arm's state is passive: nothing is commanded, no mode is switched,
and the gripper is not touched. Point ``env_params.recording`` at the output to
replay it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uncertain_feedback.envs.real_recording import RealRecording


def main() -> None:
    """Capture one recording and print what landed in it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="OptiTrack PC address")
    parser.add_argument(
        "--controller-host",
        default=None,
        help="emprise-gen3-controller ZMQ host; omit to record mocap only",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=60.0, help="samples per second")
    parser.add_argument(
        "--require-bodies",
        type=int,
        nargs="*",
        default=None,
        help="wait until these streaming ids are all tracked before recording",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    recording = RealRecording.capture(
        args.host,
        args.seconds,
        controller_host=args.controller_host,
        rate=args.rate,
        require_bodies=args.require_bodies,
    )
    recording.save(args.out)
    tracked = recording.valid.all(axis=0)
    print(f"wrote {args.out} ({recording.n_frames} frames)")
    print(f"rigid bodies: {recording.body_ids.tolist()}")
    print(f"tracked in every frame: {recording.body_ids[tracked].tolist()}")
    if recording.robot_q.size:
        span = np.ptp(recording.robot_q, axis=0)
        print(f"robot joints at frame 0: {np.round(recording.robot_q[0], 4)}")
        print(f"robot joint range over the window: {np.round(span, 4)}")


if __name__ == "__main__":
    main()
