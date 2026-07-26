"""Verify the OptiTrack stream and the derived arm configuration — no robot.

Prints per-rigid-body tracking validity, the effective frame rate, the derived
``(7,)`` planner configuration, and the spine3-relative wrist position, so the
registration can be checked before anything moves::

    uv run python src/uncertain_feedback/mocap/monitor.py --host 192.168.2.243 \
        --base-id 1 --collar-id 2 --shoulder-id 3 --elbow-id 4 --wrist-id 5
    ... --video mocap_check.mp4   # also render the measured arm

The torso's *shape* comes from the SMPL zero configuration rather than a run
config (its position, as in a run, is the measured collar). The registration yaw
is solved against that pose's clavicle, so both it and the printed ``q`` are
indicative only — the run's own yaw comes out of its start pose. What this checks
is that all bodies stay tracked, that the rate matches Motive, and that the
rendered arm follows the real one.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from uncertain_feedback.mocap.natnet import NatNetReceiver
from uncertain_feedback.mocap.registration import ArmRegistration, arm_keypoints
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    q_to_arm_aa,
)

_WAIT_TIMEOUT_S = 20.0
_PRINT_PERIOD_S = 0.5


def main() -> None:
    """Stream mocap, print the derived configuration, optionally render it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="OptiTrack PC address")
    parser.add_argument("--base-id", type=int, required=True)
    parser.add_argument("--collar-id", type=int, required=True)
    parser.add_argument("--shoulder-id", type=int, required=True)
    parser.add_argument("--elbow-id", type=int, required=True)
    parser.add_argument("--wrist-id", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--video", default=None, help="write a rollout video here")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    fk = SmplLeftArmFK()
    receiver = NatNetReceiver.connect(args.host)
    print(f"connected to {args.host}, NatNet {receiver.natnet_version}")
    body_ids = [
        args.base_id,
        args.collar_id,
        args.shoulder_id,
        args.elbow_id,
        args.wrist_id,
    ]
    bodies = receiver.wait_for(body_ids, _WAIT_TIMEOUT_S)
    base = bodies[args.base_id]
    keypoints = arm_keypoints(
        bodies, args.collar_id, args.shoulder_id, args.elbow_id, args.wrist_id
    )
    assert keypoints is not None
    registration = ArmRegistration.calibrate(
        fk=fk,
        q0=np.zeros(Q_DIM),
        spine3_pos=None,
        spine3_aa=None,
        base_position=base.position,
        base_orientation=base.orientation,
        collar_mocap=keypoints[0],
        shoulder_mocap=keypoints[1],
    )
    print(
        f"solved registration yaw: {registration.robot_base_yaw:+.4f} rad "
        f"({np.rad2deg(registration.robot_base_yaw):+.2f} deg) "
        "— against the zero pose's clavicle, not the run config's"
    )
    print(f"robot base in pybullet frame: {np.round(registration.base_pb, 4)}")

    measured: list[np.ndarray] = []
    processed = 0
    frames_at_start = receiver.frame_count
    last_stamp = 0.0
    last_print = 0.0
    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        stamp, bodies = receiver.latest()
        if stamp == last_stamp:
            time.sleep(0.001)
            continue
        last_stamp = stamp
        processed += 1
        keypoints = arm_keypoints(
            bodies, args.collar_id, args.shoulder_id, args.elbow_id, args.wrist_id
        )
        if keypoints is None:
            untracked = [
                body_id
                for body_id in body_ids
                if body_id not in bodies or not bodies[body_id].valid
            ]
            print(f"untracked rigid bodies: {untracked}")
            continue
        q = registration.q_from_keypoints(*keypoints)
        measured.append(q)
        if time.monotonic() - last_print > _PRINT_PERIOD_S:
            last_print = time.monotonic()
            positions = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis))
            wrist = positions[4] - positions[0]
            rate = (receiver.frame_count - frames_at_start) / (time.monotonic() - start)
            # For a left arm the wrist should sit at positive x (the person's
            # left) relative to spine3; a negative x means the registration
            # rotation is wrong (check mocap_base_correction_quat).
            print(
                f"{rate:6.1f} fps  q={np.round(q, 3)}  "
                f"wrist rel spine3={np.round(wrist, 3)}"
            )
    receiver.close()

    streamed = receiver.frame_count - frames_at_start
    print(
        f"{streamed} frames streamed, {processed} processed, "
        f"{len(measured)} with all bodies tracked"
    )
    if args.video is not None and measured:
        from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
            ArmVisualizer,
        )

        rollout = q_to_arm_aa(np.stack(measured), fk.elbow_hinge_axis)
        ArmVisualizer(fk).render_rollout_video(rollout, args.video, fps=args.fps)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
