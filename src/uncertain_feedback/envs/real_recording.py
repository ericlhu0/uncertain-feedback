"""Record what the real environment senses, and replay it without hardware.

:class:`~uncertain_feedback.envs.real.RealEnv` senses the world through exactly
two channels: OptiTrack rigid bodies over NatNet, and the Gen3's measured joint
state over ZMQ. A :class:`RealRecording` holds both, sampled together over a
window, so a later run can be driven from the file rather than from the lab —
the registration, the measured segment lengths, the measured robot base, the
measured grasp, and the IK gate all see the numbers that were really there.

:class:`ReplayReceiver` and :class:`ReplayMirror` are drop-in stand-ins for
:class:`~uncertain_feedback.mocap.natnet.NatNetReceiver` and
:class:`~uncertain_feedback.envs.real_mirror.RealArmMirror`, so ``RealEnv``
itself is unchanged apart from choosing which pair to build.

What replay is *not*: the person in the file does not react to the robot. The
mocap side is open-loop playback, and the arm is an ideal tracker seeded at the
recorded joint configuration — it goes exactly where it is commanded. So a
replay run exercises the whole pipeline on real geometry, but it is not a
substitute for the closed loop on a real person.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.consts import PROJECT_ROOT
from uncertain_feedback.mocap.natnet import (
    MocapStaleError,
    NatNetReceiver,
    RigidBodyPose,
)

REPO_ROOT = PROJECT_ROOT.parent.parent

# Robot state fields worth keeping. Only `joint_positions` is replayed; the rest
# are recorded because the point of a snapshot is not having to go back to the
# lab for a signal nobody needed yet.
_ROBOT_FIELDS = (
    "joint_positions",
    "joint_velocities",
    "joint_efforts",
    "full_joint_positions",
    "gripper_position",
    "ee_position",
    "ee_orientation",
)
_ROBOT_PREFIX = "robot__"
_POLL_PERIOD_S = 0.001


@dataclass(frozen=True)
class RealRecording:
    """A window of everything the real env can sense, sampled frame by frame.

    ``positions``/``orientations``/``valid`` are indexed ``(frame, body)`` over
    ``body_ids`` — *every* streamed rigid body, not only the configured ones, so
    a later run can use ids this one did not. ``robot`` holds one array per
    recorded state field, each indexed by frame.
    """

    stamps: np.ndarray  # (T,) seconds since the first sample
    body_ids: np.ndarray  # (B,) NatNet streaming ids
    positions: np.ndarray  # (T, B, 3)
    orientations: np.ndarray  # (T, B, 4) xyzw
    valid: np.ndarray  # (T, B) bool
    robot: dict[str, np.ndarray]  # field -> (T, ...)
    metadata: dict[str, Any]

    @property
    def n_frames(self) -> int:
        """How many samples the window holds."""
        return int(self.stamps.shape[0])

    @property
    def robot_q(self) -> np.ndarray:
        """The measured ``(T, 7)`` arm joint trajectory, empty if unrecorded."""
        return self.robot.get("joint_positions", np.zeros((self.n_frames, 0)))

    def frame(self, index: int) -> dict[int, RigidBodyPose]:
        """One sample in the mapping shape a NatNet frame arrives in."""
        return {
            int(body_id): RigidBodyPose(
                position=self.positions[index, b],
                orientation=self.orientations[index, b],
                valid=bool(self.valid[index, b]),
            )
            for b, body_id in enumerate(self.body_ids)
        }

    def save(self, path: str | Path) -> None:
        """Write the recording to ``path`` as a single compressed .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            stamps=self.stamps,
            body_ids=self.body_ids,
            positions=self.positions,
            orientations=self.orientations,
            valid=self.valid,
            metadata=np.array(json.dumps(self.metadata)),
            **{f"{_ROBOT_PREFIX}{k}": v for k, v in self.robot.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "RealRecording":
        """Read a saved recording; a relative path is taken from the repo root.

        Not from the working directory: loading MDM chdirs the process into the
        submodule and never changes back, so by the time a run builds its env a
        path relative to where the run was launched no longer resolves.
        """
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        data = np.load(path, allow_pickle=False)
        return cls(
            stamps=data["stamps"],
            body_ids=data["body_ids"],
            positions=data["positions"],
            orientations=data["orientations"],
            valid=data["valid"],
            robot={
                key[len(_ROBOT_PREFIX) :]: data[key]
                for key in data.files
                if key.startswith(_ROBOT_PREFIX)
            },
            metadata=json.loads(str(data["metadata"])),
        )

    @classmethod
    def capture(
        cls,
        mocap_host: str,
        seconds: float,
        controller_host: str | None = None,
        rate: float = 60.0,
        wait_timeout: float = 20.0,
        require_bodies: list[int] | None = None,
    ) -> "RealRecording":
        """Sample the live mocap stream and arm state together for ``seconds``.

        A sample is taken when a new mocap frame has arrived *and* ``1 / rate``
        has passed, so the file is paced by the stream rather than by this
        loop, and the arm state is read alongside each kept mocap frame — the
        two channels stay aligned the way a run reads them. With
        ``require_bodies`` the capture blocks until those ids are all tracked
        before it starts, so the window does not open on a dropout.
        """
        receiver = NatNetReceiver.connect(mocap_host)
        controller = None
        if controller_host is not None:
            from emprise_gen3_controller import (  # pylint: disable=import-outside-toplevel
                ArmController,
            )

            controller = ArmController.connect(controller_host)
        if require_bodies:
            receiver.wait_for(require_bodies, wait_timeout)

        body_ids = sorted(receiver.latest()[1])
        stamps: list[float] = []
        positions: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        robot: dict[str, list[np.ndarray]] = {field: [] for field in _ROBOT_FIELDS}
        period = 1.0 / rate
        start = time.monotonic()
        last_stamp = 0.0
        last_kept = -period
        while time.monotonic() - start < seconds:
            stamp, bodies = receiver.latest()
            elapsed = time.monotonic() - start
            if stamp == last_stamp or elapsed - last_kept < period:
                time.sleep(_POLL_PERIOD_S)
                continue
            last_stamp = stamp
            last_kept = elapsed
            missing = RigidBodyPose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), False)
            poses = [bodies.get(body_id, missing) for body_id in body_ids]
            stamps.append(elapsed)
            positions.append(np.stack([pose.position for pose in poses]))
            orientations.append(np.stack([pose.orientation for pose in poses]))
            valid.append(np.array([pose.valid for pose in poses]))
            if controller is not None:
                state = controller.get_state()
                for field in _ROBOT_FIELDS:
                    robot[field].append(
                        np.asarray(getattr(state, field), dtype=np.float64)
                    )
        receiver.close()

        metadata: dict[str, Any] = {
            "mocap_host": mocap_host,
            "controller_host": controller_host,
            "natnet_version": list(receiver.natnet_version),
            "seconds": seconds,
            "rate": rate,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if controller is not None:
            state = controller.get_state()
            metadata["robot_mode"] = str(state.mode)
            metadata["robot_tool"] = str(state.tool)
        return cls(
            stamps=np.asarray(stamps, dtype=np.float64),
            body_ids=np.asarray(body_ids, dtype=np.int64),
            positions=np.stack(positions),
            orientations=np.stack(orientations),
            valid=np.stack(valid),
            robot={
                field: np.stack(values) for field, values in robot.items() if values
            },
            metadata=metadata,
        )


class ReplayReceiver:
    """A :class:`NatNetReceiver` that reads a recording instead of the network.

    Each :meth:`latest` advances one recorded frame and holds at the last one,
    rather than following wall-clock: an MPC step costs far more than a mocap
    frame, so a clocked replay would run the file out during the first few
    solves, and stepping per read also makes a replay run reproducible.
    """

    def __init__(self, recording: RealRecording) -> None:
        self._recording = recording
        self._index = -1
        self.natnet_version = tuple(recording.metadata.get("natnet_version", (4, 0)))

    @property
    def index(self) -> int:
        """Which recorded frame the next read will return."""
        return max(self._index, 0)

    @property
    def frame_count(self) -> int:
        """Frames read so far — the live receiver's stream counter."""
        return self._index + 1

    def latest(self) -> tuple[float, dict[int, RigidBodyPose]]:
        """Advance one recorded frame and return it, stamped now."""
        self._index = min(self._index + 1, self._recording.n_frames - 1)
        return time.monotonic(), self._recording.frame(self._index)

    def wait_for(self, body_ids: list[int], timeout: float) -> dict[int, RigidBodyPose]:
        """Return the next frame, requiring the ids to be tracked in it."""
        del timeout
        _, bodies = self.latest()
        untracked = [i for i in body_ids if i not in bodies or not bodies[i].valid]
        if untracked:
            raise MocapStaleError(
                f"Rigid bodies {untracked} are not tracked in recorded frame "
                f"{self._index}. Recorded ids: {sorted(bodies)}."
            )
        return bodies

    def close(self) -> None:
        """Nothing to release."""


class ReplayMirror:
    """A :class:`RealArmMirror` that tracks perfectly and moves nothing.

    Seeded at the recorded joint configuration, so the first grasp is measured
    against the gripper pose that was really there; afterwards it reports back
    whatever it was last commanded. That makes the robot side of a replay run an
    ideal tracker — the arm never lags, never yields, and never hits a hardware
    fault — which is the part of the loop replay cannot honestly reproduce.
    """

    def __init__(self, q0: np.ndarray) -> None:
        self._q = np.asarray(q0, dtype=np.float64).copy()

    def start_from_grasp(self) -> None:
        """No grasp to take, no mode to switch."""

    def send(self, q: np.ndarray) -> None:
        """Accept a joint target; an ideal tracker is already there."""
        self._q = np.asarray(q, dtype=np.float64).copy()

    def current_q(self) -> np.ndarray:
        """The recorded configuration, or the last one commanded."""
        return self._q.copy()

    def stop(self) -> None:
        """Nothing to halt."""
