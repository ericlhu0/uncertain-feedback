"""Minimal NatNet client: OptiTrack rigid-body poses over multicast.

Motive streams every tracked asset as a NatNet ``FrameOfData`` datagram. Only
the rigid-body section is needed here, so this decoder reads the sections that
precede it purely to advance the offset and stops once the rigid bodies are
out; skeletons, labeled markers, force plates, devices, and timing are never
parsed.

Written in-repo rather than pulled from PyPI: the only published clients
(``natnet`` 0.3.0, ``natnetclient`` 0.8.3) are NatNet-2-era packages, and
NatNet 2 nests marker data inside each rigid body — they would mis-parse a
NatNet 3+ stream silently.

Usage::

    receiver = NatNetReceiver.connect("192.168.2.243")
    stamp, bodies = receiver.latest()
"""

from __future__ import annotations

import atexit
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

NAT_CONNECT = 0
NAT_SERVERINFO = 1
NAT_FRAMEOFDATA = 7

MULTICAST_DEFAULT = "239.255.42.99"
COMMAND_PORT_DEFAULT = 1510
DATA_PORT_DEFAULT = 1511

# Motive's connect request carries a padded options block; the reply is a
# ServerInfo packet (app name, app version, NatNet version).
_PING_PAYLOAD = b"Ping\0" + b"\0" * 265
_HANDSHAKE_TIMEOUT_S = 3.0
_RECV_TIMEOUT_S = 0.5
_WAIT_POLL_S = 0.05
_RECV_BUFFER = 65535
# Tracking-valid is bit 0 of the rigid body's params field.
_TRACKING_VALID = 0x01


class MocapStaleError(RuntimeError):
    """Raised when no valid mocap pose has arrived within the hold timeout."""


def require_fresh(age: float, timeout: float) -> None:
    """Raise :class:`MocapStaleError` past ``timeout`` seconds without a pose."""
    if age > timeout:
        raise MocapStaleError(
            f"No valid mocap pose for {age:.2f}s (hold timeout {timeout:.2f}s)"
        )


@dataclass(frozen=True)
class RigidBodyPose:
    """One rigid body's pose in the mocap world frame."""

    position: np.ndarray
    orientation: np.ndarray  # xyzw
    valid: bool


def decode_frame(data: bytes) -> dict[int, RigidBodyPose]:
    """Decode the rigid-body section of a NatNet ``FrameOfData`` datagram."""
    offset = 8  # message id + payload size + frame number
    (n_marker_sets,) = struct.unpack_from("<i", data, offset)
    offset += 4
    for _ in range(n_marker_sets):
        offset = data.index(b"\0", offset) + 1  # set name
        (n_markers,) = struct.unpack_from("<i", data, offset)
        offset += 4 + 12 * n_markers
    (n_unlabeled,) = struct.unpack_from("<i", data, offset)
    offset += 4 + 12 * n_unlabeled

    (n_bodies,) = struct.unpack_from("<i", data, offset)
    offset += 4
    bodies: dict[int, RigidBodyPose] = {}
    for _ in range(n_bodies):
        body_id = struct.unpack_from("<i", data, offset)[0]
        pose = struct.unpack_from("<7f", data, offset + 4)
        (params,) = struct.unpack_from("<h", data, offset + 36)
        offset += 38
        bodies[body_id] = RigidBodyPose(
            position=np.array(pose[0:3], dtype=np.float64),
            orientation=np.array(pose[3:7], dtype=np.float64),
            valid=bool(params & _TRACKING_VALID),
        )
    return bodies


class NatNetReceiver:
    """Background receiver holding the most recent rigid-body frame."""

    def __init__(self, data_socket: socket.socket, natnet_version: tuple[int, ...]):
        self._socket = data_socket
        self.natnet_version = natnet_version
        self._lock = threading.Lock()
        self._stamp = 0.0
        self._bodies: dict[int, RigidBodyPose] = {}
        self._frames = 0
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    @classmethod
    def connect(
        cls,
        host: str,
        multicast: str = MULTICAST_DEFAULT,
        command_port: int = COMMAND_PORT_DEFAULT,
        data_port: int = DATA_PORT_DEFAULT,
        local_ip: str | None = None,
    ) -> "NatNetReceiver":
        """Handshake with Motive on ``host``, then join the data multicast group.

        ``local_ip`` is the interface to join the group on; it defaults to this
        machine's address on the route to ``host``. Motive's Local Interface
        must be set to its own LAN address — on loopback the stream never
        leaves the OptiTrack PC and no packets arrive here.
        """
        if local_ip is None:
            local_ip = _local_ip_towards(host)
        version = _handshake(host, command_port, local_ip)
        if version[0] < 3:
            raise RuntimeError(
                f"NatNet {version[0]} stream from {host}: this decoder requires "
                "NatNet 3 or newer (NatNet 2 nests marker data inside each "
                "rigid body). Upgrade Motive."
            )
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_socket.bind(("", data_port))
        data_socket.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(multicast) + socket.inet_aton(local_ip),
        )
        data_socket.settimeout(_RECV_TIMEOUT_S)
        return cls(data_socket, version)

    def latest(self) -> tuple[float, dict[int, RigidBodyPose]]:
        """Return the monotonic stamp and bodies of the most recent frame.

        The stamp is 0.0 and the mapping empty until the first frame arrives.
        """
        with self._lock:
            return self._stamp, self._bodies

    @property
    def frame_count(self) -> int:
        """Data frames decoded since connecting — the true stream rate's numerator.

        A consumer that polls :meth:`latest` more slowly than frames arrive sees
        only the newest pose (which is what an MPC step wants), so counting
        consumer iterations would understate the stream.
        """
        with self._lock:
            return self._frames

    def wait_for(
        self, body_ids: Sequence[int], timeout: float
    ) -> dict[int, RigidBodyPose]:
        """Block until every id in ``body_ids`` is tracked, then return the frame."""
        deadline = time.monotonic() + timeout
        while True:
            _, bodies = self.latest()
            if all(i in bodies and bodies[i].valid for i in body_ids):
                return bodies
            if time.monotonic() > deadline:
                tracked = sorted(i for i, pose in bodies.items() if pose.valid)
                raise MocapStaleError(
                    f"Rigid bodies {list(body_ids)} not all tracked within "
                    f"{timeout:.0f}s (streamed and tracked: {tracked}). Check the "
                    "configured streaming ids and that the markers are visible."
                )
            time.sleep(_WAIT_POLL_S)

    def close(self) -> None:
        """Stop the receive thread and release the socket."""
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=2.0)
        self._socket.close()

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data = self._socket.recv(_RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                return
            if struct.unpack_from("<H", data, 0)[0] != NAT_FRAMEOFDATA:
                continue
            bodies = decode_frame(data)
            with self._lock:
                self._stamp = time.monotonic()
                self._bodies = bodies
                self._frames += 1


def _local_ip_towards(host: str) -> str:
    """This machine's address on the route to ``host`` (sends no packets)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((host, COMMAND_PORT_DEFAULT))
        return str(probe.getsockname()[0])
    finally:
        probe.close()


def _handshake(host: str, command_port: int, local_ip: str) -> tuple[int, ...]:
    """Ping Motive's command port and return the streamed NatNet version."""
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    command_socket.bind((local_ip, 0))
    command_socket.settimeout(_HANDSHAKE_TIMEOUT_S)
    try:
        command_socket.sendto(
            struct.pack("<HH", NAT_CONNECT, len(_PING_PAYLOAD)) + _PING_PAYLOAD,
            (host, command_port),
        )
        try:
            reply = command_socket.recv(_RECV_BUFFER)
        except socket.timeout as exc:
            raise RuntimeError(
                f"No NatNet reply from {host}:{command_port}. Check that Motive "
                "is running and streaming, and that its Local Interface is the "
                "OptiTrack PC's LAN address rather than loopback."
            ) from exc
    finally:
        command_socket.close()
    message_id = struct.unpack_from("<H", reply, 0)[0]
    if message_id != NAT_SERVERINFO:
        raise RuntimeError(
            f"Unexpected NatNet handshake reply id {message_id} from {host} "
            f"(expected {NAT_SERVERINFO})"
        )
    return tuple(reply[264:268])
