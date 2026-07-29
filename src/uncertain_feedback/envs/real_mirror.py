"""Mirror the sim robot's joint trajectory onto a real Kinova Gen3 arm.

The sim (:class:`~uncertain_feedback.envs.sim_mannequin.SimMannequinEnv` with
``robot="kinova_gen3"``) stays the source of truth — physics, mannequin
read-back, and planning all happen in sim. Each MPC step the achieved sim
joint configuration is forwarded to the real arm over ZMQ, which shadows it
in a 1 kHz joint tracking mode (position by default, joint impedance via
``control_mode="compliant_joint"``): the server-side OTG interpolates between
the sparse targets and holds the last one, so no streaming rate is required.

Requires a running ``emprise-gen3-controller`` ZMQ server (see
``scripts/launch_server.py`` in that repo) driving the arm.
"""

from __future__ import annotations

import atexit
import signal
import threading
import time
from types import FrameType

import numpy as np
from emprise_gen3_controller import (
    ArmController,
    CloseGripperCommand,
    JointCommand,
    OpenGripperCommand,
)
from emprise_gen3_controller.controller import _GEN3_JOINT_LIMITS
from emprise_gen3_controller.state import ControlMode

# The controller's enforced joint limits — narrower than the kortex URDF's
# for the revolute joints, so a sim shadowed by the real arm must plan
# against these, not the URDF's.
GEN3_JOINT_LIMITS: tuple[tuple[float, float], ...] = tuple(_GEN3_JOINT_LIMITS)

# Soft-limit box for streamed setpoints: the controller (client and server)
# pads the limits by 15 deg (0.2618 rad); staying a hair inside that keeps
# every command clear of its joint-limit checks. Matches the sim's
# robot_joint_limit_padding in the real-mirror config.
_SOFT_LIMIT_PAD_RAD = 0.27
_LIMITS_LOWER, _LIMITS_UPPER = (
    np.array(x, dtype=np.float64) for x in zip(*GEN3_JOINT_LIMITS)
)
_SOFT_LOWER = np.where(
    np.isfinite(_LIMITS_LOWER), _LIMITS_LOWER + _SOFT_LIMIT_PAD_RAD, _LIMITS_LOWER
)
_SOFT_UPPER = np.where(
    np.isfinite(_LIMITS_UPPER), _LIMITS_UPPER - _SOFT_LIMIT_PAD_RAD, _LIMITS_UPPER
)

# Gen3 7-DOF continuous joints (0-indexed). Sim IK accumulates unbounded
# angles on these, while the real arm reports wrapped readings; commands are
# re-branched to the equivalent angle nearest the arm's current reading.
_CONTINUOUS_JOINTS = (0, 2, 4, 6)

_HOME_TOLERANCE_RAD = float(np.deg2rad(2.0))
_HOME_TIMEOUT_S = 30.0
_POLL_PERIOD_S = 0.2
# Repositioning moves (zero, move-to-start) stream a joint-space ramp
# through position mode at this peak joint speed, rather than a high-level
# move_to (whose speed is fixed by the firmware and whose reply only arrives
# after the trajectory ends).
_MOVE_SPEED_RAD_S = 0.15
_MOVE_CMD_PERIOD_S = 0.1
# Gripper commands are asynchronous, and entering low-level servoing captures
# and holds the gripper's current position — switching modes too soon freezes
# it mid-travel. Wait for the full stroke (~1 s on the Robotiq 2F-85) first.
_GRIPPER_SETTLE_S = 2.0


class RealArmMirror:
    """Streams joint configurations from the sim robot to the real arm."""

    def __init__(
        self,
        ctrl: ArmController,
        confirm_start: bool = True,
        control_mode: str = "position_joint",
    ) -> None:
        self._ctrl = ctrl
        self._confirm_start = confirm_start
        self._track_mode = ControlMode[control_mode.upper()]
        atexit.register(self.stop)
        # Halt the arm on Ctrl+C before the exception unwinds — atexit alone
        # is not enough if the application catches KeyboardInterrupt.
        # signal.signal is main-thread-only; elsewhere atexit still covers us.
        if threading.current_thread() is threading.main_thread():
            self._prev_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        prev = self._prev_sigint
        signal.signal(signal.SIGINT, prev if prev is not None else signal.SIG_DFL)
        # If an RPC is in flight, its socket lock may be held by this very
        # thread (the handler runs on top of the interrupted frame) — calling
        # stop() here would deadlock. Unwinding releases the lock and the
        # atexit stop() halts the arm instead.
        backend = self._ctrl._backend  # pylint: disable=protected-access
        backend_lock = getattr(backend, "_cmd_lock", None)
        if backend_lock is None or not backend_lock.locked():
            self.stop()
        raise KeyboardInterrupt

    @classmethod
    def connect(
        cls,
        host: str,
        state_port: int = 5555,
        cmd_port: int = 5556,
        confirm_start: bool = True,
        control_mode: str = "position_joint",
    ) -> "RealArmMirror":
        """Open the ZMQ link to an emprise-gen3-controller server on ``host``."""
        return cls(
            ArmController.connect(host, state_port=state_port, cmd_port=cmd_port),
            confirm_start=confirm_start,
            control_mode=control_mode,
        )

    def start(self, q: np.ndarray) -> None:
        """Zero, move to the grasp configuration, open, enter tracking mode.

        The gripper closes first so the arm approaches compactly; zeroing
        makes every start move begin from the same upright reference pose.
        Both moves are streamed slowly through position mode
        (``_MOVE_SPEED_RAD_S``); unless ``confirm_start`` is False the
        operator must confirm each at the terminal, and confirm opening the
        gripper once the grasp configuration is reached (gripper commands
        need high-level mode, so the open happens between mode switches).
        The arm is left in the configured tracking mode, ready for
        :meth:`send`.
        """
        self._ctrl.execute(CloseGripperCommand())
        time.sleep(_GRIPPER_SETTLE_S)
        self.zero()
        self._move_and_wait(np.asarray(q, dtype=np.float64))
        self._ctrl.switch_mode(ControlMode.HIGH_LEVEL)
        if self._confirm_start:
            input("[mirror] At the grasp pose. Press Enter to open the gripper: ")
        self._ctrl.execute(OpenGripperCommand())
        time.sleep(_GRIPPER_SETTLE_S)
        self._ctrl.switch_mode(self._track_mode)

    def start_from_grasp(self) -> None:
        """Enter tracking mode without moving, for an already-taken grasp.

        What :class:`~uncertain_feedback.envs.real.RealEnv` uses: the gripper is
        already closed on the person's forearm, so there is nothing to approach
        and the gripper must not be touched. Unlike :meth:`start` this only hands
        control to the tracking mode, ready for :meth:`send`.
        """
        if self._confirm_start:
            input("\n[mirror] Grasp already taken. Press Enter to start tracking: ")
        self._ctrl.switch_mode(self._track_mode)

    def zero(self) -> None:
        """Move every joint to zero (the upright reference pose)."""
        self._move_and_wait(np.zeros(len(GEN3_JOINT_LIMITS)))

    def send(self, q: np.ndarray) -> None:
        """Command the arm to track one sim joint configuration."""
        self._ctrl.execute(
            JointCommand(self._nearest_branch(np.asarray(q, dtype=np.float64)))
        )

    def current_q(self) -> np.ndarray:
        """The arm's measured joint configuration."""
        return np.asarray(self._ctrl.get_state().joint_positions, dtype=np.float64)

    def stop(self) -> None:
        """Return the arm to high-level mode (holds position via servoing)."""
        try:
            self._ctrl.switch_mode(ControlMode.HIGH_LEVEL)
        except Exception:  # pylint: disable=broad-except
            # An interrupted RPC leaves its stale reply queued on the socket,
            # failing the next round-trip; the server processes requests
            # regardless, so one retry both resyncs and re-issues the switch.
            self._ctrl.switch_mode(ControlMode.HIGH_LEVEL)

    def _move_and_wait(self, q: np.ndarray) -> None:
        target = np.clip(self._nearest_branch(q), _SOFT_LOWER, _SOFT_UPPER)
        current = self._ctrl.get_state().joint_positions
        distance = float(np.max(np.abs(target - current)))
        if self._confirm_start:
            response = input(
                f"\n[mirror] Move distance: {np.rad2deg(distance):.1f} deg "
                f"(max joint delta) at {_MOVE_SPEED_RAD_S:.2f} rad/s. "
                f"Proceed? (y/n): "
            )
            if response.strip().lower() != "y":
                raise RuntimeError("Repositioning move aborted by operator")
        # Repositioning ramps always run stiff: the reach-target check below
        # assumes accurate tracking, which a compliant mode does not provide.
        self._ctrl.switch_mode(ControlMode.POSITION_JOINT)
        n_steps = max(
            1, int(np.ceil(distance / (_MOVE_SPEED_RAD_S * _MOVE_CMD_PERIOD_S)))
        )
        for k in range(1, n_steps + 1):
            # The arm may start inside the padding band (e.g. parked near a
            # limit); clamping pulls it out immediately instead of tripping
            # the joint-limit check on the early setpoints.
            setpoint = np.clip(
                current + (target - current) * (k / n_steps),
                _SOFT_LOWER,
                _SOFT_UPPER,
            )
            self._ctrl.execute(JointCommand(setpoint))
            time.sleep(_MOVE_CMD_PERIOD_S)
        deadline = time.monotonic() + _HOME_TIMEOUT_S
        while self._max_joint_error(target) > _HOME_TOLERANCE_RAD:
            if time.monotonic() > deadline:
                error_deg = np.rad2deg(self._max_joint_error(target))
                raise RuntimeError(
                    f"Real arm did not reach the commanded configuration "
                    f"(max joint error {error_deg:.1f} deg)"
                )
            time.sleep(_POLL_PERIOD_S)

    def _nearest_branch(self, q: np.ndarray) -> np.ndarray:
        """Re-branch continuous joints nearest the arm's current reading."""
        current = self._ctrl.get_state().joint_positions
        target = q.copy()
        idx = list(_CONTINUOUS_JOINTS)
        delta = q[idx] - current[idx]
        target[idx] = current[idx] + np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
        return target

    def _max_joint_error(self, target: np.ndarray) -> float:
        current = self._ctrl.get_state().joint_positions
        delta = target - current
        delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
        return float(np.max(np.abs(delta)))
