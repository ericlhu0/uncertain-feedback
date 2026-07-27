"""Operator-driven pausing for live runs on the real rig.

A scripted run injects one preset correction at ``text_time``. A real person
speaks when *they* want to, so both the pause and the words have to arrive
mid-run — which means watching stdin without ever blocking the MPC step that is
holding their arm.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import TextIO


class OperatorPause:
    """Stdin watcher letting the operator pause a live run and type a correction.

    One reader thread owns stdin, so polling between MPC steps never blocks and
    the feedback prompt cannot race the watcher for the operator's line. The
    first line requests the pause; if it already carries the correction that text
    is used, otherwise :meth:`feedback` asks for one. Both come off the same
    queue, so a line typed while the previous round was still generating requests
    the next pause instead of being swallowed.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._lines: queue.Queue[str] = queue.Queue()
        self._pending: str | None = None
        threading.Thread(target=self._read_lines, daemon=True).start()
        print(
            "[interactive] press enter to pause and correct, "
            "or type the correction and press enter"
        )

    def _read_lines(self) -> None:
        for line in self._stream:
            self._lines.put(line.strip())

    def requested(self) -> bool:
        """Whether the operator has asked to pause, without consuming their line."""
        if self._pending is None:
            try:
                self._pending = self._lines.get_nowait()
            except queue.Empty:
                return False
        return True

    def feedback(self, step: int) -> str:
        """Block for the correction to apply; the arm stays held meanwhile."""
        line = self._pending or ""
        self._pending = None
        print(f"\n>>> paused at step {step} — the robot is holding the arm.")
        while not line:
            print("feedback: ", end="", flush=True)
            line = self._lines.get()
        print(f">>> correcting with: {line!r}")
        return line
