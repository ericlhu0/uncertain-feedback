"""Feedback-method contract: how a user correction enters the MPC.

A feedback method turns a user's correction into a trajectory the planner
plays back directly — one rate-limited frame per step, bypassing the sampling
optimisation — before the goal-space phase resumes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeedbackMethod(ABC):
    """Playback state machine for a corrected trajectory."""

    trajectory_fraction: float
    mdm_goal: np.ndarray | None
    preview_q: np.ndarray | None

    @property
    @abstractmethod
    def started(self) -> bool:
        """Whether a correction has ever been queued."""

    @abstractmethod
    def in_playback(self) -> bool:
        """Whether a trajectory is still being followed frame by frame."""

    @abstractmethod
    def set_frames(self, q_frames: np.ndarray) -> None:
        """Queue a ``(n_frames, 7)`` trajectory for direct playback."""

    @abstractmethod
    def advance(self, current_q: np.ndarray) -> np.ndarray:
        """One rate-limited step toward the current frame, advancing the
        cursor once the frame is reached."""

    @abstractmethod
    def remaining(self, current_q: np.ndarray) -> np.ndarray | None:
        """The current pose followed by the unexecuted playback targets."""

    @abstractmethod
    def reset_stall(self) -> None:
        """Reset the stall-skip tracker (new trajectory or new frame)."""
