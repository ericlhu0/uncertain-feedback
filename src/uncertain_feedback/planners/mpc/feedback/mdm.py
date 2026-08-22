"""MDM feedback: natural-language corrections played back frame by frame.

The Motion Diffusion Model translates a natural-language correction into a
plausible arm trajectory; :class:`MdmFeedback` holds that trajectory and walks
a rate-limited cursor along it. Each step moves toward the current frame by at
most ``max_playback_delta`` radians per joint, so large frame-to-frame jumps
(and the initial jump from the live pose into the trajectory) are traversed
smoothly rather than snapped in a single step.

With a feasibility constraint active, the cursor additionally tracks closest
approach to the current frame — monotone, so measurement jitter cannot reset
the counter — and skips a frame after ``stall_steps`` steps without progress,
so an unreachable stretch cannot stall the run. A person actively resisting
reads as a stall too, which is the desired yielding behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uncertain_feedback.planners.mpc.feedback.base import FeedbackMethod
from uncertain_feedback.planners.mpc.kinematics import (
    _frame_distance,
    _rate_limited_step_q,
)
from uncertain_feedback.uncertainty.uq_selector import UqConfig


@dataclass(frozen=True)
class FeedbackConfig:
    """MDM feedback parameters (the ``feedback:`` section).

    Args:
        max_playback_delta: Maximum per-joint rotation (radians) applied per
                            step while following the MDM trajectory.
        trajectory_fraction: Fraction of MDM-generated frames to enqueue
                            (e.g. ``0.75`` enqueues the first 75 % of frames).
        frames:             Exact number of MDM frames to generate (``None``
                            keeps the generator default). Run-level.
        text_time:          Step at which the run pauses for a correction
                            prompt. Run-level.
        anchor_correction:  Re-anchor the generated correction onto the arm's
                            current configuration before tracking it, dropping
                            the pinned frame (see
                            ``SmplLeftArmFK.anchor_arm_trajectory``). Removes the
                            frame-0 seam; set false to track the raw sample.
        uq:                 Optional UQ layer: sample several diffusion
                            outputs, cluster, and pick, instead of following a
                            single sample.
    """

    max_playback_delta: float = 0.05
    trajectory_fraction: float = 1.0
    frames: int | None = None
    text_time: int = 0
    anchor_correction: bool = True
    uq: UqConfig | None = None


@dataclass
class MdmFeedback(FeedbackMethod):
    """Direct playback buffer for a validated MDM trajectory.

    Args:
        max_playback_delta: Per-step angular cap while following the frames.
        trajectory_fraction: Fraction of generated frames enqueued per query.
        stall_steps: Steps without closest-approach progress before the
            current frame is skipped; ``None`` disables the stall-skip (no
            feasibility constraint active).
        anchor_correction: Re-anchor a generated correction onto the current
            configuration before enqueuing it.
    """

    max_playback_delta: float = 0.05
    trajectory_fraction: float = 1.0
    stall_steps: int | None = None
    anchor_correction: bool = True

    mdm_goal: np.ndarray | None = None
    preview_q: np.ndarray | None = None
    _frames: np.ndarray | None = field(default=None, repr=False)
    _idx: int = 0
    _stall_best_dist: float = np.inf
    _stall_count: int = 0

    @property
    def started(self) -> bool:
        return self._frames is not None

    def in_playback(self) -> bool:
        return self._frames is not None and self._idx < len(self._frames)

    def set_frames(self, q_frames: np.ndarray) -> None:
        self._frames = q_frames
        self._idx = 0
        self.preview_q = q_frames[-1].copy()

    def remaining(self, current_q: np.ndarray) -> np.ndarray | None:
        if not self.in_playback():
            return None
        assert self._frames is not None
        remaining = self._frames[self._idx :]
        return np.concatenate((current_q[np.newaxis], remaining.copy()), axis=0)

    def advance(self, current_q: np.ndarray) -> np.ndarray:
        """One rate-limited step toward the current frame.

        Advances the cursor only once the frame is reached; with
        ``stall_steps`` set, a frame the arm stops making progress toward is
        skipped instead of holding the cursor forever.
        """
        assert self._frames is not None
        idx = self._idx
        target_q = self._frames[idx]
        next_q, reached = _rate_limited_step_q(
            current_q, target_q, self.max_playback_delta
        )
        if reached:
            self._idx += 1
            self.reset_stall()
            return next_q
        if self.stall_steps is None:
            return next_q
        dist = _frame_distance(current_q, target_q)
        if dist < self._stall_best_dist - 0.1 * self.max_playback_delta:
            self._stall_best_dist = dist
            self._stall_count = 0
            return next_q
        self._stall_count += 1
        if self._stall_count >= self.stall_steps:
            print(
                f"[playback] frame {idx} stalled {dist:.4f} rad away after "
                f"{self._stall_count} steps without progress; skipping."
            )
            self._idx += 1
            self.reset_stall()
        return next_q

    def reset_stall(self) -> None:
        self._stall_best_dist = np.inf
        self._stall_count = 0
