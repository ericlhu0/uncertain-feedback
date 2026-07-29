"""Feasibility-constraint contract for the sampling MPC.

A constraint answers one question in three places: can execution actually
realize this motion? Infeasible rollouts are masked to infinite cost in the
solve loop; feedback trajectories are screened frame by frame when pushed;
and each playback step can be held when the next command is unreachable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uncertain_feedback.planners.mpc.action_spaces.base import RolloutBatch


class FeasibilityConstraint(ABC):
    """One feasibility check applied to rollouts and feedback playback."""

    playback_stall_steps: int | None = None
    """Consecutive steps without closest-approach progress on a playback frame
    before the cursor skips it; ``None`` disables the stall-skip."""

    @abstractmethod
    def rollout_feasible(self, batch: RolloutBatch) -> np.ndarray:
        """``(N,)`` mask of rollouts execution can realize."""

    def screen_frames(
        self, q_frames: np.ndarray, current_q: np.ndarray
    ) -> np.ndarray:
        """Drop frames of a feedback trajectory this constraint rules out."""
        _ = current_q
        return q_frames

    def step_reachable(self, current_q: np.ndarray, q_cmd: np.ndarray) -> bool:
        """Whether the next playback command can be executed from here."""
        _ = current_q, q_cmd
        return True
