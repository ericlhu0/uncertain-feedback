"""Action-space contract for the sampling MPC.

An action space owns one sampling representation: it draws warm-started action
sequences, integrates them into rollouts, shapes the stage cost with any
space-specific feasibility terms, and executes the chosen command through the
env. The MPC solve loop is space-agnostic — it only sees a
:class:`RolloutBatch` and a :data:`StageCost`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv


@dataclass
class RolloutBatch:
    """One solve call's sampled rollouts, in every representation costs need.

    ``actions`` and ``aa_trajs`` are always present; the remaining fields are
    per-space (human rollouts carry ``q_trajs``, robot rollouts carry the
    projected ``wrist_pos``/``robot_trajs``/``grasp_residual``).
    """

    actions: np.ndarray  # (N, H, 7) sampled action sequences
    aa_trajs: np.ndarray  # (N, H+1, 3, 3) arm axis-angles (the cost boundary)
    q_trajs: np.ndarray | None = None  # (N, H+1, 7) human-q rollouts
    wrist_pos: np.ndarray | None = None  # (N, H+1, 3) projected wrist positions
    robot_trajs: np.ndarray | None = None  # (N, H+1, 7) robot joint rollouts
    grasp_residual: np.ndarray | None = None  # (N, H+1) projection residual


StageCost = Callable[[RolloutBatch], np.ndarray]
"""Maps a rollout batch to an ``(N,)`` cost per rollout."""


class ActionSpace(ABC):
    """One sampling representation for the MPC solve loop."""

    @abstractmethod
    def rollouts(
        self, env: ExecutionEnv, current_q: np.ndarray, mean: np.ndarray
    ) -> RolloutBatch:
        """Sample ``(N, H, 7)`` actions around ``mean`` and integrate them."""

    @abstractmethod
    def shape_costs(self, batch: RolloutBatch, stage_cost: StageCost) -> np.ndarray:
        """Evaluate ``stage_cost`` plus any space-specific feasibility terms."""

    @abstractmethod
    def command(self, batch: RolloutBatch, best_idx: int) -> np.ndarray:
        """The next executable command from the chosen rollout."""

    @abstractmethod
    def execute(
        self, env: ExecutionEnv, current_q: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        """Realize ``command`` through the env; returns the achieved state."""

    @abstractmethod
    def hold(self, env: ExecutionEnv, current_q: np.ndarray) -> np.ndarray:
        """Hold the current pose (nothing left to do)."""
