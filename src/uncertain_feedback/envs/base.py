"""Shared interface for execution environments.

An :class:`ExecutionEnv` is the boundary between the planner and the world
that physically moves the human arm. Each MPC step the planner produces a
commanded ``(7,)`` joint configuration; the env realizes it (kinematically,
through a simulated robot, or on real hardware) and returns the configuration
actually achieved, which feeds the next planning step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ExecutionEnv(ABC):
    """Abstract base class for execution environments."""

    @abstractmethod
    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        """Realize one commanded ``(7,)`` arm configuration.

        Blocks until the step has been executed and returns the ``(7,)``
        configuration actually achieved.
        """

    def hold(self, q: np.ndarray) -> np.ndarray:
        """Handle one planner step that commands no motion.

        Default: send nothing and report ``q`` unchanged. Robot envs may
        override to actively hold the arm at ``q`` (e.g. keep an impedance
        controller engaged).
        """
        return q
