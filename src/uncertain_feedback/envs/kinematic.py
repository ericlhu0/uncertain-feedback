"""Kinematic execution env: the commanded configuration is achieved exactly."""

from __future__ import annotations

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv


class KinematicEnv(ExecutionEnv):
    """Pass-through env reproducing the original open-loop kinematic rollout."""

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        return q_cmd
