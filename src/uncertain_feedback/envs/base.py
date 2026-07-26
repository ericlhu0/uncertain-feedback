"""Shared interface for execution environments.

An :class:`ExecutionEnv` is the boundary between the planner and the world
that physically moves the human arm. Each MPC step the planner produces a
commanded ``(7,)`` joint configuration; the env realizes it (kinematically,
through a simulated robot, or on real hardware) and returns the configuration
actually achieved, which feeds the next planning step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


class ExecutionEnv(ABC):
    """Abstract base class for execution environments."""

    def __init__(self) -> None:
        self._fk: SmplLeftArmFK | None = None
        self._spine3_pos: np.ndarray | None = None
        self._spine3_aa: np.ndarray | None = None
        self._body_pos: np.ndarray | None = None

    def set_pose_context(
        self,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        body_pos: np.ndarray | None = None,
    ) -> None:
        """Attach the run's kinematics so the env matches the planner's FK.

        ``body_pos`` is the ``(22, 3)`` decoded initial body pose; envs that
        render the whole body use it, defaulting to the SMPL T-pose.
        """
        self._fk = fk
        self._spine3_pos = spine3_pos
        self._spine3_aa = spine3_aa
        self._body_pos = body_pos

    def show_goal(self, q_goal: np.ndarray) -> None:
        """Display the ``(7,)`` configuration the run drives toward.

        Called once the goal is known, which is after :meth:`initial_q` — a
        measured torso anchor moves every spine3-relative goal with it. Default:
        envs with nothing to draw ignore it.
        """

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

    @abstractmethod
    def visualize(self, path: Path | None = None) -> np.ndarray:
        """Render the last executed configuration as an ``(H, W, 3)`` image.

        Saves the image to ``path`` when given. Requires
        :meth:`set_pose_context` and at least one :meth:`execute`/:meth:`hold`.
        """

    @abstractmethod
    def save_video(self, path: str | Path, fps: int = 20) -> None:
        """Write a video of every configuration executed or held so far.

        Requires :meth:`set_pose_context` and at least one
        :meth:`execute`/:meth:`hold`.
        """
