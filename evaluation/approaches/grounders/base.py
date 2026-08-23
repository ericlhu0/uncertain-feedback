"""Grounder ABC: turns one utterance into candidate motions and a correction."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.rig import EvalRig
from evaluation.structs import GroundingResult, InteractionTask
from uncertain_feedback.planners.mpc.kinematics import SMPL_JOINT_NAMES_22
from uncertain_feedback.simulated_users import SimulatedUser

ClusterSelector = Callable[[dict[int, np.ndarray]], tuple[int, float]]

# Landmarks a seated-care correction plausibly references, as (prompt name,
# SMPL-22 joint index); "chest" is the prompt-facing name for spine3.
LANDMARKS = tuple(
    (name, SMPL_JOINT_NAMES_22.index(joint))
    for name, joint in (
        ("pelvis", "pelvis"),
        ("left_hip", "left_hip"),
        ("chest", "spine3"),
        ("neck", "neck"),
        ("head", "head"),
        ("right_shoulder", "right_shoulder"),
    )
)


class Grounder(abc.ABC):
    """One grounding mechanism: language to candidate motions to selection."""

    requires_generator: bool = False

    def __init__(self) -> None:
        self._rig: EvalRig | None = None
        self._user: SimulatedUser | None = None
        self._episode_dir = Path(".")

    @property
    def rig(self) -> EvalRig:
        """The bound rig; valid after :meth:`reset`."""
        assert self._rig is not None, "reset() must run before use"
        return self._rig

    @property
    def user(self) -> SimulatedUser:
        """The bound persona; valid after :meth:`reset`."""
        assert self._user is not None, "reset() must run before use"
        return self._user

    def reset(
        self,
        rig: EvalRig,
        user: SimulatedUser,
        task: InteractionTask,
        episode_dir: Path,
    ) -> None:
        """Bind the episode; subclasses extend for per-episode state."""
        del task
        self._rig = rig
        self._user = user
        self._episode_dir = episode_dir

    @abc.abstractmethod
    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
        goal: np.ndarray,
    ) -> GroundingResult:
        """Turn one utterance into candidate motions and a selected correction.

        Must call ``cluster_selector`` exactly once, as its last selector use.
        """
