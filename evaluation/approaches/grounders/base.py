"""Grounder ABC: turns one utterance into candidate motions and a correction."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.human import Human
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


def required_llm_model(cfg: MpcRunConfig) -> str:
    """The planner yaml's ``llm_cost.model``, which LLM-backed grounders need."""
    model = cfg.llm_cost.model
    if model is None:
        raise ValueError("LLM grounders need llm_cost.model in the planner yaml.")
    return model


class Grounder(abc.ABC):
    """One grounding mechanism: language to candidate motions to selection."""

    requires_generator: bool = False

    def __init__(self) -> None:
        self._cfg: MpcRunConfig | None = None
        self._gen: MotionGenerator | None = None
        self._user: SimulatedUser | None = None
        self._episode_dir = Path(".")

    @property
    def cfg(self) -> MpcRunConfig:
        """The bound planner config; valid after :meth:`reset`."""
        assert self._cfg is not None, "reset() must run before use"
        return self._cfg

    @property
    def gen(self) -> MotionGenerator:
        """The bound motion generator; only generator-backed grounders have one."""
        assert self._gen is not None, "this grounder needs a motion generator"
        return self._gen

    @property
    def user(self) -> SimulatedUser:
        """The bound persona; valid after :meth:`reset`."""
        assert self._user is not None, "reset() must run before use"
        return self._user

    def reset(
        self,
        cfg: MpcRunConfig,
        gen: MotionGenerator | None,
        user: SimulatedUser,
        seed: int,
        episode_dir: Path,
    ) -> None:
        """Bind the episode; subclasses extend for per-episode state."""
        del seed
        self._cfg = cfg
        self._gen = gen
        self._user = user
        self._episode_dir = episode_dir

    def begin_goal(self, goal: np.ndarray, oracle_path: np.ndarray) -> None:
        """Called once per goal before any round; the oracle grounder reads it."""
        del goal, oracle_path

    @abc.abstractmethod
    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        """Turn one utterance into candidate motions and a selected correction.

        ``human`` is the person at the feedback moment (``human.q`` is where the
        correction starts). Must call ``cluster_selector`` exactly once, as its last selector use.
        """
