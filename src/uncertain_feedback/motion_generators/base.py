"""Shared interface for text-to-motion backends.

A :class:`MotionGenerator` turns a natural-language prompt into sampled body
motion. Backends differ in their internal pose representation (e.g. MDM uses
HML263 feature vectors); the pose they condition on comes from the
:class:`~uncertain_feedback.planners.mpc.human.Human` being moved
(``human.hml_pose``), patched with its current arm or recent arm history.

The drawn samples are SMPL joint positions in the backend's own frame and
proportions. Callers turn them into the person's arm states with
``human.ik_q_from_positions`` and never compare them with positions the Human
computes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from uncertain_feedback.motion_generators.steering import SteeringEvent, SteeringSpec
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.human import Human


class MotionGenerator(ABC):
    """Abstract base class for text-to-motion backends.

    Subclasses implement the abstract methods using their own pose
    representation. Samples are SMPL joint positions ``(n_frames, 22, 3)``.
    """

    def __init__(self) -> None:
        # The SMPL template skeleton, for encoding arm poses; no person's collar.
        self._fk: SmplLeftArmFK = SmplLeftArmFK()

    @property
    def last_steering_events(self) -> tuple[SteeringEvent, ...]:
        """Steering diagnostics from the most recent generation.

        Empty for backends that do not implement ``steering`` and for unsteered
        generations.
        """
        return ()

    # ------------------------------------------------------------------
    # Abstract interface (backend-specific pose representation)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def prefix_frames(self) -> int:
        """Number of leading frames pinned as generation-time conditioning."""

    @abstractmethod
    def build_pose_from_arm_aa(
        self,
        base_pose: np.ndarray,
        arm_aa: np.ndarray,
    ) -> np.ndarray:
        """Return a copy of ``base_pose`` with the arm joints set to ``arm_aa``."""

    @abstractmethod
    def build_prefix_from_arm_history(
        self,
        base_pose: np.ndarray,
        arm_aa_seq: np.ndarray,
    ) -> np.ndarray:
        """Return a ``(prefix_frames, 263)``-style prefix from an arm history.

        ``arm_aa_seq`` is ``(prefix_frames, 3, 3)``, oldest → newest, ending at
        the current configuration.  The result is passed as ``start_pose`` to
        condition generation on the recent trajectory instead of a single pose.
        """

    @abstractmethod
    def generate_left_arm_position_samples(
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        *,
        steering: SteeringSpec | None = None,
        speed: float | None = None,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Generate samples and return ``(num_samples, n_frames, 22, 3)`` SMPL positions.

        The ``prefix_frames`` frames pinned to ``start_pose`` are
        generation-time conditioning only: they are additive to the requested
        length and never part of the returned motion, apart from the last one,
        which is frame 0 of the output — the configuration the arm is in now.
        ``num_frames`` is therefore the returned length. ``speed`` requests a
        mean left-arm speed in metres per frame; backends without an explicit
        speed channel ignore it. ``save_path`` writes a diagnostic video of the
        first sample, prefix included.
        """

    # ------------------------------------------------------------------
    # Human-facing generation
    # ------------------------------------------------------------------

    def start_pose(self, human: Human, prefix: bool) -> np.ndarray:
        """The conditioning pose for a motion that continues from ``human``.

        With ``prefix`` it is the arm's last :attr:`prefix_frames` executed
        states (padded with the oldest when the history is shorter) patched
        into the person's pose; otherwise just the current arm state.
        """
        if human.hml_pose is None:
            raise ValueError("Motion generation needs a Human built from a pose file.")
        if not prefix:
            return self.build_pose_from_arm_aa(
                human.hml_pose, human.arm_aa_from_q(human.q)
            )
        recent = human.history[-self.prefix_frames :]
        padding = np.repeat(recent[:1], self.prefix_frames - len(recent), axis=0)
        return self.build_prefix_from_arm_history(
            human.hml_pose, human.arm_aa_from_q(np.concatenate([padding, recent]))
        )

    def generate_positions(
        self,
        text: str,
        human: Human,
        *,
        prefix: bool,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        steering: SteeringSpec | None = None,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Sample ``(num_samples, n_frames, 22, 3)`` motions continuing from ``human``.

        Frame 0 of each sample is the conditioning's last frame (the current
        arm state); see :meth:`start_pose` for ``prefix``.
        """
        return self.generate_left_arm_position_samples(
            text,
            start_pose=self.start_pose(human, prefix),
            num_samples=num_samples,
            num_frames=num_frames,
            frozen_body=frozen_body,
            steering=steering,
            save_path=save_path,
        )
