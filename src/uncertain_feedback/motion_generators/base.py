"""Shared interface for text-to-motion backends.

A :class:`MotionGenerator` turns a natural-language prompt into a left-arm
trajectory the MPC can track. Backends differ in their internal pose
representation (e.g. MDM uses HML263 feature vectors), so callers treat the
``pose`` arrays returned by
:meth:`load_pose` / :meth:`build_pose_from_arm_aa` as opaque: they load a
pose, decode it, optionally patch the arm configuration into it, and pass it
back as ``start_pose`` — never inspecting its contents.

The shared, backend-agnostic conversion
:meth:`smpl_positions_to_left_arm_trajectory` (SMPL XYZ positions → arm
axis-angles via :class:`SmplLeftArmFK`) is implemented here so every backend
inherits it.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    smpl_body_pose_to_arm_aa,
    smpl_positions_batch_to_body_pose,
)
from uncertain_feedback.motion_generators.steering import SteeringEvent, SteeringSpec
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


class MotionGenerator(ABC):
    """Abstract base class for text-to-motion backends.

    Subclasses implement the abstract methods using their own pose
    representation. All trajectory outputs are in the shared MPC conventions:
    left-arm axis-angles ``(n_frames, 3, 3)`` for
    ``[left_shoulder, left_elbow, left_wrist]`` and SMPL joint positions
    ``(n_frames, 22, 3)``.
    """

    def __init__(self) -> None:
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

    @abstractmethod
    def load_pose(self, path: str | Path) -> np.ndarray:
        """Load a saved start pose in the backend's own representation."""

    @abstractmethod
    def decode_pose(
        self, pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode a pose into MPC state, body positions, spine, and collar.

        Returns ``(arm_aa (3,3), body_positions (22,3), spine3_aa (3,), collar_aa (3,))``.
        """

    @abstractmethod
    def build_pose_from_arm_aa(
        self,
        base_pose: np.ndarray,
        arm_aa: np.ndarray,
    ) -> np.ndarray:
        """Return a copy of ``base_pose`` with the arm joints set to ``arm_aa``."""

    @abstractmethod
    def generate_left_arm_trajectory(
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        save_path: str | Path | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        spine3_aa: np.ndarray | None = None,
        *,
        steering: SteeringSpec | None = None,
    ) -> np.ndarray:
        """Generate a left-arm axis-angle trajectory from text.

        Returns ``(n_frames, 3, 3)`` when ``num_samples == 1``, otherwise
        ``(num_samples, n_frames, 3, 3)``.
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
    ) -> np.ndarray:
        """Generate samples and return ``(num_samples, n_frames, 22, 3)`` SMPL positions."""

    # ------------------------------------------------------------------
    # Shared, backend-agnostic conversion
    # ------------------------------------------------------------------

    def _align_fk_collar_to_pose(self, start_pose: np.ndarray | None) -> None:
        """Set the internal FK's fixed collar from ``start_pose``.

        Arm extraction (:meth:`smpl_positions_to_left_arm_trajectory`) expresses
        the shoulder relative to ``self._fk.collar_aa``.  The MPC sets its own
        FK's collar to the start pose's collar, so the generator must use the
        same collar or the extracted trajectory lands in a different frame
        (identity collar) than the one the MPC tracks.  Backends call this at
        the top of their ``generate_*`` methods.
        """
        if start_pose is not None:
            self._fk.collar_aa = np.asarray(
                self.decode_pose(start_pose)[3], dtype=np.float64
            )

    def smpl_positions_to_left_arm_trajectory(
        self,
        positions: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert SMPL XYZ positions to a left-arm axis-angle trajectory.

        Args:
            positions: ``(n_frames, 22, 3)`` or ``(n_samples, n_frames, 22, 3)``
                global SMPL joint positions.
            spine3_aa: Optional fixed MPC spine3 world axis-angle.

        Returns:
            ``(n_frames, 3, 3)`` for a single trajectory, otherwise
            ``(n_samples, n_frames, 3, 3)``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        single = positions.ndim == 3
        if single:
            positions = positions[None, ...]

        convert_t0 = time.perf_counter()
        if spine3_aa is not None:
            arm_aa = self._fk.arm_aa_from_positions_batch(
                positions,
                spine3_aa=spine3_aa,
            )
            print(
                "[timing] selected position-to-fixed-base arm IK total: "
                f"{time.perf_counter() - convert_t0:.3f}s"
            )
            return arm_aa[0] if single else arm_aa

        body_pose = smpl_positions_batch_to_body_pose(
            positions, self._fk.tpose_all_joints
        )
        print(
            "[timing] selected position-to-arm IK total: "
            f"{time.perf_counter() - convert_t0:.3f}s"
        )
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        return arm_aa[0] if single else arm_aa
