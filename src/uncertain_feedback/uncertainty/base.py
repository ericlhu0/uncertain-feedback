"""Abstract base class for trajectory clusterers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TrajectoryClusterer(ABC):  # pylint: disable=too-few-public-methods
    """Cluster a batch of arm trajectories into integer labels.

    Subclasses implement :meth:`cluster` to map a controlled
    ``(num_samples, n_frames, 3, 3)`` axis-angle trajectory batch to a
    ``(num_samples,)`` array of integer cluster labels.  Legacy full-arm
    ``(..., 4, 3)`` batches may also be accepted by concrete implementations.
    """

    @abstractmethod
    def cluster(self, trajectories: np.ndarray) -> np.ndarray:
        """Assign integer cluster labels to a batch of trajectories.

                Args:
                    trajectories: ``(num_samples, n_frames, 3, 3)`` axis-angle
                                  trajectory batch, as returned by
                                  :meth:`~uncertain_feedback.motion_generators.mdm\
        .mdm_api.MdmMotionGenerator.generate_left_arm_trajectory` with
                                  ``num_samples > 1``.

                Returns:
                    ``(num_samples,)`` integer cluster labels, contiguous from 0.
        """
