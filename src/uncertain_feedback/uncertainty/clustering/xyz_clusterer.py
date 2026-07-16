"""XYZ-position-based trajectory clusterers."""

from __future__ import annotations

import numpy as np

from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SmplLeftArmFK,
)
from uncertain_feedback.uncertainty.clustering.base import (
    TrajectoryClusterer,
    agglomerative_labels,
)


def end_pose_features(positions: np.ndarray) -> np.ndarray:
    """Spine3-relative arm-chain positions at a single late frame.

    Args:
        positions: ``(num_samples, n_frames, 22, 3)`` global SMPL joint
            positions.

    Returns:
        ``(num_samples, 15)`` features: the arm chain
        ``[spine3, left_collar, left_shoulder, left_elbow, left_wrist]`` at
        frame ``min(100, n_frames - 1)``, relative to spine3, flattened.
    """
    num_samples, n_frames = positions.shape[:2]
    frame_idx = min(100, n_frames - 1)
    spine3_j = LEFT_ARM_CHAIN_INDICES[0]
    arm_chain = positions[:, frame_idx, LEFT_ARM_CHAIN_INDICES, :]
    return (arm_chain - positions[:, frame_idx, spine3_j, None, :]).reshape(
        num_samples, -1
    )


class XyzPositionClusterer(TrajectoryClusterer):
    """Cluster trajectories by a single-frame end-pose feature vector.

        Each sample is reduced to the spine3-relative XYZ positions of the
        left-arm chain ``[spine3, left_collar, left_shoulder, left_elbow,
        left_wrist]`` at frame ``min(100, n_frames - 1)``, flattened to a
        15-dim vector, and the resulting ``(num_samples, 15)`` matrix is
        clustered with KMeans (the base-class default).

        Args:
            n_clusters:   Number of clusters (K in KMeans).
            fk:           :class:`~uncertain_feedback.planners.mpc.kinematics\
    .SmplLeftArmFK` instance used for forward kinematics.  If ``None``,
                          a default instance is constructed (requires the SMPL
                          PKL file inside the MDM submodule).
            random_state: Random seed forwarded to KMeans for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        fk: SmplLeftArmFK | None = None,
        random_state: int = 0,
    ) -> None:
        super().__init__(n_clusters, random_state)
        self._fk = fk if fk is not None else SmplLeftArmFK()

    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        """Convert arm trajectories to ``(num_samples, 5*3)`` XYZ features."""
        num_samples, n_frames, _, _ = trajectories.shape
        frame_idx = min(100, n_frames - 1)
        poses = trajectories[:, frame_idx]  # (num_samples, 3, 3)
        positions = self._fk.fk_batch(poses)
        return positions.reshape(num_samples, -1).astype(np.float64)

    def _positions_to_features(self, positions: np.ndarray) -> np.ndarray:
        return end_pose_features(positions)


class AggloEndPoseClusterer(XyzPositionClusterer):
    """End-pose features clustered with average-linkage agglomerative."""

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        return agglomerative_labels(features, self._n_clusters)
