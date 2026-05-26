"""XYZ-position-based trajectory clusterer."""

from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans

from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SmplLeftArmFK,
)
from uncertain_feedback.uncertainty.base import TrajectoryClusterer


class XyzPositionClusterer(
    TrajectoryClusterer
):  # pylint: disable=too-few-public-methods
    """Cluster trajectories by their FK joint-position feature vectors.

        Each trajectory ``(n_frames, 3, 3)`` is converted to XYZ positions via
        the controlled-arm FK helpers,
        producing ``(n_frames, 5, 3)`` world positions for
        ``[spine3, left_collar, left_shoulder, left_elbow, left_wrist]``.
        These are flattened to a ``(n_frames * 5 * 3,)`` feature vector per
        sample, and the resulting ``(num_samples, n_frames * 5 * 3)`` matrix is
        clustered with KMeans.

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
        self._n_clusters = n_clusters
        self._fk = fk if fk is not None else SmplLeftArmFK()
        self._random_state = random_state

    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        """Convert arm trajectories to ``(num_samples, 5*3)`` XYZ features."""
        num_samples, n_frames, _, _ = trajectories.shape
        frame_idx = min(100, n_frames - 1)
        poses = trajectories[:, frame_idx]  # (num_samples, 3, 3)
        positions = self._fk.fk_batch(poses)
        return positions.reshape(num_samples, -1).astype(np.float64)

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=10,
        )
        kmeans_t0 = time.perf_counter()
        labels = kmeans.fit_predict(features).astype(np.intp)
        print(f"[timing] KMeans fit_predict: {time.perf_counter() - kmeans_t0:.3f}s")
        return labels

    def cluster(self, trajectories: np.ndarray) -> np.ndarray:
        """Cluster trajectories by XYZ joint positions.

        Args:
            trajectories: ``(num_samples, n_frames, 3, 3)`` axis-angle batch.

        Returns:
            ``(num_samples,)`` integer labels in ``[0, n_clusters)``.

        Raises:
            ValueError: If ``num_samples < n_clusters``.
        """
        trajectories = np.asarray(trajectories, dtype=np.float64)
        num_samples = trajectories.shape[0]
        if num_samples < self._n_clusters:
            raise ValueError(
                f"num_samples ({num_samples}) must be >= n_clusters ({self._n_clusters})"
            )
        feature_t0 = time.perf_counter()
        features = self._to_features(trajectories)
        print(
            "[timing] clustering feature extraction: "
            f"{time.perf_counter() - feature_t0:.3f}s"
        )
        return self._fit_predict(features)

    def cluster_positions(self, positions: np.ndarray) -> np.ndarray:
        """Cluster trajectories directly from SMPL XYZ positions.

        Args:
            positions: ``(num_samples, n_frames, 22, 3)`` global SMPL joint
                positions.

        Returns:
            ``(num_samples,)`` integer labels in ``[0, n_clusters)``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        num_samples, n_frames = positions.shape[:2]
        if num_samples < self._n_clusters:
            raise ValueError(
                f"num_samples ({num_samples}) must be >= n_clusters ({self._n_clusters})"
            )

        feature_t0 = time.perf_counter()
        frame_idx = min(100, n_frames - 1)
        spine3_j = LEFT_ARM_CHAIN_INDICES[0]
        arm_chain = positions[:, frame_idx, LEFT_ARM_CHAIN_INDICES, :]
        features = (arm_chain - positions[:, frame_idx, spine3_j, None, :]).reshape(
            num_samples, -1
        )
        print(
            "[timing] position clustering feature extraction: "
            f"{time.perf_counter() - feature_t0:.3f}s"
        )
        return self._fit_predict(features.astype(np.float64))
