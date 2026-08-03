"""Full-trajectory path-shape clusterer (arclength resample + PCA)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from uncertain_feedback.planners.mpc.arm_features import resample_equidistant
from uncertain_feedback.planners.mpc.kinematics import LEFT_ARM_CHAIN_INDICES
from uncertain_feedback.uncertainty.clustering.base import (
    TrajectoryClusterer,
    agglomerative_labels,
)

_N_WAYPOINTS = 15


class PathPcaClusterer(TrajectoryClusterer):
    """Cluster trajectories by their full arm path, timing-normalized.

    Each sample's spine3-relative arm-chain positions are resampled to
    ``_N_WAYPOINTS`` points equidistant in arclength, flattened, reduced with
    PCA to 95% variance, and clustered with average-linkage agglomerative.
    """

    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PathPcaClusterer only supports positions.")

    def _positions_to_features(self, positions: np.ndarray) -> np.ndarray:
        spine3_j = LEFT_ARM_CHAIN_INDICES[0]
        rel = (
            positions[:, :, LEFT_ARM_CHAIN_INDICES, :]
            - positions[:, :, spine3_j, None, :]
        )
        paths = np.stack(
            [resample_equidistant(sample, _N_WAYPOINTS) for sample in rel]
        ).reshape(positions.shape[0], -1)
        return PCA(n_components=0.95, svd_solver="full").fit_transform(paths)

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        return agglomerative_labels(features, self._n_clusters)
