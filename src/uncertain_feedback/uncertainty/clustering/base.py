"""Template-method base class for trajectory clusterers.

A clustering method varies along two independent axes:

* **Feature representation** — how a trajectory batch is turned into a
  ``(num_samples, n_features)`` matrix.  Implement :meth:`_to_features`.
* **Clustering algorithm** — how that matrix is partitioned into labels.
  Override :meth:`_fit_predict` (defaults to KMeans).

Subclasses therefore implement only what is distinct; the shared scaffolding
(input validation, the default KMeans fit, timing prints, and the
:meth:`cluster` template) lives here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans


def agglomerative_labels(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """Partition a feature matrix with average-linkage agglomerative clustering.

    Args:
        features: ``(num_samples, n_features)`` matrix.

    Returns:
        ``(num_samples,)`` integer labels in ``[0, n_clusters)``.
    """
    agglo = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average", metric="euclidean"
    )
    agglo_t0 = time.perf_counter()
    labels = agglo.fit_predict(features).astype(np.intp)
    print(
        "[timing] Agglomerative fit_predict: "
        f"{time.perf_counter() - agglo_t0:.3f}s"
    )
    return labels


class TrajectoryClusterer(ABC):
    """Cluster a batch of arm trajectories into integer labels.

    Subclasses implement :meth:`_to_features` to map a controlled
    ``(num_samples, n_frames, 3, 3)`` axis-angle trajectory batch to a
    ``(num_samples, n_features)`` matrix; :meth:`cluster` is a concrete
    template that validates, extracts features, and fits.  Legacy full-arm
    ``(..., 4, 3)`` batches may also be accepted by concrete implementations.

    Args:
        n_clusters:   Number of clusters (K in KMeans).
        random_state: Random seed forwarded to KMeans for reproducibility.
    """

    def __init__(self, n_clusters: int, random_state: int = 0) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._position_features: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        """Return the configured number of clusters."""
        return self._n_clusters

    @property
    def supports_positions(self) -> bool:
        """Whether this clusterer implements the position-batch path."""
        return (
            type(self)._positions_to_features
            is not TrajectoryClusterer._positions_to_features
            or type(self).cluster_positions
            is not TrajectoryClusterer.cluster_positions
        )

    @abstractmethod
    def _to_features(self, trajectories: np.ndarray) -> np.ndarray:
        """Convert a trajectory batch to a ``(num_samples, n_features)`` matrix.

        Args:
            trajectories: ``(num_samples, n_frames, 3, 3)`` axis-angle batch.

        Returns:
            ``(num_samples, n_features)`` float feature matrix.
        """

    def _positions_to_features(self, positions: np.ndarray) -> np.ndarray:
        """Convert a position batch to a ``(num_samples, n_features)`` matrix.

        Args:
            positions: ``(num_samples, n_frames, 22, 3)`` global SMPL joint
                positions.

        Returns:
            ``(num_samples, n_features)`` float feature matrix.
        """
        raise NotImplementedError

    def _fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Partition a feature matrix into integer labels.

        Default implementation is KMeans.  Override to swap in a different
        clustering algorithm (DBSCAN, agglomerative, …).

        Args:
            features: ``(num_samples, n_features)`` matrix.

        Returns:
            ``(num_samples,)`` integer labels in ``[0, n_clusters)``.
        """
        kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=10,
        )
        kmeans_t0 = time.perf_counter()
        labels = kmeans.fit_predict(features).astype(np.intp)
        print(f"[timing] KMeans fit_predict: {time.perf_counter() - kmeans_t0:.3f}s")
        return labels

    def _validate_num_samples(self, num_samples: int) -> None:
        """Raise if there are fewer samples than requested clusters."""
        if num_samples < self._n_clusters:
            raise ValueError(
                f"num_samples ({num_samples}) must be >= n_clusters "
                f"({self._n_clusters})"
            )

    def cluster(self, trajectories: np.ndarray) -> np.ndarray:
        """Assign integer cluster labels to a batch of trajectories.

        Args:
            trajectories: ``(num_samples, n_frames, 3, 3)`` axis-angle batch,
                as returned by
                :meth:`~uncertain_feedback.motion_generators.mdm.mdm_api\
.MdmMotionGenerator.generate_left_arm_trajectory` with ``num_samples > 1``.

        Returns:
            ``(num_samples,)`` integer cluster labels in ``[0, n_clusters)``.

        Raises:
            ValueError: If ``num_samples < n_clusters``.
        """
        trajectories = np.asarray(trajectories, dtype=np.float64)
        self._validate_num_samples(trajectories.shape[0])
        feature_t0 = time.perf_counter()
        features = self._to_features(trajectories)
        print(
            "[timing] clustering feature extraction: "
            f"{time.perf_counter() - feature_t0:.3f}s"
        )
        return self._fit_predict(features)

    def cluster_positions(self, positions: np.ndarray) -> np.ndarray:
        """Assign integer cluster labels from SMPL XYZ positions.

        Args:
            positions: ``(num_samples, n_frames, 22, 3)`` global SMPL joint
                positions.

        Returns:
            ``(num_samples,)`` integer labels in ``[0, n_clusters)``.

        Raises:
            ValueError: If ``num_samples < n_clusters``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        self._validate_num_samples(positions.shape[0])
        feature_t0 = time.perf_counter()
        features = self._positions_to_features(positions).astype(np.float64)
        print(
            "[timing] position clustering feature extraction: "
            f"{time.perf_counter() - feature_t0:.3f}s"
        )
        self._position_features = features
        return self._fit_predict(features)

    def medoid_indices(self, labels: np.ndarray) -> dict[int, int]:
        """Return per-cluster medoid sample indices in the clusterer's feature space.

        Uses the features cached by the most recent :meth:`cluster_positions`
        call, so labels must come from that same call.

        Args:
            labels: ``(num_samples,)`` integer labels from
                :meth:`cluster_positions`.

        Returns:
            Mapping from cluster label to the index (into the clustered batch)
            of the member minimizing summed distance to its cluster.
        """
        if self._position_features is None:
            raise ValueError("Call cluster_positions before medoid_indices.")
        medoids: dict[int, int] = {}
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            members = self._position_features[idx]
            dists = np.linalg.norm(members[:, None] - members[None, :], axis=-1)
            medoids[int(label)] = int(idx[dists.sum(axis=1).argmin()])
        return medoids
