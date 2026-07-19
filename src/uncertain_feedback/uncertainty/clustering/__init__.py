"""Trajectory clustering methods for MDM uncertainty quantification.

Add a new clustering method by subclassing
:class:`~uncertain_feedback.uncertainty.clustering.base.TrajectoryClusterer`
and implementing ``_positions_to_features`` (and optionally overriding
``_fit_predict`` to swap the clustering algorithm), then registering it in
``CLUSTERER_BUILDERS``. Builders import their module lazily so importing this
registry never forces the heavy torch/MDM dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer
from uncertain_feedback.uncertainty.clustering.xyz_clusterer import (
    AggloEndPoseClusterer,
    XyzPositionClusterer,
)

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK


def _build_kmeans_end_pose(
    n_clusters: int, fk: "SmplLeftArmFK | None"
) -> TrajectoryClusterer:
    return XyzPositionClusterer(n_clusters, fk=fk)


def _build_agglo_end_pose(
    n_clusters: int, fk: "SmplLeftArmFK | None"
) -> TrajectoryClusterer:
    return AggloEndPoseClusterer(n_clusters, fk=fk)


def _build_agglo_path_pca(
    n_clusters: int, _fk: "SmplLeftArmFK | None"
) -> TrajectoryClusterer:
    from uncertain_feedback.uncertainty.clustering.path_pca_clusterer import (  # pylint: disable=import-outside-toplevel
        PathPcaClusterer,
    )

    return PathPcaClusterer(n_clusters)


def _build_agglo_t2m(
    n_clusters: int, _fk: "SmplLeftArmFK | None"
) -> TrajectoryClusterer:
    from uncertain_feedback.uncertainty.clustering.t2m_clusterer import (  # pylint: disable=import-outside-toplevel
        T2mEmbeddingClusterer,
    )

    return T2mEmbeddingClusterer(n_clusters)


CLUSTERER_BUILDERS: dict[
    str, Callable[[int, "SmplLeftArmFK | None"], TrajectoryClusterer]
] = {
    "kmeans_end_pose": _build_kmeans_end_pose,
    "agglo_end_pose": _build_agglo_end_pose,
    "agglo_path_pca": _build_agglo_path_pca,
    "agglo_t2m": _build_agglo_t2m,
}


def make_clusterer(
    name: str, n_clusters: int, fk: "SmplLeftArmFK | None" = None
) -> TrajectoryClusterer:
    """Construct the trajectory clusterer selected by ``name``.

    ``fk`` is forwarded only to the end-pose clusterers, which use it to avoid
    constructing a fresh :class:`SmplLeftArmFK`.
    """
    if name not in CLUSTERER_BUILDERS:
        raise ValueError(
            f"Unknown clusterer '{name}'. Available: {sorted(CLUSTERER_BUILDERS)}"
        )
    return CLUSTERER_BUILDERS[name](n_clusters, fk)


__all__ = [
    "CLUSTERER_BUILDERS",
    "TrajectoryClusterer",
    "XyzPositionClusterer",
    "AggloEndPoseClusterer",
    "make_clusterer",
]
