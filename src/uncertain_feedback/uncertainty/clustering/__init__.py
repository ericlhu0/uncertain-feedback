"""Trajectory clustering methods for MDM uncertainty quantification.

Add a new clustering method by subclassing
:class:`~uncertain_feedback.uncertainty.clustering.base.TrajectoryClusterer`
and implementing ``_to_features`` (and optionally overriding ``_fit_predict``
to swap the clustering algorithm).
"""

from uncertain_feedback.uncertainty.clustering.base import TrajectoryClusterer
from uncertain_feedback.uncertainty.clustering.xyz_clusterer import (
    XyzPositionClusterer,
)

__all__ = [
    "TrajectoryClusterer",
    "XyzPositionClusterer",
]
