"""Tests for the trajectory clusterer registry and its implementations."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

# uncertainty.clustering <-> planners.mpc import cycle: planners.mpc must be
# fully initialized before the clustering package is imported.
import uncertain_feedback.planners.mpc  # noqa: F401  # pylint: disable=unused-import
from uncertain_feedback.uncertainty.clustering import make_clusterer


def _two_group_positions(n_per_group: int = 6, n_frames: int = 30) -> np.ndarray:
    """Two groups whose arm chains move differently relative to spine3."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n_frames)[:, None]
    base = rng.normal(scale=0.01, size=(22, 3))
    arm = [13, 16, 18, 20]  # collar, shoulder, elbow, wrist
    up = np.broadcast_to(base, (n_frames, 22, 3)).copy()
    up[:, arm] += t[:, None] * np.array([0.0, 0.5, 0.0])
    out = np.broadcast_to(base, (n_frames, 22, 3)).copy()
    out[:, arm] += t[:, None] * np.array([0.5, 0.0, 0.2])
    samples = np.concatenate(
        [
            np.repeat(up[None], n_per_group, axis=0),
            np.repeat(out[None], n_per_group, axis=0),
        ]
    )
    return samples + rng.normal(scale=0.005, size=samples.shape)


@pytest.mark.parametrize(
    "name", ["kmeans_end_pose", "agglo_end_pose", "agglo_path_pca"]
)
def test_clusterers_separate_distinct_groups(name: str) -> None:
    positions = _two_group_positions()
    clusterer = make_clusterer(name, 2, fk=object())  # type: ignore[arg-type]
    labels = clusterer.cluster_positions(positions)

    assert labels.shape == (12,)
    assert set(np.unique(labels)) == {0, 1}
    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]

    medoids = clusterer.medoid_indices(labels)
    assert set(medoids) == {0, 1}
    for label, idx in medoids.items():
        assert labels[idx] == label


def test_medoid_is_min_summed_distance_member() -> None:
    positions = _two_group_positions(n_per_group=4)
    clusterer = make_clusterer("kmeans_end_pose", 2, fk=object())  # type: ignore[arg-type]
    labels = clusterer.cluster_positions(positions)
    features = clusterer._position_features
    for label, idx in clusterer.medoid_indices(labels).items():
        members = np.flatnonzero(labels == label)
        sums = [
            np.linalg.norm(features[members] - features[m], axis=1).sum()  # type: ignore[index]
            for m in members
        ]
        assert idx == members[int(np.argmin(sums))]


def test_medoid_indices_requires_prior_clustering() -> None:
    clusterer = make_clusterer("kmeans_end_pose", 2, fk=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cluster_positions"):
        clusterer.medoid_indices(np.zeros(4, dtype=np.intp))


def test_make_clusterer_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown clusterer"):
        make_clusterer("nope", 2)
