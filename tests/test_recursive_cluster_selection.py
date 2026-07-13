from types import MethodType

import numpy as np
import pytest

from uncertain_feedback.demo_designer import core
from uncertain_feedback.demo_designer.core import DemoSession
from uncertain_feedback.simulated_users import SimulatedUser
from uncertain_feedback.uncertainty.cluster_picker import (
    _LevelPickResult,
    _navigate_cluster_levels,
)


def test_recursive_navigation_returns_final_global_sample() -> None:
    samples = np.arange(8, dtype=np.float64)[:, None]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.intp)
    actions = iter(
        [
            _LevelPickResult("refine", 0, {0: 0.8, 1: 1.0}),
            _LevelPickResult("refine", 1, {0: 0.8, 1: 0.6}),
            _LevelPickResult("confirm", 1, {0: 0.6, 1: 0.4}),
        ]
    )
    clustered: list[np.ndarray] = []

    def recluster(subset: np.ndarray) -> np.ndarray:
        clustered.append(subset[:, 0].copy())
        return np.repeat(np.arange(2, dtype=np.intp), len(subset) // 2)

    result = _navigate_cluster_levels(
        samples, labels, lambda _level: next(actions), recluster, 1.0
    )

    assert result.root_label == 0
    np.testing.assert_array_equal(result.sample_indices, [3])
    assert result.scale == 0.4
    np.testing.assert_array_equal(clustered[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(clustered[1], [2, 3])


def test_recursive_navigation_back_restores_parent_selection_and_scale() -> None:
    samples = np.arange(8, dtype=np.float64)[:, None]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.intp)
    seen = []
    actions = iter(
        [
            _LevelPickResult("refine", 1, {0: 1.0, 1: 0.7}),
            _LevelPickResult("back", None, {0: 0.7, 1: 0.7}),
            _LevelPickResult("confirm", 1, {0: 1.0, 1: 0.7}),
        ]
    )

    def show(level):
        seen.append(level.selected_label)
        return next(actions)

    result = _navigate_cluster_levels(
        samples,
        labels,
        show,
        lambda subset: np.arange(len(subset), dtype=np.intp) % 2,
        1.0,
    )

    assert seen == [None, None, 1]
    np.testing.assert_array_equal(result.sample_indices, [4, 5, 6, 7])
    assert result.scale == 0.7


class _FakeGenerator:
    def smpl_positions_to_left_arm_trajectory(
        self, positions: np.ndarray, spine3_aa: np.ndarray | None = None
    ) -> np.ndarray:
        del spine3_aa
        return positions[:, :3, :]


class _DeterministicClusterer:
    def __init__(self, n_clusters: int, fk: object) -> None:
        del fk
        self.n_clusters = n_clusters

    def cluster_positions(self, positions: np.ndarray) -> np.ndarray:
        return np.arange(len(positions), dtype=np.intp) % self.n_clusters


def _demo_session() -> DemoSession:
    session = DemoSession.__new__(DemoSession)
    session.samples = np.broadcast_to(
        np.arange(8, dtype=np.float64)[:, None, None, None], (8, 2, 22, 3)
    ).copy()
    session.cluster_levels = []
    session.goal = np.zeros(3)
    session.persona_name = "test"
    session.gen = _FakeGenerator()
    session.fk = object()
    session.context = object()
    session.spine3_aa = np.zeros(3)
    session.q_history = []
    session.labels = None
    session.cluster_means = {}
    session.cluster_corrections = {}
    session.cluster_fulls = {}
    session.chosen_label = None
    session.scaled_correction = None
    session.scale = 1.0
    user = SimulatedUser("test", "", "", bounds=())
    session.get_persona = MethodType(lambda self, _name: user, session)
    session._cfg_with_goal = MethodType(lambda self, _goal: object(), session)
    session.package_trajectory = MethodType(
        lambda self, traj, _user: {"n_frames": len(traj)}, session
    )
    return session


def test_demo_designer_refines_recursively_and_backs_up(monkeypatch) -> None:
    monkeypatch.setattr(core, "XyzPositionClusterer", _DeterministicClusterer)
    monkeypatch.setattr(
        core,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(core, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(core, "goal_reach", lambda *_args: {"reached": True})
    session = _demo_session()

    root = session.recluster(2, 0.9)
    assert root["depth"] == 0
    assert root["active_sample_count"] == 8
    session.pick_cluster(0)
    child = session.refine_cluster(0, 2, 0.9)
    assert child["depth"] == 1
    assert child["path"] == [0]
    assert child["active_sample_count"] == 4

    session.pick_cluster(0)
    grandchild = session.refine_cluster(0, 2, 0.5)
    assert grandchild["depth"] == 2
    assert grandchild["active_sample_count"] == 2
    with pytest.raises(ValueError, match="enough selected samples"):
        session.refine_cluster(0, 2, 1.0)

    parent = session.back_cluster()
    assert parent["depth"] == 1
    assert parent["selected_label"] == 0
    assert parent["scale"] == 0.9
    assert session.scaled_correction is not None
    root_again = session.back_cluster()
    assert root_again["depth"] == 0
    assert root_again["selected_label"] == 0
    assert root_again["scale"] == 0.9


def test_demo_designer_reclusters_only_current_subset(monkeypatch) -> None:
    monkeypatch.setattr(core, "XyzPositionClusterer", _DeterministicClusterer)
    monkeypatch.setattr(
        core,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(core, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(core, "goal_reach", lambda *_args: {"reached": True})
    session = _demo_session()
    session.recluster(2, 1.0)
    session.pick_cluster(1)
    session.refine_cluster(1, 2, 1.0)

    payload = session.recluster(2, 1.0)

    assert payload["depth"] == 1
    assert payload["active_sample_count"] == 4
    assert payload["selected_label"] is None
