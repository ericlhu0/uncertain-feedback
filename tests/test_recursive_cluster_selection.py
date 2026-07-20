"""Tests for recursive UQ cluster refinement and selection."""

# pylint: disable=missing-function-docstring

import json
from types import SimpleNamespace

import numpy as np
import pytest

from uncertain_feedback.demo_runner import session as demo_session
from uncertain_feedback.demo_runner.session import Session
from uncertain_feedback.experiments.experiment_pipeline import (
    _rejected_candidate_trajs,
)
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
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


def test_recursive_navigation_splits_small_subset_into_singletons() -> None:
    samples = np.arange(4, dtype=np.float64)[:, None]
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    actions = iter(
        [
            _LevelPickResult("refine", 0, {0: 1.0, 1: 1.0}),
            _LevelPickResult("confirm", 1, {0: 1.0, 1: 1.0}),
        ]
    )

    def recluster(_subset: np.ndarray) -> np.ndarray:
        raise AssertionError("small subsets should bypass the clusterer")

    result = _navigate_cluster_levels(
        samples,
        labels,
        lambda _level: next(actions),
        recluster,
        1.0,
        n_clusters=3,
    )

    np.testing.assert_array_equal(result.sample_indices, [1])


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

    def medoid_indices(self, labels: np.ndarray) -> dict[int, int]:
        return {
            int(label): int(np.flatnonzero(labels == label)[0])
            for label in np.unique(labels)
        }


def _fake_make_clusterer(
    _name: str, n_clusters: int, fk: object = None
) -> _DeterministicClusterer:
    return _DeterministicClusterer(n_clusters, fk)


def _demo_session(tmp_path) -> Session:
    trajectory = SimpleNamespace()
    trajectory.samples = np.broadcast_to(
        np.arange(8, dtype=np.float64)[:, None, None, None], (8, 2, 22, 3)
    ).copy()
    trajectory.cluster_levels = []
    trajectory.goal = np.zeros(3)
    trajectory.q_history = []
    trajectory.labels = None
    trajectory.cluster_means = {}
    trajectory.cluster_corrections = {}
    trajectory.cluster_fulls = {}
    trajectory.chosen_label = None
    trajectory.scaled_correction = None
    trajectory.scale = 1.0
    trajectory.prompt = None
    trajectory._last_cost_payload = None
    user = SimulatedUser("test", "", "", bounds=())
    fk = SmplLeftArmFK()
    rig = SimpleNamespace(
        gen=_FakeGenerator(),
        fk=fk,
        context=MpcCostContext(
            fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
        ),
        spine3_aa=np.zeros(3),
        _cfg_with_goal=lambda goal: object(),
        package_trajectory=lambda traj, selected: {"n_frames": len(traj)},
    )
    session = Session.__new__(Session)
    session.rig = rig  # type: ignore[assignment]
    session.user = user
    session.trajectory = trajectory  # type: ignore[assignment]
    session.dir = tmp_path
    session.persona_name = user.name
    session.started = ""
    session.beats = []
    return session


def test_demo_runner_refines_recursively_and_backs_up(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_session, "make_clusterer", _fake_make_clusterer)
    monkeypatch.setattr(
        demo_session,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *_args: {"reached": True})
    session = _demo_session(tmp_path)

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
    leaf = session.refine_cluster(0, 2, 1.0)
    assert leaf["depth"] == 3
    assert leaf["active_sample_count"] == 1
    assert [cluster["count"] for cluster in leaf["clusters"]] == [1]
    assert leaf["clusters"][0]["can_refine"] is True
    session.back_cluster()

    parent = session.back_cluster()
    assert parent["depth"] == 1
    assert parent["selected_label"] == 0
    assert parent["scale"] == 0.9
    assert session.trajectory.scaled_correction is not None  # type: ignore[union-attr]
    root_again = session.back_cluster()
    assert root_again["depth"] == 0
    assert root_again["selected_label"] == 0
    assert root_again["scale"] == 0.9


def test_demo_runner_reclusters_only_current_subset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_session, "make_clusterer", _fake_make_clusterer)
    monkeypatch.setattr(
        demo_session,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *_args: {"reached": True})
    session = _demo_session(tmp_path)
    session.recluster(2, 1.0)
    session.pick_cluster(1)
    session.refine_cluster(1, 2, 1.0)

    payload = session.recluster(2, 1.0)

    assert payload["depth"] == 1
    assert payload["active_sample_count"] == 4
    assert payload["selected_label"] is None


def test_replay_records_only_the_final_cluster_selection_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(demo_session, "make_clusterer", _fake_make_clusterer)
    monkeypatch.setattr(
        demo_session,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *_args: {"reached": True})
    session = _demo_session(tmp_path)

    session.recluster(2, 1.0)
    session.pick_cluster(1)
    session.recluster(2, 0.5)
    session.pick_cluster(0)
    session.refine_cluster(0, 2, 0.75)
    session.pick_cluster(1)

    assert session.beats == []

    session._record_final_feedback()

    assert [beat["kind"] for beat in session.beats] == [
        "clusters",
        "pick",
        "clusters",
        "pick",
    ]
    replay_dir = tmp_path / "replay"
    recorded = [
        json.loads((replay_dir / beat["file"]).read_text(encoding="utf-8"))
        for beat in session.beats
    ]
    assert recorded[0]["data"]["scale"] == 0.5
    assert recorded[1]["data"] == {"label": 0}
    assert recorded[3]["data"] == {"label": 1}


def test_explicit_rejected_candidate_split() -> None:
    candidates = {
        3: np.full((1, 1), 3.0),
        1: np.full((1, 1), 1.0),
        2: np.full((1, 1), 2.0),
    }

    rejected = _rejected_candidate_trajs(candidates, 2, frozenset({3, 1}))

    assert [float(trajectory[0, 0]) for trajectory in rejected] == [1.0, 3.0]
    assert (  # pylint: disable=use-implicit-booleaness-not-comparison
        _rejected_candidate_trajs(candidates, 2, frozenset()) == ()
    )
    with pytest.raises(ValueError, match="cannot be undesirable"):
        _rejected_candidate_trajs(candidates, 2, frozenset({2}))


def test_demo_runner_marks_restore_across_cluster_levels(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_session, "make_clusterer", _fake_make_clusterer)
    monkeypatch.setattr(
        demo_session,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *_args: {"reached": True})
    session = _demo_session(tmp_path)

    root = session.recluster(2, 1.0)
    assert root["undesirable_labels"] == []
    assert session.mark_cluster(1, True)["undesirable_labels"] == [1]
    session.pick_cluster(1)
    assert session.trajectory.cluster_levels[-1].undesirable_labels == {1}  # type: ignore[union-attr]
    assert session.mark_cluster(1, True)["undesirable_labels"] == [1]
    assert session.mark_cluster(1, False)["undesirable_labels"] == []

    session.pick_cluster(0)
    session.mark_cluster(1, True)
    child = session.refine_cluster(0, 2, 1.0)
    assert child["undesirable_labels"] == []
    session.mark_cluster(1, True)
    parent = session.back_cluster()
    assert parent["undesirable_labels"] == [1]


def test_demo_runner_threads_scaled_clusters_to_cost_generation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(demo_session, "make_clusterer", _fake_make_clusterer)
    monkeypatch.setattr(
        demo_session,
        "oracle_cluster_scores",
        lambda _user, _context, means, _scale: {label: float(label) for label in means},
    )
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *_args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *_args: {"reached": True})
    captured: dict[str, object] = {}

    def fake_generate_cost_for_cluster(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(generated_cost=None)

    monkeypatch.setattr(
        demo_session, "generate_cost_for_cluster", fake_generate_cost_for_cluster
    )
    session = _demo_session(tmp_path)
    session.corpus = SimpleNamespace(dir=tmp_path)  # type: ignore[assignment]
    session.rig._extra_costs = lambda _user: object()  # type: ignore[assignment, method-assign, return-value]
    session.rig.body_pos = None  # type: ignore[assignment]
    session.rig.spine3_pos = None  # type: ignore[assignment]
    session.trajectory.base_traj = np.zeros((1, 3, 3))  # type: ignore[union-attr]
    session.trajectory.start_q = np.zeros((3, 3))  # type: ignore[union-attr]
    session.trajectory.q_feedback = None  # type: ignore[union-attr]
    session.trajectory.prompt = "move comfortably"  # type: ignore[union-attr]
    session.trajectory.samples[:, 1] += 10.0  # type: ignore[index, union-attr]
    session.recluster(2, 0.4)
    session.pick_cluster(0)
    session.mark_cluster(1, True)

    with pytest.raises(ValueError, match="produced no cost"):
        session.generate_cost("llm")

    assert captured["undesirable_labels"] == frozenset({1})
    candidate_trajs = captured["candidate_trajs"]
    assert isinstance(candidate_trajs, dict)
    for label, correction in session.trajectory.cluster_corrections.items():  # type: ignore[union-attr]
        np.testing.assert_allclose(candidate_trajs[label], correction)
        assert not np.array_equal(
            candidate_trajs[label], session.trajectory.cluster_means[label]  # type: ignore[union-attr]
        )
    np.testing.assert_allclose(captured["cluster_traj"], candidate_trajs[0])  # type: ignore[call-overload]
