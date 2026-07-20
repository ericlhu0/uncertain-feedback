"""Tests for demo-runner oracle rollouts, cost fields, and round combining."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import cast

import numpy as np

from uncertain_feedback.demo_runner import server as demo_server
from uncertain_feedback.demo_runner import session as demo_session
from uncertain_feedback.demo_runner.core import DemoRig
from uncertain_feedback.demo_runner.session import (
    Session,
    _generated_bounds_from_artifacts,
    _rationale_from_artifacts,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    CostRound,
    MpcCostContext,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import HiddenCostTerm, SimulatedUser


def test_rationale_from_artifacts_round_trip(tmp_path) -> None:
    payload = {"interpret": {"preference": "raise the elbow"}, "ranking": None}
    (tmp_path / "rationale.json").write_text(json.dumps(payload), encoding="utf-8")

    assert _rationale_from_artifacts(tmp_path) == payload
    assert _rationale_from_artifacts(tmp_path / "missing") is None


def test_generated_bounds_use_structured_rationale_grounding(tmp_path) -> None:
    rationale = {
        "ground": {
            "terms": [
                {
                    "feature": "elbow_flexion",
                    "bound_type": "upper_bound",
                    "values": {
                        "cond_feature": "shoulder_elevation",
                        "points": [[1.2, 1.75], [1.38, 1.7], [1.42, 1.65]],
                    },
                }
            ]
        }
    }
    (tmp_path / "rationale.json").write_text(json.dumps(rationale), encoding="utf-8")

    assert _generated_bounds_from_artifacts(tmp_path) == [
        {
            "feature": "elbow_flexion",
            "bound_type": "upper_bound",
            "coupled": True,
            "cond_feature": "shoulder_elevation",
            "points": [[1.2, 1.75], [1.38, 1.7], [1.42, 1.65]],
        }
    ]


def test_generated_bounds_skip_invalid_structured_terms(tmp_path) -> None:
    rationale = {
        "ground": {
            "terms": [
                {
                    "feature": "elbow_flexion",
                    "bound_type": "upper_bound",
                    "values": {
                        "cond_feature": "shoulder_elevation",
                        "points": [[1.4, 1.7], [1.2, 1.6]],
                    },
                },
                {
                    "feature": "not_a_feature",
                    "bound_type": "lower_bound",
                    "values": {"threshold": 1.0},
                },
            ]
        }
    }
    (tmp_path / "rationale.json").write_text(json.dumps(rationale), encoding="utf-8")

    assert (  # pylint: disable=use-implicit-booleaness-not-comparison
        _generated_bounds_from_artifacts(tmp_path) == []
    )


def test_generated_cost_field_identifies_single_named_feature(
    monkeypatch, tmp_path
) -> None:
    class SingleFeatureCost:
        """Generated cost depending on exactly one named feature."""

        code = """def cost(q_trajs, context, params):
    elbow = context.elbow_flexion_angles(q_trajs[:, 1:])
    return np.sum(elbow ** 2, axis=1)
"""

        def __call__(self, q_trajs):
            return np.arange(q_trajs.shape[0], dtype=np.float64)

    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    session = Session(
        rig=SimpleNamespace(context=context),  # type: ignore[arg-type]
        persona_name="persona",
        user=SimpleNamespace(),  # type: ignore[arg-type]
        dir=tmp_path,
        corpus=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session.trajectory = SimpleNamespace(  # type: ignore[assignment]
        base_traj=np.zeros((2, 3, 3)),
        cluster_fulls={},
        cluster_corrections={},
    )
    monkeypatch.setattr(
        demo_session,
        "arm_feature_series",
        lambda poses, context: {"elbow_flexion": np.linspace(0.0, 1.0, poses.shape[0])},
    )

    field = Session.generated_cost_field(session, SingleFeatureCost())  # type: ignore[arg-type]

    assert field["active_features"] == ["elbow_flexion"]
    assert len(field["features"]["elbow_flexion"]) == len(field["penalty"])


def test_demo_trajectory_payload_uses_canonical_shoulder_twist() -> None:
    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    rig = SimpleNamespace(
        fk=fk,
        context=context,
        spine3_pos=context.spine3_pos,
        spine3_aa=context.spine3_aa,
        meshes=SimpleNamespace(register=lambda *_args, **_kwargs: "mesh"),
    )
    axis = fk.tpose_joints[3] - fk.tpose_joints[2]
    axis = axis / np.linalg.norm(axis)
    trajectory = np.zeros((2, 7), dtype=np.float64)
    trajectory[-1, 3:6] = axis * 0.5
    trajectory[-1, :3] = axis * -0.8

    payload = DemoRig.package_trajectory(
        cast(DemoRig, rig), trajectory, SimulatedUser("test", "", "", bounds=())
    )

    np.testing.assert_allclose(
        payload["features"]["shoulder_internal_external_rotation"], [0.0, 0.5]
    )
    assert "shoulder_elevation" in payload["features"]


def test_artifact_route_serves_only_artifact_root(monkeypatch, tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    cost_dir = artifact_root / "run"
    cost_dir.mkdir(parents=True)
    (cost_dir / "rationale.json").write_text('{"ok": true}', encoding="utf-8")
    (tmp_path / "outside.json").write_text('{"secret": true}', encoding="utf-8")
    monkeypatch.setattr(demo_server, "_ARTIFACT_ROOT", artifact_root)
    client = demo_server.app.test_client()

    response = client.get("/api/artifact/run/rationale.json")

    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert client.get("/api/artifact/run/missing.json").status_code == 404
    assert client.get("/api/artifact/%2e%2e/outside.json").status_code == 404


def test_manual_trajectory_requires_session(monkeypatch) -> None:
    monkeypatch.setattr(demo_server, "rig", SimpleNamespace(session=None))
    client = demo_server.app.test_client()

    response = client.post(
        "/api/manual_trajectory/start",
        json={"arm_aa": np.zeros((3, 3)).tolist(), "goal": [0.1, 0.2, 0.3]},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "No active session. Start or resume a session first."
    }


def test_oracle_rollouts_start_at_initial_and_trigger_poses(
    monkeypatch, tmp_path
) -> None:
    user = SimulatedUser("persona", "", "", bounds=())
    rig = SimpleNamespace(
        context=object(),
        body_pos=np.zeros((22, 3)),
        spine3_pos=np.zeros(3),
        spine3_aa=np.zeros(3),
        _cfg_with_goal=lambda goal: object(),
        _extra_costs=lambda selected: CompositeTrajectoryCost([]),
        package_trajectory=lambda traj, selected, pin_mesh=False: {
            "values": traj[:, 0, 0].tolist(),
            "mesh_id": "mesh",
        },
        meshes=SimpleNamespace(
            unpin=lambda mesh_id: unpinned.append(  # pylint: disable=unnecessary-lambda
                mesh_id
            )
        ),
    )
    unpinned: list[str] = []
    q_feedback = np.full((3, 3), 2.0)
    trajectory = SimpleNamespace(
        start_q=np.full((3, 3), 9.0),
        goal=np.array([0.1, 0.2, 0.3]),
        q_feedback=q_feedback,
        q_history=[
            np.full((3, 3), 0.0),
            np.full((3, 3), 1.0),
            q_feedback,
        ],
        oracle_traj=None,
        oracle_source=None,
        oracle_package=None,
    )
    session = Session.__new__(Session)
    session.rig = rig  # type: ignore[assignment]
    session.user = user
    session.trajectory = trajectory  # type: ignore[assignment]
    session.dir = tmp_path
    session.persona_name = user.name
    session.started = ""
    session.beats = []
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        start = args[1][0, 0]
        result = np.zeros((2, 3, 3))
        result[:, 0, 0] = [start, start + 1]
        return result

    monkeypatch.setattr(demo_session, "rollout_to_goal", fake_rollout)
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *args: {})

    initial = session.run_oracle(from_trigger=False)  # pylint: disable=not-callable
    trigger = session.run_oracle(from_trigger=True)  # pylint: disable=not-callable

    np.testing.assert_array_equal(calls[0][0][1], trajectory.start_q)
    np.testing.assert_array_equal(calls[1][0][1], trajectory.q_feedback)
    assert isinstance(calls[0][0][4].terms()[-1], HiddenCostTerm)
    assert initial["source"] == "initial"
    assert initial["trajectory"]["values"] == [9.0, 10.0]
    assert trigger["source"] == "trigger"
    assert trigger["trajectory"]["values"] == [0.0, 1.0, 2.0, 3.0]


def test_combine_rounds_uses_last_persisted_round_and_rolls_from_start(
    monkeypatch, tmp_path
) -> None:
    class CombinedCost:
        """Marker cost standing in for the combined result."""

        description = "combined"
        code = "def cost(): pass"
        params: dict[str, float] = {}

        def __call__(self, q_trajs):
            return np.zeros(q_trajs.shape[0])

    combined = CombinedCost()
    constructor_args = {}

    class FakeCombinator:
        """Combinator stand-in returning the marker combined cost."""

        def __init__(self, **kwargs) -> None:
            constructor_args.update(kwargs)

        def generate(self, install):
            assert install is False
            return combined

    eval_state = SimpleNamespace(
        make_generated_context=lambda: "persisted context",
        make_rollout_fn=lambda: "persisted rollout",
    )

    class FakeEvalState:
        """EvalState stand-in returning canned context and rollout hooks."""

        @classmethod
        def load(cls, path):
            assert path == tmp_path / "second.pkl"
            return eval_state

    rounds = [
        SimpleNamespace(
            feedback_text="first",
            state_path=tmp_path / "first.pkl",
            summaries={"round": 1},
            image_paths=(),
        ),
        SimpleNamespace(
            feedback_text="second",
            state_path=tmp_path / "second.pkl",
            summaries={"round": 2},
            image_paths=(tmp_path / "second.png",),
        ),
    ]
    planner = SimpleNamespace(set_extra_costs=lambda costs: None)
    trajectory = SimpleNamespace(
        goal=np.array([0.1, 0.2, 0.3]),
        start_q=np.ones((3, 3)),
        mpc=planner,
        base_traj=np.zeros((1, 3, 3)),
        cluster_fulls={},
        cluster_corrections={},
    )
    rig = SimpleNamespace(
        cfg=SimpleNamespace(
            llm_cost=SimpleNamespace(
                use_images=False,
                model=None,
                strict=True,
                codex_cmd="codex",
            )
        ),
        context=object(),
        body_pos=np.zeros((22, 3)),
        spine3_pos=np.zeros(3),
        spine3_aa=np.zeros(3),
        _cfg_with_goal=lambda goal: object(),
        _extra_costs=lambda user: CompositeTrajectoryCost([]),
        package_trajectory=lambda traj, user: {"values": traj[:, 0, 0].tolist()},
    )
    session = Session.__new__(Session)
    session.rig = rig  # type: ignore[assignment]
    session.user = object()  # type: ignore[assignment]
    session.rounds = rounds  # type: ignore[assignment]
    session.round_records = []
    session.trajectory = trajectory  # type: ignore[assignment]
    session.dir = tmp_path
    session.corpus = SimpleNamespace(  # type: ignore[assignment]
        dir=tmp_path / "trajectory_corpus", entries=lambda: []
    )
    session._save = MethodType(lambda self: None, session)  # type: ignore[method-assign]
    session.generated_cost_field = MethodType(lambda self, cost: {}, session)  # type: ignore[method-assign]
    rollout = np.zeros((2, 3, 3))
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        return rollout

    monkeypatch.setattr(demo_session, "CombineCostGenerator", FakeCombinator)
    monkeypatch.setattr(demo_session, "EvalState", FakeEvalState)
    monkeypatch.setattr(demo_session, "rollout_to_goal", fake_rollout)
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *args: {})

    result = session.combine_rounds()

    assert constructor_args["context"] == "persisted context"
    assert constructor_args["rollout_fn"] == "persisted rollout"
    assert constructor_args["summaries"] == {"round": 2}
    assert constructor_args["images"] == {"second.png": tmp_path / "second.png"}
    np.testing.assert_array_equal(calls[0][0][1], trajectory.start_q)
    assert calls[0][1]["progress_label"] == "unified-from-start"
    assert result["trajectory"]["values"] == [0.0, 0.0]


def test_combine_rounds_between_trajectories_skips_the_rollout(
    monkeypatch, tmp_path
) -> None:
    class CombinedCost:
        """Marker cost standing in for the combined result."""

        description = "combined"
        code = "def cost(): pass"
        params: dict[str, float] = {}

    combined = CombinedCost()

    class FakeCombinator:
        """Combinator stand-in returning the marker combined cost."""

        def __init__(self, **kwargs) -> None:
            pass

        def generate(self, install):
            del install
            return combined

    class FakeEvalState:
        """EvalState stand-in returning canned context and rollout hooks."""

        @classmethod
        def load(cls, path):
            del path
            return SimpleNamespace(
                make_generated_context=lambda: "persisted context",
                make_rollout_fn=lambda: "persisted rollout",
            )

    def fail_rollout(*args, **kwargs):
        raise AssertionError("no trajectory: nothing to roll out on")

    monkeypatch.setattr(demo_session, "CombineCostGenerator", FakeCombinator)
    monkeypatch.setattr(demo_session, "EvalState", FakeEvalState)
    monkeypatch.setattr(demo_session, "rollout_to_goal", fail_rollout)

    session = Session.__new__(Session)
    session.rig = SimpleNamespace(  # type: ignore[assignment]
        cfg=SimpleNamespace(
            llm_cost=SimpleNamespace(
                use_images=False, model=None, strict=True, codex_cmd="codex"
            )
        )
    )
    session.rounds = [
        SimpleNamespace(  # type: ignore[misc]
            feedback_text=text,
            state_path=tmp_path / "s.pkl",
            summaries={},
            image_paths=(),
        )
        for text in ("first", "second")
    ]
    session.round_records = []
    session.trajectory = None
    session.dir = tmp_path
    session.corpus = SimpleNamespace(  # type: ignore[assignment]
        dir=tmp_path / "trajectory_corpus", entries=lambda: []
    )
    session._save = MethodType(lambda self: None, session)  # type: ignore[method-assign]
    session.generated_cost_field = MethodType(lambda self, cost: {}, session)  # type: ignore[method-assign]

    result = session.combine_rounds()

    assert cast(object, session.unified_cost) is combined
    assert result["description"] == "combined"
    assert "trajectory" not in result
    assert "goal_reach" not in result


def test_session_load_recompiles_round_and_unified_costs(monkeypatch, tmp_path) -> None:
    code = """def cost(q_trajs, context, params):
    return q_trajs[:, 0, 0, 0] * 0.0 + params[\"weight\"]
"""
    session_dir = tmp_path / "20260713_120000_session_persona"
    corpus_dir = session_dir / "trajectory_corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text("[]", encoding="utf-8")
    cost_round = CostRound(
        index=0,
        goal=(0.1, 0.2, 0.3),
        feedback_text="feedback",
        trigger_step=2,
        round_dir=session_dir / "round",
        state_path=session_dir / "round" / "state.pkl",
        cost_code=code,
        params={"weight": 2.0},
        summaries={},
        image_paths=(),
        description="round cost",
        explanation="why",
        interpretation="preference",
        grounding="threshold",
    )
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "persona": "persona",
                "started": "2026-07-13 12:00:00",
                "trajectory_count": 2,
                "rounds": [cost_round.to_json()],
                "unified": {
                    "code": code,
                    "params": {"weight": 3.0},
                    "description": "unified cost",
                },
            }
        ),
        encoding="utf-8",
    )
    user = SimulatedUser("persona", "", "", bounds=())
    rig = SimpleNamespace(context=object(), get_persona=lambda name: user)
    fk = SmplLeftArmFK()
    generated_context = build_generated_cost_context(
        MpcCostContext(fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)),
        current_q=np.zeros(7),
        mdm_traj=np.zeros((1, 7)),
        q_history=[],
        window=1,
    )

    class FakeEvalState:
        """EvalState stand-in returning canned context and rollout hooks."""

        @classmethod
        def load(cls, path):
            assert path == cost_round.state_path
            return SimpleNamespace(make_generated_context=lambda: generated_context)

    monkeypatch.setattr(demo_session, "EvalState", FakeEvalState)

    loaded = Session.load(rig, session_dir)  # type: ignore[arg-type]

    assert loaded.trajectory is None
    assert loaded.trajectory_count == 2
    assert loaded.rounds[0].grounding == "threshold"
    q_trajs = np.zeros((2, 1, 3, 3))
    np.testing.assert_array_equal(loaded._round_costs[0](q_trajs), [2.0, 2.0])
    np.testing.assert_array_equal(loaded.unified_cost(q_trajs), [3.0, 3.0])  # type: ignore[misc]


def test_list_sessions_is_newest_first_with_context_counts(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "demo_runner_artifacts"
    for name, round_count, corpus_count in [
        ("20260713_120000_session_old", 1, 2),
        ("20260713_130000_session_new", 3, 4),
    ]:
        path = root / name
        corpus_dir = path / "trajectory_corpus"
        corpus_dir.mkdir(parents=True)
        (path / "session.json").write_text(
            json.dumps(
                {
                    "persona": "persona",
                    "started": name[:15],
                    "trajectory_count": 2,
                    "rounds": [{}] * round_count,
                }
            ),
            encoding="utf-8",
        )
        (corpus_dir / "manifest.json").write_text(
            json.dumps([{}] * corpus_count), encoding="utf-8"
        )
    elsewhere = tmp_path / "changed_working_directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rig = DemoRig.__new__(DemoRig)
    rig.artifact_root = root

    sessions = rig.list_sessions()

    assert [Path(item["dir"]).name for item in sessions] == [
        "20260713_130000_session_new",
        "20260713_120000_session_old",
    ]
    assert sessions[0]["round_count"] == 3
    assert sessions[0]["corpus_count"] == 4
    assert sessions[0]["name"] == "20260713_130000_session_new"


def test_delete_session_removes_directory_and_clears_active(tmp_path) -> None:
    root = tmp_path / "demo_runner_artifacts"
    session_dir = root / "20260713_130000_session_persona"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    rig = DemoRig.__new__(DemoRig)
    rig.artifact_root = root
    rig.session = SimpleNamespace(dir=session_dir)  # type: ignore[assignment]

    result = rig.delete_session(session_dir.name)

    assert not session_dir.exists()
    assert rig.session is None
    assert result == {"sessions": [], "active_deleted": True}


def test_delete_session_rejects_paths_outside_artifact_root(tmp_path) -> None:
    root = tmp_path / "demo_runner_artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "session.json").write_text("{}", encoding="utf-8")
    rig = DemoRig.__new__(DemoRig)
    rig.artifact_root = root
    rig.session = None

    with np.testing.assert_raises_regex(ValueError, "Unknown session"):
        rig.delete_session("../outside")
