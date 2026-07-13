import json
from types import MethodType, SimpleNamespace

import numpy as np

from uncertain_feedback.demo_designer import core
from uncertain_feedback.demo_designer import server as demo_server
from uncertain_feedback.demo_designer.core import (
    DemoSession,
    _rationale_from_artifacts,
)
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.simulated_users import HiddenCostTerm


def test_rationale_from_artifacts_round_trip(tmp_path) -> None:
    payload = {"interpret": {"preference": "raise the elbow"}, "ranking": None}
    (tmp_path / "rationale.json").write_text(json.dumps(payload), encoding="utf-8")

    assert _rationale_from_artifacts(tmp_path) == payload
    assert _rationale_from_artifacts(tmp_path / "missing") is None


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


def test_oracle_rollouts_start_at_initial_and_trigger_poses(monkeypatch) -> None:
    session = DemoSession.__new__(DemoSession)
    session.context = object()
    session.body_pos = np.zeros((22, 3))
    session.spine3_pos = np.zeros(3)
    session.spine3_aa = np.zeros(3)
    session.persona_name = "persona"
    session.start_arm_aa = np.full((3, 3), 9.0)
    session.goal = np.array([0.1, 0.2, 0.3])
    session.q_feedback = np.full((3, 3), 2.0)
    session.q_history = [
        np.full((3, 3), 0.0),
        np.full((3, 3), 1.0),
        session.q_feedback,
    ]
    session.oracle_traj = None
    session.oracle_source = None
    session.get_persona = MethodType(lambda self, name: object(), session)
    session._cfg_with_goal = MethodType(lambda self, goal: object(), session)
    session._extra_costs = MethodType(
        lambda self, user: CompositeTrajectoryCost([]), session
    )
    session.package_trajectory = MethodType(
        lambda self, traj, user: {"values": traj[:, 0, 0].tolist()}, session
    )
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        start = args[1][0, 0]
        result = np.zeros((2, 3, 3))
        result[:, 0, 0] = [start, start + 1]
        return result

    monkeypatch.setattr(core, "rollout_to_goal", fake_rollout)
    monkeypatch.setattr(core, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(core, "goal_reach", lambda *args: {})

    initial = session.run_oracle(from_trigger=False)
    trigger = session.run_oracle(from_trigger=True)

    np.testing.assert_array_equal(calls[0][0][1], session.start_arm_aa)
    np.testing.assert_array_equal(calls[1][0][1], session.q_feedback)
    assert isinstance(calls[0][0][4].terms()[-1], HiddenCostTerm)
    assert initial["source"] == "initial"
    assert initial["trajectory"]["values"] == [9.0, 10.0]
    assert trigger["source"] == "trigger"
    assert trigger["trajectory"]["values"] == [0.0, 1.0, 2.0, 3.0]


def test_oracle_rollout_resumes_from_feedback_pose(monkeypatch) -> None:
    session = DemoSession.__new__(DemoSession)
    session.unified_cost = None
    session.context = object()
    session.body_pos = np.zeros((22, 3))
    session.spine3_pos = np.zeros(3)
    session.spine3_aa = np.zeros(3)
    session.get_persona = MethodType(lambda self, name: object(), session)
    session._cfg_with_goal = MethodType(lambda self, goal: object(), session)
    session._extra_costs = MethodType(
        lambda self, user: CompositeTrajectoryCost([]), session
    )
    session.package_trajectory = MethodType(
        lambda self, traj, user: {"values": traj[:, 0, 0].tolist()}, session
    )

    base = np.zeros((3, 3, 3))
    base[:, 0, 0] = [0.0, 1.0, 2.0]
    continuation = np.zeros((2, 3, 3))
    continuation[:, 0, 0] = [1.0, 3.0]
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        return base if len(calls) == 1 else continuation

    def fake_set_trigger(self, user, traj) -> None:
        self.trigger_step = 1
        self.q_feedback = traj[1]
        self.q_history = [traj[0], traj[1]]

    session._set_trigger = MethodType(fake_set_trigger, session)
    monkeypatch.setattr(core, "rollout_to_goal", fake_rollout)
    monkeypatch.setattr(core, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(core, "goal_reach", lambda *args: {})

    result = session.run_base(
        np.zeros((3, 3)).tolist(),
        [0.1, 0.2, 0.3],
        "persona",
        show_oracle=True,
    )

    np.testing.assert_array_equal(calls[1][0][1], base[1])
    assert isinstance(calls[1][0][4].terms()[-1], HiddenCostTerm)
    assert result["oracle"]["trajectory"]["values"] == [0.0, 1.0, 3.0]


def test_combine_rounds_rolls_unified_cost_from_start(monkeypatch, tmp_path) -> None:
    class CombinedCost:
        description = "combined"
        code = "def cost(): pass"

        def __call__(self, q_trajs):
            return np.zeros(q_trajs.shape[0])

    combined = CombinedCost()

    class FakeCombinator:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def generate(self, install):
            del install
            return combined

    session = DemoSession.__new__(DemoSession)
    session.rounds = [
        SimpleNamespace(feedback_text="first"),
        SimpleNamespace(feedback_text="second"),
    ]
    session.round_records = []
    session.goal = np.array([0.1, 0.2, 0.3])
    session.start_arm_aa = np.ones((3, 3))
    session.persona_name = "persona"
    session.context = object()
    session.body_pos = np.zeros((22, 3))
    session.spine3_pos = np.zeros(3)
    session.spine3_aa = np.zeros(3)
    session.cfg = SimpleNamespace(
        llm_cost=SimpleNamespace(
            use_images=False,
            model=None,
            strict=True,
            codex_cmd="codex",
        )
    )
    eval_state = SimpleNamespace(make_rollout_fn=lambda: None)
    session._latest_round_generation = SimpleNamespace(
        generated_context=object(), summaries={}, images={}, eval_state=eval_state
    )
    session.get_persona = MethodType(lambda self, name: object(), session)
    session._cfg_with_goal = MethodType(lambda self, goal: object(), session)
    session._extra_costs = MethodType(
        lambda self, user: CompositeTrajectoryCost([]), session
    )
    session.package_trajectory = MethodType(
        lambda self, traj, user: {"values": traj[:, 0, 0].tolist()}, session
    )
    session.generated_cost_field = MethodType(lambda self, cost: {}, session)
    rollout = np.zeros((2, 3, 3))
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        return rollout

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "CombineCostGenerator", FakeCombinator)
    monkeypatch.setattr(core, "rollout_to_goal", fake_rollout)
    monkeypatch.setattr(core, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(core, "goal_reach", lambda *args: {})

    result = session.combine_rounds()

    np.testing.assert_array_equal(calls[0][0][1], session.start_arm_aa)
    assert calls[0][1]["progress_label"] == "unified-from-start"
    assert result["trajectory"]["values"] == [0.0, 0.0]
