import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from uncertain_feedback.demo_designer.core import DemoRig
from uncertain_feedback.demo_designer import session as demo_session
from uncertain_feedback.demo_designer.session import ClusterLevel, Session
from uncertain_feedback.experiments.trajectory_corpus import TrajectoryCorpus
from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    CostRound,
    MpcCostContext,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import HiddenBound, SimulatedUser


class FakeCost:
    description = "cost"
    code = "def cost(): pass"
    params: dict[str, float] = {}

    def __call__(self, q_trajs):
        return np.zeros(q_trajs.shape[0])


class FakePlanner:
    mdm_ready_to_terminate = True
    mdm_tracking_complete = True

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.pushed = None
        self.costs = None

    def step(self, q):
        return np.asarray(q) + 1.0

    def goal_reached(self, q):
        del q
        return False

    def set_mdm_goal(self, goal):
        self.mdm_goal = goal

    def push_trajectory(self, trajectory):
        self.pushed = trajectory

    def set_extra_costs(self, costs):
        self.costs = costs


def test_mdm_locked_seed_resets_before_each_generation() -> None:
    generator = MdmMotionGenerator.__new__(MdmMotionGenerator)
    generator._seed = 7
    generator._lock_seed = True
    calls: list[int] = []
    generator._fixseed = calls.append

    generator._reset_seed_if_locked()
    generator._reset_seed_if_locked()

    assert calls == [7, 7]


def make_session(monkeypatch, tmp_path) -> tuple[Session, SimulatedUser]:
    config_path = tmp_path / "mpc.yaml"
    config_path.write_text(
        """
planner: arm_mpc_cartesian
steps: 6
horizon: 2
n_mpc_samples: 2
max_angle_delta: 0.01
pose: pose.npy
text_time: 5
preference_learning: false
trajectory_fraction: 1.0
uq:
  diffusion_samples: 2
  n_clusters: 1
cartesian:
  goals:
    - [0.1, 0.2, 0.3]
corrections:
  trigger_threshold: 0.02
""",
        encoding="utf-8",
    )
    user = SimulatedUser(
        name="restricted",
        description="",
        feedback_text="keep it comfortable",
        bounds=(HiddenBound("elbow_flexion", "lower_bound", low=0.5),),
    )
    fk = SmplLeftArmFK()
    rig = DemoRig.__new__(DemoRig)
    rig.cfg = load_mpc_config(config_path)
    rig.fk = fk
    rig.context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    rig.body_pos = np.zeros((22, 3))
    rig.spine3_pos = fk.tpose_spine3_pos
    rig.spine3_aa = np.zeros(3)
    rig._extra_costs = MethodType(
        lambda self, selected: CompositeTrajectoryCost([]), rig
    )
    rig._manual_planner = MethodType(
        lambda self, start, goal, extra: FakePlanner(extra_costs=extra), rig
    )
    rig._cfg_with_goal = MethodType(lambda self, goal: self.cfg, rig)
    rig.package_trajectory = MethodType(
        lambda self, traj, selected, current_mesh_only=False, pin_mesh=False: {
            "n_frames": len(traj),
            "current_mesh_only": current_mesh_only,
            "mesh_id": "mesh",
        },
        rig,
    )
    rig.meshes = SimpleNamespace(unpin=lambda mesh_id: None)
    session_dir = tmp_path / "demo_designer_artifacts" / "test_session"
    session = Session(
        rig=rig,
        persona_name=user.name,
        user=user,
        dir=session_dir,
        corpus=TrajectoryCorpus.create(session_dir / "trajectory_corpus", rig.context),
    )
    rig.session = session
    session.run_oracle = MethodType(lambda self, from_trigger: {}, session)
    monkeypatch.setattr(demo_session, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(demo_session, "goal_reach", lambda *args: {})
    return session, user


def test_trajectory_pauses_and_logs_discomfort_segment(monkeypatch, tmp_path) -> None:
    session, _ = make_session(monkeypatch, tmp_path)
    monkeypatch.setattr(
        demo_session,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] == 1 else 0.0]),
    )

    trajectory = session.start_trajectory(np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6])
    payload = session._trajectory_payload()

    assert payload["status"] == "paused"
    assert payload["trigger"] == {
        "step": 1,
        "reason": "discomfort",
        "violation": 0.03,
    }
    assert payload["trajectory"]["n_frames"] == 2
    assert trajectory.samples is None
    assert trajectory.mpc.pushed is None
    assert session.corpus.entries()[0] == {
        "index": 0,
        "kind": "executed_segment",
        "round": 0,
        "goal": [0.4, 0.5, 0.6],
        "n_frames": 2,
        "trigger_step": 1,
        "trigger_violation": 0.03,
        "feedback_text": "keep it comfortable",
        "comfortable_until": 1,
        "traj_file": "traj_000.npy",
        "features_file": "traj_000_features.csv",
    }


def test_trajectory_can_advance_one_live_frame_at_a_time(monkeypatch, tmp_path) -> None:
    session, _ = make_session(monkeypatch, tmp_path)
    monkeypatch.setattr(
        demo_session,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] == 1 else 0.0]),
    )

    session.start_trajectory(np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6], advance=False)

    assert session._trajectory_payload()["status"] == "running"
    assert session.corpus.entries() == []

    running = session.advance_trajectory(current_mesh_only=True)

    assert running["status"] == "running"
    assert running["trajectory"]["n_frames"] == 2
    assert running["trajectory"]["current_mesh_only"] is True
    assert session.corpus.entries() == []

    paused = session.advance_trajectory(current_mesh_only=True)

    assert paused["status"] == "paused"
    assert paused["trigger"]["reason"] == "discomfort"
    assert paused["trajectory"]["current_mesh_only"] is False
    assert session.corpus.entries()[0]["n_frames"] == 2


def test_exit_trajectory_keeps_session_context(monkeypatch, tmp_path) -> None:
    session, _ = make_session(monkeypatch, tmp_path)
    monkeypatch.setattr(
        demo_session,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] == 1 else 0.0]),
    )
    session.start_trajectory(np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6])
    artifact_dir = str(session.trajectory.artifact_dir)
    session.round_records = [{"index": 0}]
    session._round_costs = [FakeCost()]
    session.unified_cost = FakeCost()

    result = session.exit_trajectory()

    assert result == {"ok": True, "artifact_dir": artifact_dir}
    assert session.trajectory is None
    assert session.round_records == [{"index": 0}]
    assert len(session._round_costs) == 1
    assert session.unified_cost is not None


def test_new_trajectory_installs_session_costs_from_frame_zero(
    monkeypatch, tmp_path
) -> None:
    session, _ = make_session(monkeypatch, tmp_path)
    monkeypatch.setattr(
        demo_session,
        "compute_violations",
        lambda selected, context, q: np.array([0.0]),
    )
    learned = FakeCost()
    session._round_costs = [learned]

    trajectory = session.start_trajectory(np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6])

    assert trajectory.mpc.kwargs["extra_costs"].terms() == (learned,)
    assert session.trajectory_count == 1
    saved = json.loads((session.dir / "session.json").read_text(encoding="utf-8"))
    assert saved["trajectory_count"] == 1


def test_apply_round_resumes_same_planner_and_pauses_again(
    monkeypatch, tmp_path
) -> None:
    session, _ = make_session(monkeypatch, tmp_path)
    monkeypatch.setattr(
        demo_session,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] in (1, 3) else 0.0]),
    )
    session.start_trajectory(np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6])
    trajectory = session.trajectory
    planner = trajectory.mpc
    correction = np.stack([np.full((3, 3), 0.5), np.ones((3, 3))])
    trajectory.scaled_correction = correction
    trajectory._last_cost = FakeCost()

    def fake_commit(self):
        self._round_costs.append(self.trajectory._last_cost)
        self.round_records.append({"index": 0})
        return {"rounds": self.round_records, "unified": None}

    session.commit_round = MethodType(fake_commit, session)

    payload = session.apply_round_and_continue()

    assert session.trajectory.mpc is planner
    np.testing.assert_array_equal(planner.pushed, correction)
    assert payload["status"] == "paused"
    assert payload["trigger"]["step"] == 3
    assert payload["trajectory"]["n_frames"] == 4
    assert [entry["n_frames"] for entry in session.corpus.entries()] == [2, 2]


def test_ignore_comfort_violation_resumes_until_a_new_violation(
    monkeypatch, tmp_path
) -> None:
    session, _ = make_session(monkeypatch, tmp_path)
    monkeypatch.setattr(
        demo_session,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] in (1, 3) else 0.0]),
    )
    session.start_trajectory(np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6])

    payload = session.ignore_comfort_violation()

    assert payload["status"] == "paused"
    assert payload["trigger"] == {
        "step": 3,
        "reason": "discomfort",
        "violation": 0.03,
    }
    assert payload["trajectory"]["n_frames"] == 4
    assert payload["rounds"] == []


def test_commit_round_persists_all_recursive_cluster_labels(tmp_path) -> None:
    session = Session.__new__(Session)
    session.persona_name = "restricted"
    session.user = SimulatedUser("restricted", "", "", bounds=())
    session.started = "now"
    session.dir = tmp_path / "session"
    session.rounds = []
    session.round_records = []
    session._round_costs = []
    session.unified_cost = None
    session.combined_corpus_count = 0
    session.corpus = SimpleNamespace(entries=lambda: [])
    session.trajectory_count = 1
    session.beats = []
    trajectory = SimpleNamespace(
        goal=np.array([0.4, 0.5, 0.6]),
        trigger_step=1,
        trigger_reason="discomfort",
        trigger_violation=0.03,
    )
    session.trajectory = trajectory
    trajectory.cluster_levels = [
        ClusterLevel(
            np.array([], dtype=np.intp), np.array([], dtype=np.intp), 2, 3, 1.0
        ),
        ClusterLevel(
            np.array([], dtype=np.intp), np.array([], dtype=np.intp), 0, 3, 1.0
        ),
        ClusterLevel(
            np.array([], dtype=np.intp), np.array([], dtype=np.intp), 1, 2, 1.0
        ),
    ]
    cost_dir = tmp_path / "cost"
    cost_dir.mkdir()

    class FakeEvalState:
        def save(self, path: Path) -> None:
            path.write_bytes(b"state")

    trajectory._last_generation = SimpleNamespace(
        eval_state=FakeEvalState(),
        summaries={},
        images={},
        description="description",
        explanation="",
        interpretation="",
        grounding="",
    )
    trajectory._last_cost = FakeCost()
    trajectory._last_cost_dir = cost_dir
    trajectory._last_instruction = "feedback"
    trajectory._last_cost_payload = {"description": "accepted cost"}

    session.commit_round()

    assert session.rounds[0].cluster_labels == (2, 0, 1)
    assert session.round_records[0]["cluster_labels"] == [2, 0, 1]
    assert session.round_records[0]["code"] == session.rounds[0].cost_code
    saved = json.loads((session.dir / "session.json").read_text(encoding="utf-8"))
    assert saved["rounds"][0]["cluster_labels"] == [2, 0, 1]
    assert [beat["kind"] for beat in session.beats] == ["cost", "round"]


def test_cost_round_loads_legacy_records_without_cluster_labels(tmp_path) -> None:
    round_ = CostRound(
        index=0,
        goal=(0.1, 0.2, 0.3),
        feedback_text="feedback",
        trigger_step=1,
        round_dir=Path(tmp_path),
        state_path=Path(tmp_path) / "state.pkl",
        cost_code="code",
        params={},
        summaries={},
        image_paths=(),
        cluster_labels=(2, 1, 0),
    )

    data = round_.to_json()
    assert data["cluster_labels"] == [2, 1, 0]
    assert CostRound.from_json(data).cluster_labels == (2, 1, 0)
    data.pop("cluster_labels")
    assert CostRound.from_json(data).cluster_labels == ()


def test_remove_round_reindexes_remaining_feedback(tmp_path) -> None:
    def round_(index: int) -> CostRound:
        return CostRound(
            index=index,
            goal=(0.1, 0.2, 0.3),
            feedback_text=f"feedback {index}",
            trigger_step=index,
            round_dir=Path(tmp_path),
            state_path=Path(tmp_path) / "state.pkl",
            cost_code="code",
            params={},
            summaries={},
            image_paths=(),
        )

    session = Session.__new__(Session)
    session.rig = SimpleNamespace()
    session.persona_name = "restricted"
    session.started = "now"
    session.trajectory_count = 0
    session.dir = tmp_path / "session"
    session.rounds = [round_(0), round_(1), round_(2)]
    session.round_records = [{"index": 0}, {"index": 1}, {"index": 2}]
    session._round_costs = [FakeCost(), FakeCost(), FakeCost()]
    session.trajectory = None
    session.unified_cost = FakeCost()
    session.combined_corpus_count = 0

    payload = session.remove_round(1)

    assert [round_.index for round_ in session.rounds] == [0, 1]
    assert [round_.feedback_text for round_ in session.rounds] == [
        "feedback 0",
        "feedback 2",
    ]
    assert payload["unified"] is None
    saved = json.loads((session.dir / "session.json").read_text(encoding="utf-8"))
    assert [round_["index"] for round_ in saved["rounds"]] == [0, 1]
