from pathlib import Path
from types import MethodType

import numpy as np

from uncertain_feedback.demo_designer import core
from uncertain_feedback.demo_designer.core import DemoSession
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

    def __call__(self, q_trajs):
        return np.zeros(q_trajs.shape[0])


class FakePlanner:
    mdm_ready_to_terminate = True

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


def make_session(monkeypatch, tmp_path) -> tuple[DemoSession, SimulatedUser]:
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
    session = DemoSession.__new__(DemoSession)
    session.cfg = load_mpc_config(config_path)
    session.fk = fk
    session.context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    session.body_pos = np.zeros((22, 3))
    session.spine3_pos = fk.tpose_spine3_pos
    session.spine3_aa = np.zeros(3)
    session.gen = object()
    session.initial_hml_pose = np.zeros(263)
    session.persona_name = user.name
    session.get_persona = MethodType(lambda self, name: user, session)
    session._extra_costs = MethodType(
        lambda self, selected: CompositeTrajectoryCost([]), session
    )
    session.package_trajectory = MethodType(
        lambda self, traj, selected: {"n_frames": len(traj)}, session
    )
    monkeypatch.setattr(core, "LeftArmMPCCartesian", FakePlanner)
    monkeypatch.setattr(core, "violation_metrics", lambda *args: {})
    monkeypatch.setattr(core, "goal_reach", lambda *args: {})
    return session, user


def test_manual_trajectory_pauses_for_cluster_workflow(monkeypatch, tmp_path) -> None:
    session, user = make_session(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        core,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] == 1 else 0.0]),
    )

    payload = session.start_manual_trajectory(
        np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6], user.name
    )

    assert payload["status"] == "paused"
    assert payload["trigger"] == {
        "step": 1,
        "reason": "discomfort",
        "violation": 0.03,
    }
    assert payload["trajectory"]["n_frames"] == 2
    assert session.samples is None
    assert session.manual_trajectory.mpc.pushed is None


def test_exit_manual_trajectory_clears_trajectory_state(monkeypatch, tmp_path) -> None:
    session, user = make_session(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        core,
        "compute_violations",
        lambda selected, context, q: np.array([0.03 if q[0, 0, 0] == 1 else 0.0]),
    )
    payload = session.start_manual_trajectory(
        np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6], user.name
    )
    artifact_dir = payload["artifact_dir"]

    result = session.exit_manual_trajectory()

    assert result == {"ok": True, "artifact_dir": artifact_dir}
    assert session.manual_trajectory is None
    assert session.base_traj is None
    assert session.goal is None
    assert session.trigger_step is None
    assert session.q_feedback is None
    assert session.q_history == []
    assert session.oracle_traj is None
    assert session.rounds == []
    assert session.samples is None


def test_apply_round_resumes_same_planner_and_pauses_again(
    monkeypatch, tmp_path
) -> None:
    session, user = make_session(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        core,
        "compute_violations",
        lambda selected, context, q: np.array(
            [0.03 if q[0, 0, 0] in (1, 3) else 0.0]
        ),
    )
    session.start_manual_trajectory(
        np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6], user.name
    )
    planner = session.manual_trajectory.mpc
    correction = np.stack([np.full((3, 3), 0.5), np.ones((3, 3))])
    session.scaled_correction = correction
    session._last_cost = FakeCost()
    session._round_costs = []
    session._round_generations = []
    session.rounds = []
    session.round_records = []
    session.unified_cost = None

    def fake_commit(self):
        self._round_costs.append(self._last_cost)
        self.round_records.append({"index": 0})
        return {"rounds": self.round_records, "unified": None}

    session.commit_round = MethodType(fake_commit, session)
    session._clear_pending_feedback = MethodType(lambda self: None, session)

    payload = session.apply_round_and_continue()

    assert session.manual_trajectory.mpc is planner
    np.testing.assert_array_equal(planner.pushed, correction)
    assert payload["status"] == "paused"
    assert payload["trigger"]["step"] == 3
    assert payload["trajectory"]["n_frames"] == 4


def test_ignore_comfort_violation_resumes_until_a_new_violation(
    monkeypatch, tmp_path
) -> None:
    session, user = make_session(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        core,
        "compute_violations",
        lambda selected, context, q: np.array(
            [0.03 if q[0, 0, 0] in (1, 3) else 0.0]
        ),
    )
    session.start_manual_trajectory(
        np.zeros((3, 3)).tolist(), [0.4, 0.5, 0.6], user.name
    )

    payload = session.ignore_comfort_violation()

    assert payload["status"] == "paused"
    assert payload["trigger"] == {
        "step": 3,
        "reason": "discomfort",
        "violation": 0.03,
    }
    assert payload["trajectory"]["n_frames"] == 4
    assert payload["rounds"] == []


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

    session = DemoSession.__new__(DemoSession)
    session.rounds = [round_(0), round_(1), round_(2)]
    session.round_records = [{"index": 0}, {"index": 1}, {"index": 2}]
    session._round_costs = [FakeCost(), FakeCost(), FakeCost()]
    session._round_generations = [object(), object(), object()]
    session.manual_trajectory = None
    session.unified_cost = FakeCost()

    payload = session.remove_round(1)

    assert [round_.index for round_ in session.rounds] == [0, 1]
    assert [round_.feedback_text for round_ in session.rounds] == [
        "feedback 0",
        "feedback 2",
    ]
    assert payload["unified"] is None
