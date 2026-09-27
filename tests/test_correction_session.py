"""Tests for the correction session loop: triggers, rounds, and persistence."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import io
import time
from argparse import Namespace

import numpy as np

from uncertain_feedback.envs.kinematic import KinematicEnv
from uncertain_feedback.planners import correction_session as session_module
from uncertain_feedback.planners import run as run_module
from uncertain_feedback.planners.correction_session import (
    CorrectionRoundResult,
    CorrectionSession,
    CorrectionTrigger,
)
from uncertain_feedback.planners.interactive import OperatorPause
from uncertain_feedback.planners.mpc import ArmMPC, FeedbackConfig
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import LEFT_ARM_CHAIN_INDICES
from uncertain_feedback.planners.run import RunSetup, run_repeated_correction_session
from uncertain_feedback.simulated_users import HiddenBound, SimulatedUser


def test_discomfort_before_text_time_consumes_first_trigger() -> None:
    trigger = CorrectionTrigger(threshold=0.02, text_time=10)

    assert trigger.evaluate(2, 0.03) == "discomfort"
    assert trigger.evaluate(10, 0.03) is None


def test_text_time_is_first_trigger_when_comfortable() -> None:
    trigger = CorrectionTrigger(threshold=0.02, text_time=3)

    assert trigger.evaluate(2, 0.0) is None
    assert trigger.evaluate(3, 0.0) == "text_time"


def test_discomfort_rearms_only_after_returning_to_comfort() -> None:
    trigger = CorrectionTrigger(threshold=0.02, text_time=0)

    assert trigger.evaluate(0, 0.0) == "text_time"
    assert trigger.evaluate(1, 0.03) is None
    assert trigger.evaluate(2, 0.02) is None
    assert trigger.evaluate(3, 0.03) == "discomfort"
    assert trigger.evaluate(4, 0.04) is None


def test_unrestricted_trigger_uses_text_time_only() -> None:
    trigger = CorrectionTrigger(threshold=0.02, text_time=3, automatic=False)

    assert trigger.evaluate(1, 1.0) is None
    assert trigger.evaluate(3, None) == "text_time"
    assert trigger.evaluate(5, 1.0) is None


def test_text_time_can_be_disabled() -> None:
    trigger = CorrectionTrigger(threshold=0.02, text_time=None)

    assert trigger.evaluate(0, 0.0) is None
    assert trigger.evaluate(10, 0.0) is None


def test_operator_outranks_text_time_and_needs_no_rearming() -> None:
    requests = iter((True, False, True, True))
    trigger = CorrectionTrigger(
        threshold=0.02,
        text_time=1,
        automatic=False,
        operator_requested=lambda: next(requests),
    )

    assert trigger.evaluate(0, None) == "operator"
    # The operator consumed the first trigger, so text_time no longer fires.
    assert trigger.evaluate(1, None) is None
    assert trigger.evaluate(2, None) == "operator"
    assert trigger.evaluate(3, None) == "operator"


def _await_request(pause: OperatorPause) -> None:
    deadline = time.monotonic() + 5.0
    while not pause.requested():
        assert time.monotonic() < deadline, "the operator's line never arrived"


def test_operator_pause_uses_the_requesting_line_as_feedback() -> None:
    pause = OperatorPause(io.StringIO("raise my arm higher\n"))

    _await_request(pause)
    # Requesting does not consume the line, so the pause survives repeated polls.
    assert pause.requested()
    assert pause.feedback(7) == "raise my arm higher"
    assert not pause.requested()


def test_operator_pause_prompts_when_the_request_line_is_empty() -> None:
    pause = OperatorPause(io.StringIO("\nkeep my elbow down\n"))

    _await_request(pause)
    assert pause.feedback(3) == "keep my elbow down"


def test_remaining_mdm_trajectory_is_snapshot_and_replacement_discards_suffix() -> None:
    human = Human()
    old = np.stack([np.full((3, 3), value) for value in (0.1, 0.2, 0.3)])
    old_q = human.q_from_arm_aa(old)
    planner = ArmMPC(
        human,
        visualize=False,
        feedback=FeedbackConfig(max_playback_delta=10.0),
    )
    planner.push_trajectory(old)
    q1 = planner.step().q

    snapshot = planner.remaining_mdm_trajectory(q1)
    assert snapshot is not None
    np.testing.assert_allclose(snapshot[0], q1)
    np.testing.assert_allclose(snapshot[1:], old_q[1:])

    replacement = np.stack([np.full((3, 3), value) for value in (0.8, 0.9)])
    replacement_q = human.q_from_arm_aa(replacement)
    planner.push_trajectory(replacement)
    snapshot[1:] = -1.0

    np.testing.assert_allclose(planner.step().q, replacement_q[0])
    np.testing.assert_allclose(planner.step().q, replacement_q[1])
    assert planner.remaining_mdm_trajectory(replacement_q[1]) is None


def test_session_triggers_again_after_comfort_rearms(monkeypatch, tmp_path) -> None:
    q0 = np.zeros(7, dtype=np.float64)
    planner = ArmMPC(Human(), visualize=False, feedback=FeedbackConfig())
    monkeypatch.setattr(planner, "step", lambda: planner.human)
    violations = iter((0.0, 0.03, 0.0, 0.03, 0.04))
    monkeypatch.setattr(
        session_module,
        "compute_violations",
        lambda _user, _human, _traj: np.array([next(violations)]),
    )
    user = SimulatedUser(
        name="restricted",
        description="",
        feedback_text="keep it comfortable",
        bounds=(HiddenBound("elbow_flexion", "lower_bound", low=0.5),),
    )
    handled: list[tuple[int, str]] = []

    def handle(step, _human, reason, violation, local_index):
        handled.append((step, reason))
        return CorrectionRoundResult(
            round_index=local_index,
            trajectory_index=0,
            trigger_step=step,
            trigger_reason=reason,
            trigger_violation=violation,
            feedback_text=user.feedback_text,
            correction_traj=np.stack([q0]),
            generated_cost=None,
            cost_round=None,
            artifact_dir=tmp_path / f"round_{local_index}",
        )

    session = CorrectionSession(
        mpc=planner,
        user=user,
        feedback_text=user.feedback_text,
        trigger_threshold=0.02,
        text_time=0,
        artifact_dir=tmp_path,
        handle_correction=handle,
    )

    result = session.run_trajectory(5)

    assert handled == [(0, "text_time"), (3, "discomfort")]
    assert len(result.rounds) == 2


def test_interactive_runner_takes_its_correction_from_the_operator(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "mpc.yaml"
    config_path.write_text(
        """
steps: 4
horizon: 2
n_mpc_samples: 2
max_angle_delta: 0.01
feedback:
  text_time: 0
preference_learning: false
llm_cost:
  enabled: false
  artifact_dir: artifacts
corrections:
  trigger_threshold: 0.02
""",
        encoding="utf-8",
    )
    cfg = load_mpc_config(config_path)
    planner = ArmMPC(Human(), visualize=False, feedback=FeedbackConfig())
    monkeypatch.setattr(planner, "step", lambda: planner.human)
    requests = iter((False, True, False, False))

    class FakePause:
        """Operator who asks for one correction at step 1."""

        @staticmethod
        def requested() -> bool:
            return next(requests)

        @staticmethod
        def feedback(_step: int) -> str:
            return "raise my arm higher"

    monkeypatch.setattr(run_module, "OperatorPause", FakePause)
    user = SimulatedUser(
        name="unrestricted", description="", feedback_text="", bounds=()
    )

    class FakeGenerator:
        """Motion generator stand-in returning a canned correction."""

        @staticmethod
        def generate_positions(_text, human, **_kwargs):
            # Mirrors the real contract: the returned motion starts at the
            # person's current pose.
            positions = np.zeros((1, 2, 22, 3))
            positions[:, :, LEFT_ARM_CHAIN_INDICES] = human.fk_positions_from_q(human.q)
            return positions

    setup = RunSetup(
        mpc=planner,
        gen=FakeGenerator(),  # type: ignore[arg-type]
        human=planner.human,
        uses_mdm=True,
        visualize=False,
        compact=False,
        user=user,
        env=KinematicEnv(),
        extra_costs=CompositeTrajectoryCost([]),
    )
    args = Namespace(
        text=None,
        text_time=None,
        interactive=True,
        mdm_frames=None,
        save_motion=None,
        frozen_body=False,
        mpc_config=config_path,
    )

    result, _, _ = run_repeated_correction_session(
        args, cfg, setup, tmp_path, tmp_path / "learned.yaml"
    )

    # text_time: 0 is ignored, so the only round is the one the operator asked for.
    assert [(r.trigger_step, r.trigger_reason) for r in result.rounds] == [
        (1, "operator")
    ]
    assert result.rounds[0].feedback_text == "raise my arm higher"


def test_runner_persists_multiple_corrections_in_one_trajectory(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "mpc.yaml"
    config_path.write_text(
        """
steps: 5
horizon: 2
n_mpc_samples: 2
max_angle_delta: 0.01
feedback:
  text_time: 0
preference_learning: false
llm_cost:
  enabled: false
  artifact_dir: artifacts
corrections:
  trigger_threshold: 0.02
""",
        encoding="utf-8",
    )
    cfg = load_mpc_config(config_path)
    planner = ArmMPC(Human(), visualize=False, feedback=FeedbackConfig())
    monkeypatch.setattr(planner, "step", lambda: planner.human)
    violations = iter((0.0, 0.03, 0.0, 0.03, 0.04))
    monkeypatch.setattr(
        session_module,
        "compute_violations",
        lambda _user, _human, _traj: np.array([next(violations)]),
    )
    user = SimulatedUser(
        name="restricted",
        description="",
        feedback_text="keep it comfortable",
        bounds=(HiddenBound("elbow_flexion", "lower_bound", low=0.5),),
    )

    class FakeGenerator:
        """Motion generator stand-in returning a canned correction."""

        @staticmethod
        def generate_positions(_text, human, **_kwargs):
            # Mirrors the real contract: the returned motion starts at the
            # person's current pose.
            positions = np.zeros((1, 2, 22, 3))
            positions[:, :, LEFT_ARM_CHAIN_INDICES] = human.fk_positions_from_q(human.q)
            return positions

    setup = RunSetup(
        mpc=planner,
        gen=FakeGenerator(),  # type: ignore[arg-type]
        human=planner.human,
        uses_mdm=True,
        visualize=False,
        compact=False,
        user=user,
        env=KinematicEnv(),
        extra_costs=CompositeTrajectoryCost([]),
    )
    args = Namespace(
        text=None,
        text_time=None,
        interactive=False,
        mdm_frames=None,
        save_motion=None,
        frozen_body=False,
        mpc_config=config_path,
    )

    result, _, _ = run_repeated_correction_session(
        args, cfg, setup, tmp_path, tmp_path / "learned.yaml"
    )

    assert [round_.trigger_step for round_ in result.rounds] == [0, 3]
    assert (result.artifact_dir / "round_00" / "correction.npy").exists()
    assert (result.artifact_dir / "round_01" / "correction.npy").exists()
    assert (result.artifact_dir / "executed_trajectory.npy").exists()
