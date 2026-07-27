"""Tests for the correction session loop: triggers, rounds, and persistence."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from argparse import Namespace

import numpy as np

from uncertain_feedback.envs.kinematic import KinematicEnv
from uncertain_feedback.planners import correction_session as session_module
from uncertain_feedback.planners.correction_session import (
    CorrectionRoundResult,
    CorrectionSession,
    CorrectionTrigger,
)
from uncertain_feedback.planners.mpc import LeftArmMPCMDM, SmplLeftArmFK
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    MpcCostContext,
)
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


def test_remaining_mdm_trajectory_is_snapshot_and_replacement_discards_suffix() -> None:
    fk = SmplLeftArmFK()
    q0 = np.zeros(7, dtype=np.float64)
    old = np.stack([np.full((3, 3), value) for value in (0.1, 0.2, 0.3)])
    old_q = fk.arm_aa_to_q_batch(old)
    planner = LeftArmMPCMDM(
        goals=[q0],
        visualize=False,
        fk=fk,
        max_playback_delta=10.0,
    )
    planner.push_trajectory(old)
    q1 = planner.step(q0)

    snapshot = planner.remaining_mdm_trajectory(q1)
    assert snapshot is not None
    np.testing.assert_allclose(snapshot[0], q1)
    np.testing.assert_allclose(snapshot[1:], old_q[1:])

    replacement = np.stack([np.full((3, 3), value) for value in (0.8, 0.9)])
    replacement_q = fk.arm_aa_to_q_batch(replacement)
    planner.push_trajectory(replacement)
    snapshot[1:] = -1.0

    np.testing.assert_allclose(planner.step(q1), replacement_q[0])
    np.testing.assert_allclose(planner.step(replacement_q[0]), replacement_q[1])
    assert planner.remaining_mdm_trajectory(replacement_q[1]) is None


def test_session_triggers_again_after_comfort_rearms(monkeypatch, tmp_path) -> None:
    q0 = np.zeros(7, dtype=np.float64)
    fk = SmplLeftArmFK()
    planner = LeftArmMPCMDM(goals=[np.ones(7)], visualize=False, fk=fk)
    monkeypatch.setattr(planner, "step", np.asarray)
    violations = iter((0.0, 0.03, 0.0, 0.03, 0.04))
    monkeypatch.setattr(
        session_module,
        "compute_violations",
        lambda _user, _context, _q: np.array([next(violations)]),
    )
    user = SimulatedUser(
        name="restricted",
        description="",
        feedback_text="keep it comfortable",
        bounds=(HiddenBound("elbow_flexion", "lower_bound", low=0.5),),
    )
    handled: list[tuple[int, str]] = []

    def handle(step, _q, _history, reason, violation, local_index):
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
        cost_context=MpcCostContext(
            fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
        ),
        feedback_text=user.feedback_text,
        trigger_threshold=0.02,
        text_time=0,
        artifact_dir=tmp_path,
        handle_correction=handle,
    )

    result = session.run_trajectory(q0, 5)

    assert handled == [(0, "text_time"), (3, "discomfort")]
    assert len(result.rounds) == 2


def test_runner_persists_multiple_corrections_in_one_trajectory(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "mpc.yaml"
    config_path.write_text(
        """
planner: arm_mpc_mdm
steps: 5
horizon: 2
n_mpc_samples: 2
max_angle_delta: 0.01
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
    q0 = np.zeros(7, dtype=np.float64)
    fk = SmplLeftArmFK()
    planner = LeftArmMPCMDM(goals=[np.ones(7)], visualize=False, fk=fk)
    monkeypatch.setattr(planner, "step", np.asarray)
    violations = iter((0.0, 0.03, 0.0, 0.03, 0.04))
    monkeypatch.setattr(
        session_module,
        "compute_violations",
        lambda _user, _context, _q: np.array([next(violations)]),
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
        def build_pose_from_arm_aa(_initial_pose, arm_aa):
            return np.asarray(arm_aa)

        @staticmethod
        def generate_left_arm_trajectory(_text, *, start_pose, **_kwargs):
            return np.stack([start_pose, start_pose])

    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    setup = RunSetup(
        mpc=planner,
        gen=FakeGenerator(),
        fk=fk,
        cost_context=context,
        body_pos=None,
        spine3_pos=fk.tpose_spine3_pos,
        spine3_aa=np.zeros(3),
        q0=q0,
        initial_pose=np.zeros(263),
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
