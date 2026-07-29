"""Shared orchestration for repeated MDM corrections in one MPC trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import numpy as np

from uncertain_feedback.planners.mpc import LeftArmMPCMDM
from uncertain_feedback.planners.mpc.costs import GeneratedPythonCost, MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.simulated_users import SimulatedUser, compute_violations

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.costs import CostRound
    from uncertain_feedback.planners.run import LoopResult

TriggerReason = Literal["text_time", "discomfort", "operator"]


@dataclass
class CorrectionTrigger:
    """Edge-trigger repeated discomfort while preserving the first text trigger."""

    threshold: float
    text_time: int | None
    automatic: bool = True
    first_correction_triggered: bool = False
    discomfort_armed: bool = True
    operator_requested: Callable[[], bool] | None = None

    def evaluate(self, step: int, violation: float | None) -> TriggerReason | None:
        """Decide whether this step should pause for feedback, and why."""
        if self.operator_requested is not None and self.operator_requested():
            # A live person asking outranks both the scripted step and the
            # discomfort edge, and needs no re-arming: they can ask again at will.
            self.first_correction_triggered = True
            self.discomfort_armed = False
            return "operator"
        uncomfortable = (
            self.automatic and violation is not None and violation > self.threshold
        )
        if self.first_correction_triggered:
            if not self.automatic or violation is None:
                return None
            if violation <= self.threshold:
                self.discomfort_armed = True
                return None
            if uncomfortable and self.discomfort_armed:
                self.discomfort_armed = False
                return "discomfort"
            return None

        if uncomfortable:
            self.first_correction_triggered = True
            self.discomfort_armed = False
            return "discomfort"
        if self.text_time is not None and step == self.text_time:
            self.first_correction_triggered = True
            self.discomfort_armed = False
            return "text_time"
        return None


@dataclass
class CorrectionRoundResult:
    """What one correction round produced: the trigger, the correction, and its cost."""

    round_index: int
    trajectory_index: int
    trigger_step: int
    trigger_reason: TriggerReason
    trigger_violation: float | None
    feedback_text: str
    correction_traj: np.ndarray
    generated_cost: GeneratedPythonCost | None
    cost_round: CostRound | None
    artifact_dir: Path


@dataclass
class CorrectionTrajectoryResult:
    """One trajectory's rollout plus every correction round taken during it."""

    loop_result: LoopResult
    rounds: list[CorrectionRoundResult]
    unified_cost: GeneratedPythonCost | None
    artifact_dir: Path


CorrectionHandler = Callable[
    [int, np.ndarray, list[np.ndarray], TriggerReason, float | None, int],
    CorrectionRoundResult,
]
FinishHandler = Callable[[Sequence[CorrectionRoundResult]], GeneratedPythonCost | None]


@dataclass
class CorrectionSession:
    """Run one MPC trajectory with any number of edge-triggered corrections."""

    mpc: LeftArmMPCMDM
    user: SimulatedUser
    cost_context: MpcCostContext
    feedback_text: str
    trigger_threshold: float
    text_time: int | None
    artifact_dir: Path
    handle_correction: CorrectionHandler
    finish: FinishHandler | None = None
    trajectory_index: int = 0
    prior_rounds: Sequence[CostRound] = ()
    prior_unified_cost: GeneratedPythonCost | None = None
    operator_requested: Callable[[], bool] | None = None
    rounds: list[CorrectionRoundResult] = field(default_factory=list, init=False)

    def run_trajectory(
        self,
        q0: np.ndarray,
        n_steps: int,
        *,
        progress: bool = False,
        progress_desc: str = "MPC",
    ) -> CorrectionTrajectoryResult:
        """Roll out ``n_steps``, pausing for a round whenever the trigger fires."""
        from uncertain_feedback.planners.run import run_planning_loop

        automatic = bool(self.user.bounds and self.user.feedback_text)
        trigger = CorrectionTrigger(
            threshold=self.trigger_threshold,
            text_time=self.text_time,
            automatic=automatic,
            operator_requested=self.operator_requested,
        )

        def on_pre_step(step: int, q: np.ndarray, q_history: list[np.ndarray]) -> None:
            violation = None
            if automatic:
                violation = float(
                    compute_violations(
                        self.user,
                        self.cost_context,
                        q_to_arm_aa(
                            q[np.newaxis], self.cost_context.fk.elbow_hinge_axis
                        ),
                    )[0]
                )
            reason = trigger.evaluate(step, violation)
            if reason is None:
                return
            result = self.handle_correction(
                step, q, q_history, reason, violation, len(self.rounds)
            )
            self.rounds.append(result)

        loop_result = run_planning_loop(
            self.mpc,
            q0,
            n_steps,
            on_pre_step=on_pre_step,
            progress=progress,
            progress_desc=progress_desc,
        )
        unified = self.prior_unified_cost
        if self.finish is not None:
            unified = self.finish(self.rounds)
        return CorrectionTrajectoryResult(
            loop_result=loop_result,
            rounds=list(self.rounds),
            unified_cost=unified,
            artifact_dir=self.artifact_dir,
        )
