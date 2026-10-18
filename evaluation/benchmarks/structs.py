"""Data structures a benchmark produces and an episode records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.approaches.cost_gen.structs import LearnOutcome
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.goal_spaces import GoalRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    CorrectionIntent,
    SimulatedUser,
    Utterance,
    compute_violations,
)

RESOLVED_RESULTS = ("ok", "no_violation")


def crossings(
    user: SimulatedUser, human: Human, trajectory: np.ndarray, threshold: float
) -> int:
    """Separate stretches of ``trajectory`` past the persona's limits by ``threshold``."""
    over = compute_violations(user, human, trajectory) > threshold
    return int(over[0]) + int(np.count_nonzero(over[1:] & ~over[:-1]))


@dataclass(frozen=True)
class InteractionTask:
    """One simulated interaction: a persona pursuing a sequence of goals.

    ``feedback_text`` is the literal utterance the ``scripted`` verbalizer
    replays every round, for comparing grounding methods on one chosen sentence
    instead of a synthesized one; the persona's hidden intent still drives
    candidate selection. ``user`` overrides the persona registry (generated
    scenarios carry synthetic users) and ``start_q`` the rig's start pose.
    """

    persona: str
    verbalizer: str
    goals: tuple[GoalRegion, ...]
    max_rounds: int
    seed: int
    feedback_text: str | None = None
    user: SimulatedUser | None = None
    start_q: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FeedbackRound:
    """One feedback round: trigger, utterance, menu, choice, learning, continuation.

    ``nominal_plan`` is what the planner would have done from ``q_feedback``
    under its costs at the time; ``intent`` is the hidden contrast against the
    oracle the utterance was verbalized from; ``grounding.candidates`` is the
    menu the approach proposed and ``choice`` how the persona judged it;
    ``correction_q`` is the executed correction (just the stop pose when the
    grounder moves nothing, as the nominal grounder does) and ``continuation`` everything
    the arm did from its end until the round's outcome: the replan under the
    learned costs and, where it stalled short of the goal, each stall proposal
    the persona approved (``approvals`` of them) and the replan after it.
    ``retrigger_step`` is its first violation. ``proposal_rejected`` means it
    ended at a stall whose proposal the persona turned down, so the next round
    corrects it. The round resolved its feedback when neither happened.
    ``plan_seconds``, ``ground_seconds`` and ``learn_seconds`` are the robot's
    wall-clock time planning (the nominal plan and the continuation),
    grounding and learning.
    """

    round_index: int
    event_index: int
    round_dir: Path
    q_feedback: np.ndarray
    nominal_plan: np.ndarray
    intent: CorrectionIntent
    utterance: Utterance
    grounding: GroundingResult
    choice: ChoiceResult
    outcome: LearnOutcome
    correction_q: np.ndarray
    continuation: np.ndarray
    retrigger_step: int | None
    ground_seconds: float
    learn_seconds: float
    proposal_rejected: bool = False
    approvals: int = 0
    plan_seconds: float = float("nan")

    @property
    def resolved(self) -> bool:
        """Whether the round's continuation needed no further feedback."""
        return self.retrigger_step is None and not self.proposal_rejected


@dataclass(frozen=True)
class Interaction:
    """One goal attempt by one approach for one persona: what every metric reads.

    ``user`` and ``human`` are carried so metrics can score any trajectory
    here against the persona's hidden bounds. ``initial_rollout`` is the motion
    before any feedback, ``trigger_step`` its first violation or, when
    ``stall_rejected``, the stall whose proposal the persona turned down, and
    ``plan_seconds`` the time spent planning it. ``executed`` is
    every frame the arm moved through for this goal, starting at the goal's
    start pose. ``result`` is ``no_violation`` (no feedback needed), ``ok``
    (resolved after feedback), ``goal_not_reached``, ``capped`` or
    ``no_feedback_content``. ``approvals`` counts the stall proposals the
    persona approved on the way, and ``approved_spans`` are the inclusive
    frame ranges of ``executed`` the arm spent following them rather than its
    learned costs. ``trigger_threshold`` is the violation the persona stops
    the arm at.
    """

    task: InteractionTask
    approach: str
    user: SimulatedUser
    human: Human
    goal_index: int
    goal: GoalRegion
    oracle_path: np.ndarray
    initial_rollout: np.ndarray
    trigger_step: int | None
    rounds: tuple[FeedbackRound, ...]
    result: str
    reached: bool
    executed: np.ndarray
    approvals: int = 0
    approved_spans: tuple[tuple[int, int], ...] = ()
    stall_rejected: bool = False
    plan_seconds: float = float("nan")
    trigger_threshold: float = 0.02

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Older pickles stored only whether a proposal was approved; those
        # episodes ended at the first approval, so that is the count.
        if "approvals" not in state:
            state = dict(state)
            state["approvals"] = int(state.pop("proposal_approved", False))
        self.__dict__.update(state)

    @property
    def rounds_used(self) -> int:
        """Feedback rounds this goal consumed."""
        return len(self.rounds)

    @property
    def user_turns(self) -> int:
        """Times the persona had to respond: feedback rounds plus approvals.

        A rejected proposal is answered by the next round's correction, so it
        is already one of the rounds.
        """
        return self.rounds_used + self.approvals

    @property
    def proposal_approved(self) -> bool:
        """Whether the arm followed any approved stall proposal on this goal.

        Kept for analysis code written before approvals were counted.
        """
        return self.approvals > 0

    @property
    def violation_count(self) -> int:
        """Times the arm went past the persona's limits by ``trigger_threshold``.

        Planned motion stops at its first crossing for feedback: the motion
        before feedback (unless it ended at a turned-down stall proposal) and
        each round's continuation cross at most once. A correction runs whole,
        so every separate crossing inside it counts.
        """
        first = self.trigger_step is not None and not self.stall_rejected
        stops = int(first) + sum(r.retrigger_step is not None for r in self.rounds)
        return stops + sum(
            crossings(self.user, self.human, r.correction_q, self.trigger_threshold)
            for r in self.rounds
        )

    @property
    def motion_steps(self) -> int:
        """Frames the arm moved through for this goal: its time to complete it."""
        return len(self.executed) - 1

    @property
    def robot_seconds(self) -> float:
        """Wall-clock time the robot spent planning, grounding and learning."""
        return self.plan_seconds + sum(
            r.plan_seconds + r.ground_seconds + r.learn_seconds for r in self.rounds
        )

    @property
    def resolved(self) -> bool:
        """Whether the goal ended without an unresolved violation."""
        return self.result in RESOLVED_RESULTS


@dataclass(frozen=True)
class MenuProbe:
    """One test reach's proposed menu after learning on the task's first goal.

    Records the feedback situation the unlearned nominal plan provoked on a
    later goal and every candidate the approach proposed there; the menu
    metric reads the candidates against the persona's hidden bounds.
    """

    task: InteractionTask
    approach: str
    user: SimulatedUser
    human: Human
    goal_index: int
    goal: GoalRegion
    utterance: Utterance
    q_feedback: np.ndarray
    nominal_plan: np.ndarray
    candidates: dict[int, np.ndarray]
    chosen_label: int
    learned_terms: int
