"""Data structures a benchmark produces and an episode records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.approaches.cost_gen.structs import LearnOutcome
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    CorrectionIntent,
    SimulatedUser,
    Utterance,
)

RESOLVED_RESULTS = ("ok", "no_violation")


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
    goals: tuple[tuple[float, float, float], ...]
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
    ``correction_q`` is the executed correction and ``continuation`` the replan
    from its end, with ``retrigger_step`` its first violation (None = resolved).
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


@dataclass(frozen=True)
class Interaction:
    """One goal attempt by one approach for one persona: what every metric reads.

    ``user`` and ``context`` are carried so metrics can score any trajectory
    here against the persona's hidden bounds. ``executed`` is every frame the
    arm moved through for this goal, starting at the goal's start pose.
    ``result`` is ``no_violation`` (no feedback needed), ``ok`` (resolved after
    feedback), ``goal_not_reached``, ``capped`` or ``no_feedback_content``.
    """

    task: InteractionTask
    approach: str
    user: SimulatedUser
    context: MpcCostContext
    goal_index: int
    goal: np.ndarray
    oracle_path: np.ndarray
    initial_rollout: np.ndarray
    trigger_step: int | None
    rounds: tuple[FeedbackRound, ...]
    result: str
    reached: bool
    executed: np.ndarray

    @property
    def rounds_used(self) -> int:
        return len(self.rounds)

    @property
    def resolved(self) -> bool:
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
    context: MpcCostContext
    goal_index: int
    goal: np.ndarray
    utterance: Utterance
    q_feedback: np.ndarray
    nominal_plan: np.ndarray
    candidates: dict[int, np.ndarray]
    chosen_label: int
    learned_terms: int
