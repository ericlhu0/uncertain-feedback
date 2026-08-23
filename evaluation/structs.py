"""Data structures shared across the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InteractionTask:
    """One simulated interaction: a persona pursuing a sequence of goals.

    ``feedback_text`` is the literal utterance the ``scripted`` verbalizer
    replays every round, for comparing grounding methods on one chosen sentence
    instead of a synthesized one; the persona's hidden intent still drives
    candidate selection.
    """

    persona: str
    verbalizer: str
    goals: tuple[tuple[float, float, float], ...]
    max_rounds: int
    seed: int
    feedback_text: str | None = None


@dataclass(frozen=True)
class GroundingResult:
    """Candidate motions for one utterance and the selected correction."""

    candidates: dict[int, np.ndarray]
    chosen_label: int
    magnitude: float
    correction_traj: np.ndarray


@dataclass(frozen=True)
class RoundContext:
    """Everything an approach needs to learn from one resolved correction."""

    round_dir: Path
    goal: np.ndarray
    utterance_text: str
    grounding: GroundingResult
    q_feedback: np.ndarray
    q_history: list[np.ndarray]
    event_index: int
    rejected_labels: frozenset[int]
    nominal_plan: np.ndarray | None = None


@dataclass(frozen=True)
class LearnOutcome:
    """What a learning update produced."""

    cost_accepted: bool
    unified_installed: bool
    description: str = ""
