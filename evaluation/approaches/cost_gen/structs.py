"""Data structures cost generation consumes and produces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.metrics.grounding.structs import GroundingResult


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
