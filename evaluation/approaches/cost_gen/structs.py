"""Data structures cost generation consumes and produces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.costs import GeneratedPythonCost
from uncertain_feedback.planners.mpc.human import Human


@dataclass(frozen=True)
class RoundContext:
    """Everything an approach needs to learn from one resolved correction.

    ``human`` is the person at the feedback moment: its history is the executed
    motion and its ``q`` the configuration the correction starts from.
    ``retire`` are learned costs a rejected stall proposal left out; they are
    retired once this correction yields a cost (see :meth:`CostGen.retire`).
    """

    round_dir: Path
    goal: np.ndarray
    utterance_text: str
    grounding: GroundingResult
    human: Human
    event_index: int
    rejected_labels: frozenset[int]
    nominal_plan: np.ndarray | None = None
    retire: tuple[GeneratedPythonCost, ...] = ()


@dataclass(frozen=True)
class LearnOutcome:
    """What a learning update produced."""

    cost_accepted: bool
    unified_installed: bool
    description: str = ""
