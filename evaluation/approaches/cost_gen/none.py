"""No cost generation: execute corrections but learn nothing."""

from __future__ import annotations

from evaluation.approaches.cost_gen.base import CostGen
from evaluation.structs import LearnOutcome, RoundContext
from uncertain_feedback.planners.mpc.costs import GeneratedPythonCost


class NoCostGen(CostGen):
    """Nothing persists into planning."""

    def learned_terms(self) -> list[GeneratedPythonCost]:
        return []

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        del ctx
        return LearnOutcome(cost_accepted=False, unified_installed=False)
