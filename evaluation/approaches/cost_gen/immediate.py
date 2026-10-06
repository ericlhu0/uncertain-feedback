"""Immediate cost generation: stack every per-round cost into planning."""

from __future__ import annotations

from evaluation.approaches.cost_gen.base import CostGen
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from uncertain_feedback.planners.mpc.costs import GeneratedPythonCost


class ImmediateCostGen(CostGen):
    """Every accepted per-round cost is kept and stacked, until retired."""

    def learned_terms(self) -> list[GeneratedPythonCost]:
        return [
            cost
            for cost, round_ in zip(self._generated, self._cost_rounds)
            if not round_.retired
        ]

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        return self.record(ctx, self.generate(ctx))
