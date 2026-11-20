"""The hidden-limit ceiling: plan with the persona's own limit, learn nothing."""

from __future__ import annotations

from evaluation.approaches.cost_gen.base import CostGen
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from uncertain_feedback.planners.mpc.costs import (
    GeneratedPythonCost,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import SimulatedUser


class HiddenLimitCostGen(CostGen):
    """Plans from the first step with the persona's hidden limit, a ceiling.

    The limit is the hand-written margins of ``benchmarks/speed.py``'s
    ``CONSTRAINTS``, installed as a learned cost would be, so the planner
    charges it hard and learned steering steers with it. Nothing is learned
    and nothing is retired. Never the method: it reads ground truth, which
    :class:`Approach` hands it through :meth:`bind`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._limit: GeneratedPythonCost | None = None

    def bind(self, human: Human, user: SimulatedUser) -> None:
        """Install ``user``'s hidden limit on ``human``; call after :meth:`reset`."""
        # Imported here: benchmarks import the approaches.
        # pylint: disable-next=import-outside-toplevel
        from evaluation.benchmarks.speed import CONSTRAINTS

        assert self._cfg is not None, "reset() must run before bind()"
        code, params = CONSTRAINTS[user.name]
        context = build_generated_cost_context(
            human,
            human.q[None],
            window=1,
            limit_weight=self._cfg.llm_cost.limit_weight,
            pullback_weight=self._cfg.llm_cost.pullback_weight,
            enforcement=self._cfg.llm_cost.enforcement,
        )
        self._limit = GeneratedPythonCost(code, params, context, "hidden limit")

    def learned_terms(self) -> list[GeneratedPythonCost]:
        return [] if self._limit is None else [self._limit]

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        del ctx
        return LearnOutcome(cost_accepted=False, unified_installed=False)
