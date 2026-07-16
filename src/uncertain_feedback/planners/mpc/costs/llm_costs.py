"""Single-pass (``llm``) staged cost generator, plus back-compat re-exports.

The shared primitives now live in :mod:`...costs.generated`; this module re-exports
them so existing ``from ...costs.llm_costs import ...`` imports keep working, and
adds :class:`LlmCostGenerator` — the one-pass generator that runs the three focused
stages (interpret -> ground -> author) once, with no rollout feedback loop.
"""

from __future__ import annotations

from uncertain_feedback.planners.mpc.costs.cost_generator import (
    CostGenerator,
    rank_candidate_cost,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedCostValidationError,
    GeneratedPythonCost,
    LlmCostResponse,
    build_generated_cost_context,
    build_motion_summaries,
    compile_generated_cost,
    parse_llm_cost_response,
    render_prompt_images,
)

__all__ = [
    "GeneratedCostContext",
    "GeneratedCostValidationError",
    "GeneratedPythonCost",
    "LlmCostResponse",
    "build_generated_cost_context",
    "build_motion_summaries",
    "compile_generated_cost",
    "parse_llm_cost_response",
    "render_prompt_images",
    "LlmCostGenerator",
]


class LlmCostGenerator(CostGenerator):
    """Single-pass staged cost generator: interpret -> ground -> author, no feedback."""

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            llm = self.make_llm()
            interpretation = self.interpret(llm)
            specification = self.ground(llm, interpretation)
            response, cost = self.author(llm, specification)
            ranking = rank_candidate_cost(self.context, cost)
            self.save_response(response)
            self.save_rationale(
                response,
                interpret_raw=interpretation,
                ground_raw=specification,
                ranking=ranking,
            )
            self.require_original_plan_improvement(ranking)
            if install:
                self.install(cost)
            self._on_success(cost, installed=install)
            return cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None
