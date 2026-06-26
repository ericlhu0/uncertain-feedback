"""Single-turn (``llm``) cost generator, plus back-compat re-exports.

The shared primitives now live in :mod:`...costs.generated`; this module re-exports
them so existing ``from ...costs.llm_costs import ...`` imports keep working, and
adds :class:`LlmCostGenerator` — the one-shot cost generator (the original
single-turn behavior, lifted out of ``planners/run.py``).
"""

from __future__ import annotations

from uncertain_feedback.planners.mpc.costs.cost_generator import CostGenerator
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
    """Single-turn cost generator: one LLM call, no feedback loop."""

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            prompt_text, image_input = self.build_prompt()
            (self.run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            llm = self.make_llm()
            raw = llm.get_full_output(prompt_text, image_input=image_input)
            (self.run_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            response, cost = self.parse_cost(raw)
            self.save_response(response)
            if install:
                self.install(cost)
            self._on_success(cost, installed=install)
            return cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None
