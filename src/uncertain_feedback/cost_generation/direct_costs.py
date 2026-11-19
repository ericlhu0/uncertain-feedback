"""Single-call (``direct``) cost generator: no interpret/ground/author stages."""

from __future__ import annotations

from uncertain_feedback.cost_generation.base import CostGenerator
from uncertain_feedback.cost_generation.prompts import build_direct_prompt
from uncertain_feedback.evaluation_mechanism import rank_candidate_cost
from uncertain_feedback.planners.mpc.costs import GeneratedPythonCost


class DirectCostGenerator(CostGenerator):
    """Ask for the cost JSON in one call from the staged path's inputs."""

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            text, image_paths = build_direct_prompt(
                self.instruction,
                self.summaries,
                self.images if self.use_images else {},
            )
            raw = self._run_stage(
                self.make_llm(),
                "direct",
                text,
                image_input=[str(path) for path in image_paths] or None,
            )
            response, cost = self.parse_cost(raw)
            ranking = rank_candidate_cost(self.context, cost)
            self.save_response(response)
            self.save_rationale(
                response, interpret_raw=None, ground_raw=None, ranking=ranking
            )
            self.require_original_plan_improvement(ranking)
            if install:
                self.install(cost)
            self._on_success(cost, installed=install)
            return cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None
