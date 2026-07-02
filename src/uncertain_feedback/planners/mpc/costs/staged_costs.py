"""Three-stage (``staged``) cost generator.

Splits the single monolithic cost-generation call into three focused LLM calls so
the model reasons about one thing at a time instead of interpreting the correction,
grounding it in numbers, and writing contract-compliant Python all at once:

1. **interpret** — instruction + contrast images + a compact summary -> a
   plain-language preference (no API, no numbers, no code).
2. **ground** — that preference + the full numeric summaries -> a concrete numeric
   specification of features and bounds (no images, no code).
3. **author** — that specification + the runtime API and output contract -> the final
   cost JSON (no images, no interpretation).

Each stage's raw text is chained into the next; only the author stage is parsed and
compiled into a :class:`GeneratedPythonCost`. Like the ``llm`` backend this is a
single pass with no rollout feedback loop.
"""

from __future__ import annotations

from uncertain_feedback.planners.mpc.costs.cost_generator import CostGenerator
from uncertain_feedback.planners.mpc.costs.generated import GeneratedPythonCost
from uncertain_feedback.planners.mpc.costs.prompts import (
    build_author_prompt,
    build_ground_prompt,
    build_interpret_prompt,
)


class StagedCostGenerator(CostGenerator):
    """Cost generator that chains interpret -> ground -> author LLM calls."""

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            llm = self.make_llm()

            interpret_text, image_paths = build_interpret_prompt(
                self.instruction, self.summaries, self.images if self.use_images else {}
            )
            image_input = [str(path) for path in image_paths] or None
            interpretation = self._stage(
                llm, "interpret", interpret_text, image_input=image_input
            )

            ground_text = build_ground_prompt(interpretation, self.summaries)
            specification = self._stage(llm, "ground", ground_text)

            author_text = build_author_prompt(specification)
            raw = self._stage(llm, "author", author_text)

            response, cost = self.parse_cost(raw)
            self.save_response(response)
            if install:
                self.install(cost)
            self._on_success(cost, installed=install)
            return cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None

    def _stage(
        self,
        llm: object,
        name: str,
        prompt_text: str,
        *,
        image_input: list[str] | None = None,
    ) -> str:
        """Run one stage, persist its prompt and raw response, and return the text."""
        (self.run_dir / f"{name}_prompt.txt").write_text(prompt_text, encoding="utf-8")
        raw = llm.get_full_output(prompt_text, image_input=image_input)  # type: ignore[attr-defined]
        (self.run_dir / f"{name}_response.txt").write_text(raw, encoding="utf-8")
        return raw
