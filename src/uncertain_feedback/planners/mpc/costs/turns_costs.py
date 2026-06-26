"""Multi-turn (``turns``) cost generator.

Holds a real conversation with the LLM (a growing message list, not stateless
re-prompting): each turn it parses the returned cost, scores it with the
evaluation harness, and feeds that score back, keeping the best-scoring cost.
"""

from __future__ import annotations

import json
import math
from typing import Any

from uncertain_feedback.planners.mpc.costs.cost_generator import (
    CostGenerator,
    evaluate_candidate_cost,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
    LlmCostResponse,
)

# Stop early once the score fails to improve this many turns in a row.
_NO_IMPROVE_PATIENCE = 2


class TurnsCostGenerator(CostGenerator):
    """Iterative cost generator driving a stateful LLM conversation."""

    def __init__(self, *args: Any, max_turns: int = 6, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_turns = max(1, int(max_turns))

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            best = self._converse()
            if best is None:
                raise GeneratedCostValidationError(
                    "no valid cost produced across turns"
                )
            best_cost, best_response = best
            self.save_response(best_response)
            if install:
                self.install(best_cost)
            self._on_success(best_cost, installed=install)
            return best_cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None

    def _converse(
        self,
    ) -> tuple[GeneratedPythonCost, LlmCostResponse] | None:
        prompt_text, image_input = self.build_prompt()
        (self.run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        llm = self.make_llm()
        messages: list[dict[str, Any]] = [
            {"role": "user", "text": prompt_text, "images": image_input}
        ]

        best: tuple[GeneratedPythonCost, LlmCostResponse] | None = None
        best_score = math.inf
        no_improve = 0

        for turn in range(self.max_turns):
            raw = llm.converse(messages)
            messages.append({"role": "assistant", "text": raw})
            turn_dir = self.run_dir / f"turn_{turn}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            (turn_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

            try:
                response, cost = self.parse_cost(raw)
            except GeneratedCostValidationError as exc:
                messages.append(
                    {
                        "role": "user",
                        "text": (
                            f"That response was not a usable cost: {exc}. "
                            "Return a corrected JSON object following the same "
                            "contract."
                        ),
                    }
                )
                continue

            score = evaluate_candidate_cost(self.context, cost)
            (turn_dir / "cost.py").write_text(response.code, encoding="utf-8")
            with open(turn_dir / "score.json", "w", encoding="utf-8") as f:
                json.dump({"turn": turn, "score": score}, f, indent=2)
            print(f"[cost-gen][turns] turn {turn}: score={score:.4f}")

            if score < best_score:
                best_score = score
                best = (cost, response)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= _NO_IMPROVE_PATIENCE:
                    break

            messages.append(
                {
                    "role": "user",
                    "text": (
                        f"That cost scored {score:.4f} (lower is better, where the "
                        "score measures how well the resulting motion matches the "
                        "user's correction). Revise the cost to lower the score and "
                        "return the JSON object again."
                    ),
                }
            )

        return best
