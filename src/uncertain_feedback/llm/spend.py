"""Meter an LLM's API spend from the usage the API reports, under a hard budget.

The shell exports ``OPENAI_API_KEY``, so a paid step bills silently; anything that
runs many completions goes through a :class:`SpendLedger`, which prices every
response's token usage, appends it to a ledger file (so a restart carries the
spend across) and says when the budget is gone. The caller checks
:attr:`SpendLedger.exhausted` before each call; the ledger itself never raises
mid-request.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from uncertain_feedback.llm.openai_model import OpenAIModel

# $ per token for the luna models, input at the uncached price (the rate
# autolabel_capped.py metered round 23 with, 2026-11-01).
LUNA_PRICE_IN = 0.20e-6
LUNA_PRICE_OUT = 1.20e-6


def _tokens(usage: Any) -> tuple[int, int]:
    """Input and output token counts of a Responses or Chat Completions usage object."""
    tokens_in = getattr(usage, "input_tokens", None)
    if tokens_in is None:
        tokens_in = usage.prompt_tokens
    tokens_out = getattr(usage, "output_tokens", None)
    if tokens_out is None:
        tokens_out = usage.completion_tokens
    return int(tokens_in), int(tokens_out)


class SpendLedger:
    """Price every metered response and stop the caller at ``budget`` dollars."""

    def __init__(
        self,
        path: Path,
        budget: float,
        price_in: float = LUNA_PRICE_IN,
        price_out: float = LUNA_PRICE_OUT,
    ) -> None:
        self.path = path
        self.budget = budget
        self.price_in, self.price_out = price_in, price_out
        self._lock = threading.Lock()
        self.costs: list[float] = (
            [json.loads(line)["usd"] for line in path.read_text().splitlines() if line]
            if path.exists()
            else []
        )

    @property
    def total(self) -> float:
        """Dollars spent so far, this run and the ledger's earlier ones."""
        return sum(self.costs)

    @property
    def exhausted(self) -> bool:
        """Whether the budget is spent; check before every call."""
        return self.total >= self.budget

    def record(self, usage: Any) -> None:
        """Price one response's usage and append it to the ledger."""
        tokens_in, tokens_out = _tokens(usage)
        usd = tokens_in * self.price_in + tokens_out * self.price_out
        with self._lock:
            self.costs.append(usd)
            with self.path.open("a", encoding="utf-8") as ledger:
                ledger.write(
                    json.dumps({"in": tokens_in, "out": tokens_out, "usd": usd}) + "\n"
                )

    def meter(self, model: OpenAIModel) -> OpenAIModel:
        """Record every response ``model`` receives from here on."""
        model.usage_hook = self.record
        return model
