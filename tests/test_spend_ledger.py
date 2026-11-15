"""The spend ledger prices what the API reports and stops at the budget."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from uncertain_feedback.llm.spend import SpendLedger


def test_record_prices_both_usage_shapes_and_persists(tmp_path: Path) -> None:
    ledger = SpendLedger(
        tmp_path / "api_cost.jsonl", budget=1.0, price_in=1e-6, price_out=2e-6
    )

    ledger.record(SimpleNamespace(input_tokens=1000, output_tokens=100))
    assert ledger.costs == [1000 * 1e-6 + 100 * 2e-6]
    ledger.record(SimpleNamespace(prompt_tokens=500, completion_tokens=50))

    lines = [
        json.loads(line)
        for line in (tmp_path / "api_cost.jsonl").read_text().splitlines()
    ]
    assert [(row["in"], row["out"]) for row in lines] == [(1000, 100), (500, 50)]
    reopened = SpendLedger(
        tmp_path / "api_cost.jsonl", budget=1.0, price_in=1e-6, price_out=2e-6
    )
    assert reopened.total == ledger.total
    assert not reopened.exhausted


def test_the_budget_is_exhausted_once_the_total_reaches_it(tmp_path: Path) -> None:
    ledger = SpendLedger(
        tmp_path / "ledger.jsonl", budget=0.002, price_in=0.001, price_out=0.0
    )
    assert not ledger.exhausted
    ledger.record(SimpleNamespace(input_tokens=1, output_tokens=0))
    assert not ledger.exhausted
    ledger.record(SimpleNamespace(input_tokens=1, output_tokens=0))
    assert ledger.exhausted


def test_meter_hooks_the_model(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl", budget=1.0)
    model: Any = SimpleNamespace(usage_hook=None)

    assert ledger.meter(cast(Any, model)) is model
    model.usage_hook(SimpleNamespace(input_tokens=10, output_tokens=10))
    assert len(ledger.costs) == 1
