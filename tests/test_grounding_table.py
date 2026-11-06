"""The seed-pooled grounding comparison skips undefined cases and pairs by label."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.paper_tables import grounding_table


def _frame(
    labels: list[str], direction: list[float], acceptable: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direction": direction,
            "arc_progress": [0.5] * len(labels),
            "violation": [0.0] * len(labels),
            "acceptable": acceptable,
            "position_diversity": [0.7] * len(labels),
        },
        index=pd.Index(labels, name="label"),
    )


def test_pool_seeds_averages_defined_values_on_shared_cases() -> None:
    seed0 = _frame(["a", "b", "c"], [0.2, np.nan, 0.6], [1.0, 0.0, 1.0])
    seed1 = _frame(["a", "b"], [0.4, 0.8], [0.0, 0.0])

    pooled = grounding_table.pool_seeds([seed0, seed1])

    assert list(pooled.index) == ["a", "b"]
    assert pooled.loc["a", "direction"] == pytest.approx(0.3)
    assert pooled.loc["b", "direction"] == pytest.approx(0.8)
    assert pooled.loc["a", "acceptable"] == pytest.approx(0.5)


def test_compare_reports_the_paired_difference_over_defined_cases() -> None:
    reference = _frame(["a", "b", "c"], [0.1, 0.1, np.nan], [1.0, 1.0, 1.0])
    ours = _frame(["a", "b", "c"], [0.3, 0.5, 0.9], [1.0, 0.0, 1.0])

    lines = grounding_table.compare(reference, ours, n_boot=50)

    assert lines[0].startswith(
        "direction           reference 0.100  ours 0.400  Δ +0.300"
    )
    assert "(n 2)" in lines[0]
    assert "acceptable          reference 1.000  ours 0.667  Δ -0.333" in lines[3]


def test_load_run_reads_the_stored_columns(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "label": ["case_000", "case_001"],
            "chosen_label": [0, 1],
            "magnitude": [1.0, 1.5],
            "no_acceptable_cluster": [False, True],
            "chosen_direction": [0.2, -0.1],
            "chosen_arc_progress": [0.6, 0.3],
            "chosen_violation": [0.0, 0.02],
            "position_diversity": [0.7, 0.8],
        }
    ).to_csv(tmp_path / "cases.csv", index=False)

    run = grounding_table.load_run(tmp_path)

    assert list(run.columns) == list(grounding_table.METRICS)
    assert run.loc["case_001", "acceptable"] == 0.0
    assert run.loc["case_000", "direction"] == pytest.approx(0.2)


def test_load_run_needs_a_body_to_recompute_direction(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "label": ["case_000"],
            "chosen_label": [0],
            "magnitude": [1.0],
            "no_acceptable_cluster": [False],
            "chosen_arc_progress": [0.6],
            "chosen_violation": [0.0],
            "position_diversity": [0.7],
        }
    ).to_csv(tmp_path / "cases.csv", index=False)

    with pytest.raises(ValueError):
        grounding_table.load_run(tmp_path)
