"""The tempo measures and the pre-registered speed criteria on synthetic wrists."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import tempo


def _reach(t_done: int, length: float = 0.2, frames: int = 49) -> np.ndarray:
    """One wrist path moving along x at a constant speed until ``t_done``, then holding."""
    progress = np.minimum(np.arange(frames), t_done) / t_done
    return np.stack([progress * length, np.zeros(frames), np.zeros(frames)], axis=-1)


def test_t90_is_when_the_motion_is_nearly_done_and_ignores_scale() -> None:
    wrist = np.stack([_reach(20), _reach(40), 0.3 * _reach(20)])
    t90 = tempo.time_to_fraction(wrist, tempo.T90_FRACTION)
    assert t90.tolist() == [18, 36, 18]
    assert tempo.time_to_fraction(wrist, tempo.T50_FRACTION).tolist() == [10, 20, 10]


def test_net_vector_speed_and_rush() -> None:
    wrist = np.stack([_reach(20, length=0.2)])
    np.testing.assert_allclose(tempo.net_vector(wrist), [[0.2, 0.0, 0.0]])
    np.testing.assert_allclose(tempo.peak_speed(wrist), [0.01])
    np.testing.assert_allclose(tempo.rush(wrist), [0.01])
    stats = tempo.sample_stats(wrist)
    assert stats["t90"] == 18.0 and stats["D"] == 0.2
    np.testing.assert_allclose(stats["v"], [0.2, 0.0, 0.0])


def test_condition_stats_average_the_prompts() -> None:
    slow = tempo.sample_stats(np.stack([_reach(40)]))
    slower = tempo.sample_stats(np.stack([_reach(44)]))
    merged = tempo.condition_stats([slow, slower])
    assert merged["t90"] == (36 + 40) / 2
    np.testing.assert_allclose(merged["v"], [0.2, 0.0, 0.0])


def _rows(gap: int, n: int = 30) -> list[dict[str, float]]:
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(n):
        jitter = rng.integers(-2, 3)
        unmarked = tempo.sample_stats(np.stack([_reach(24 + jitter)]))
        slow = tempo.sample_stats(np.stack([_reach(24 + gap + jitter)]))
        fast = tempo.sample_stats(np.stack([_reach(24 - gap + jitter)]))
        rows.append(tempo.case_row(unmarked, slow, fast))
    return rows


def test_a_clear_tempo_gap_passes_and_no_gap_fails() -> None:
    passing = tempo.speed_verdict(_rows(gap=6))
    assert passing["pass"] and passing["s1_gap"]["mean"] > tempo.S1_MIN_GAP_FRAMES
    assert passing["s3_cos_slow"] == 1.0 and passing["s4_d_ratio_slow"] == 1.0
    failing = tempo.speed_verdict(_rows(gap=0))
    assert not failing["S1"] and not failing["S2"] and not failing["pass"]
    assert "PASS on 30 cases" in tempo.verdict_table(passing)


def test_a_slow_copy_that_merely_shrinks_fails_s4_and_a_reversed_one_s3() -> None:
    unmarked = tempo.sample_stats(np.stack([_reach(24)]))
    shrunk = tempo.sample_stats(np.stack([0.1 * _reach(24)]))
    reversed_ = tempo.sample_stats(np.stack([-_reach(40)]))
    fast = tempo.sample_stats(np.stack([_reach(12)]))
    row = tempo.case_row(unmarked, shrunk, fast)
    assert row["d_ratio_slow"] == pytest.approx(0.1)
    assert row["t90_slow"] == row["t90_unmarked"]
    assert tempo.case_row(unmarked, reversed_, fast)["cos_slow"] == pytest.approx(-1.0)


def test_case_bootstrap_brackets_the_mean() -> None:
    ci = tempo.case_bootstrap(np.arange(40, dtype=float), n_boot=2000)
    assert ci["lo"] < ci["mean"] == 19.5 < ci["hi"]
