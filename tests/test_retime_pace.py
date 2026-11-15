"""Retime pace draws: within the range, on the asked side, clear of the usual band."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np

from uncertain_feedback.data_collection.dataset_auto_correction.build_dataset import (
    _MAX_RETIME,
    retime_pace,
)
from uncertain_feedback.data_collection.dataset_auto_correction.tempo_captions import (
    tempo_lines,
)


def test_the_default_band_reproduces_the_original_draw() -> None:
    rng_new, rng_old = np.random.default_rng([42, 1]), np.random.default_rng([42, 1])
    for j in range(20):
        sign = 1 if j % 2 == 0 else -1
        old = float(np.exp(sign * rng_old.uniform(0, np.log(_MAX_RETIME))))
        assert retime_pace(rng_new, faster=j % 2 == 0) == old


def test_draws_stay_clear_of_the_usual_band() -> None:
    rng = np.random.default_rng(0)
    faster = [retime_pace(rng, True, (0.7, 1.4)) for _ in range(500)]
    slower = [retime_pace(rng, False, (0.7, 1.4)) for _ in range(500)]
    assert 1.4 <= min(faster) and max(faster) <= _MAX_RETIME
    assert 1 / _MAX_RETIME <= min(slower) and max(slower) <= 0.7
    # Log-uniform: the geometric middle of the band splits the draws about evenly.
    assert 0.4 < np.mean(np.array(faster) > np.sqrt(1.4 * _MAX_RETIME)) < 0.6
    assert 0.4 < np.mean(np.array(slower) < np.sqrt(0.7 / _MAX_RETIME)) < 0.6


def test_tempo_lines_add_or_replace() -> None:
    plain, rewrites = ["lift it", "lift it", "raise it"], ["lift it, slowly"]
    assert tempo_lines(plain, rewrites, keep_plain=False) == rewrites
    assert tempo_lines(plain, rewrites, keep_plain=True) == plain + rewrites
