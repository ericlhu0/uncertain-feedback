"""Tests for the correction-progress metrics' net-change ``direction``."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import math

import numpy as np
import pytest

from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users.progress import correction_progress


@pytest.fixture(name="human")
def _human() -> Human:
    return Human()


def _q_traj(flexions: list[float] | np.ndarray) -> np.ndarray:
    q = np.zeros((len(flexions), 7), dtype=np.float64)
    q[:, 6] = flexions
    return q


def test_direction_is_one_when_the_net_change_matches(human: Human) -> None:
    oracle = _q_traj(np.linspace(0.2, 1.0, 21))
    generated = _q_traj(np.linspace(0.2, 0.6, 10))
    assert correction_progress(oracle, generated, human).direction == pytest.approx(1.0)


def test_direction_is_minus_one_when_the_net_change_opposes(human: Human) -> None:
    oracle = _q_traj(np.linspace(0.2, 1.0, 21))
    generated = _q_traj(np.linspace(0.6, 0.3, 10))
    result = correction_progress(oracle, generated, human)
    assert result.direction == pytest.approx(-1.0)


def test_direction_ignores_a_final_step_that_settles_back(human: Human) -> None:
    oracle = _q_traj(np.linspace(0.2, 1.0, 21))
    generated = _q_traj([*np.linspace(0.2, 0.8, 10), 0.799])
    result = correction_progress(oracle, generated, human)
    assert result.alignment < 0.0
    assert result.direction == pytest.approx(1.0)


def test_direction_is_zero_when_the_correction_does_not_move(human: Human) -> None:
    oracle = _q_traj(np.linspace(0.2, 1.0, 21))
    generated = _q_traj([0.5] * 10)
    assert correction_progress(oracle, generated, human).direction == 0.0


def test_direction_is_nan_when_the_oracle_barely_moves(human: Human) -> None:
    oracle = _q_traj(np.linspace(0.2, 0.3, 21))
    generated = _q_traj(np.linspace(0.2, 0.6, 10))
    assert math.isnan(correction_progress(oracle, generated, human).direction)
