"""Tests for the pure-agent grounder (LLM-written trajectories)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.approaches import Approach, NoCostGen
from evaluation.approaches.grounders.llm_trajectory import (
    LlmTrajectoryGrounder,
    _interpolate,
    feature_rows,
)
from evaluation.benchmarks.base import InteractionBenchmark
from evaluation.benchmarks.episode import run_episode
from evaluation.benchmarks.structs import InteractionTask
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import get_persona

_SMOKE_MPC = (
    Path(__file__).resolve().parents[1] / "evaluation" / "conf" / "mpc_smoke.yaml"
)
# The coupled-bound persona: its comfortable elbow flexion falls as the arm is
# raised, so this goal makes the plan violate it only near the end (step 15/24).
_PERSONA = "triceps_long_head_contracture"
_GOAL = np.array([0.25, 0.32, 0.15])
_ELBOW, _WRIST = 3, 4


class _FakeModel:
    """Stand-in for ``OpenAIModel`` replaying canned responses."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get_full_output(self, text_input: str, image_input: Any = None) -> str:
        del text_input, image_input
        self.calls += 1
        return self.responses[min(self.calls, len(self.responses)) - 1]


class _Selector:
    """Records how often the harness's cluster selector is invoked."""

    def __init__(self) -> None:
        self.calls: list[dict[int, np.ndarray]] = []

    def __call__(self, candidates: dict[int, np.ndarray]) -> tuple[int, float]:
        self.calls.append(candidates)
        return min(candidates), 1.0


def _setup() -> tuple[MpcRunConfig, Human]:
    cfg = replace(load_mpc_config(_SMOKE_MPC), seed=0)
    return cfg, Human(pose=cfg.pose, arm=cfg.arm)


def _smoke_task(cfg: MpcRunConfig) -> InteractionTask:
    bench = InteractionBenchmark(
        name="smoke",
        personas=[_PERSONA],
        verbalizers=["joint_resolved"],
        goals=[list(_GOAL)],
        max_rounds=1,
    )
    return bench.generate_tasks(0, cfg)[0]


def _bind(
    grounder: LlmTrajectoryGrounder, cfg: MpcRunConfig, tmp_path: Path, *responses: str
) -> _FakeModel:
    grounder.reset(cfg, None, get_persona(_PERSONA), _smoke_task(cfg).seed, tmp_path)
    model = _FakeModel(*responses)
    grounder._llm = model
    return model


def _nominal_plan(human: Human, n_frames: int = 21) -> np.ndarray:
    """A stand-in for the harness's nominal continuation from ``human.q``."""
    ramp = np.linspace(0.0, 1.0, n_frames)[:, None]
    delta = np.array([0.0, 0.0, 0.0, 0.1, 0.2, -0.1, 0.3])
    return human.q[None] + ramp * delta


def _position_rows(human: Human, q: np.ndarray) -> np.ndarray:
    arm = human.fk_positions_from_q(q)
    return np.concatenate([arm[:, _ELBOW], arm[:, _WRIST]], axis=1)


def _response(key: str, rows: list[list[float]], count: int) -> str:
    return json.dumps(
        {
            "interpretations": [
                {"interpretation": f"reading {i}", key: rows} for i in range(count)
            ],
            "reply": "moving your arm now",
        }
    )


def test_dense_position_frames_become_four_candidates(tmp_path: Path) -> None:
    cfg, human = _setup()
    grounder = LlmTrajectoryGrounder(output_space="positions", n_frames=16)
    target = human.q
    target[3:6] += 0.3
    rows = _position_rows(human, np.linspace(human.q, target, 16)).tolist()
    _bind(grounder, cfg, tmp_path, _response("frames", rows, 4))
    selector = _Selector()

    result = grounder.ground("lift it higher", human, _nominal_plan(human), selector)

    assert len(result.candidates) == 4
    assert len(selector.calls) == 1
    assert all(traj.shape == (16, 3, 3) for traj in result.candidates.values())
    assert result.correction_traj.shape == result.candidates[result.chosen_label].shape
    assert (tmp_path / "interpretations_00.json").exists()


def test_anatomical_frames_reproduce_the_requested_angles(tmp_path: Path) -> None:
    cfg, human = _setup()
    grounder = LlmTrajectoryGrounder(
        output_space="anatomical", n_frames=12, n_interpretations=2
    )
    target = human.q
    target[3:6] += 0.4
    target[6] += 0.5
    rows = feature_rows(np.linspace(human.q, target, 12), human)
    _bind(grounder, cfg, tmp_path, _response("frames", rows.tolist(), 2))

    result = grounder.ground(
        "bend my elbow more", human, _nominal_plan(human), _Selector()
    )

    assert len(result.candidates) == 2
    np.testing.assert_allclose(
        feature_rows(result.candidates[0], human), rows, atol=1e-9
    )


def test_interpolation_passes_through_every_waypoint_row() -> None:
    start = np.zeros(6)
    waypoints = np.array([[1.0] * 6, [2.0] * 6, [3.0] * 6])

    path = _interpolate(start, waypoints, 13)

    assert path.shape == (13, 6)
    for index, knot in enumerate([start, *waypoints]):
        np.testing.assert_allclose(path[index * 4], knot, atol=1e-12)


def test_single_waypoint_lands_the_arm_on_the_waypoint(tmp_path: Path) -> None:
    cfg, human = _setup()
    grounder = LlmTrajectoryGrounder(n_waypoints=1, n_frames=8, n_interpretations=1)
    target = human.q
    target[3:6] += 0.25
    target[6] += 0.4
    waypoint = _position_rows(human, target[None])
    _bind(grounder, cfg, tmp_path, _response("waypoints", waypoint.tolist(), 1))

    result = grounder.ground("stop there", human, _nominal_plan(human), _Selector())

    reached = human.fk.fk(result.candidates[0][-1], human.spine3_pos, human.spine3_aa)
    np.testing.assert_allclose(reached[_ELBOW], waypoint[0, :3], atol=1e-9)
    np.testing.assert_allclose(reached[_WRIST], waypoint[0, 3:], atol=1e-9)


def test_unparseable_response_falls_back_to_the_nominal_plan(tmp_path: Path) -> None:
    cfg, human = _setup()
    grounder = LlmTrajectoryGrounder()
    _bind(grounder, cfg, tmp_path, "sorry, I cannot help with that")
    nominal = _nominal_plan(human)

    result = grounder.ground("move it", human, nominal, _Selector())

    assert len(result.candidates) == 1
    np.testing.assert_allclose(
        result.candidates[0], human.arm_aa_from_q(nominal), atol=1e-12
    )


def test_agent_waypoint_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end with a stubbed interpretation call."""
    cfg, human = _setup()
    grounder = LlmTrajectoryGrounder(n_waypoints=1, n_frames=8)
    approach = Approach(name="agent_waypoint", grounder=grounder, cost_gen=NoCostGen())
    target = human.q
    target[3:6] += 0.2
    rows = _position_rows(human, target[None]).tolist()
    episode_dir = tmp_path / "episode"
    approach.reset(
        cfg, human, None, get_persona(_PERSONA), _smoke_task(cfg).seed, episode_dir
    )
    grounder._llm = _FakeModel(_response("waypoints", rows, 4))

    result = run_episode(
        cfg, human, get_persona(_PERSONA), _smoke_task(cfg), approach, episode_dir
    )

    assert (episode_dir / "episode_summary.json").exists()
    assert result, "episode recorded no interactions"
    assert len(result[0].rounds[0].grounding.candidates) == 4
