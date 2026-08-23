"""Tests for the pure-agent grounder (LLM-written trajectories)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
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
from evaluation.episode import run_episode
from evaluation.rig import EvalRig, build_rig
from evaluation.structs import InteractionTask
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
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


def _smoke_task(rig: EvalRig) -> InteractionTask:
    bench = InteractionBenchmark(
        name="smoke",
        personas=[_PERSONA],
        verbalizers=["joint_resolved"],
        goals=[list(_GOAL)],
        max_rounds=1,
    )
    return bench.generate_tasks(0, rig.cfg)[0]


def _bind(
    grounder: LlmTrajectoryGrounder, rig: EvalRig, tmp_path: Path, *responses: str
) -> _FakeModel:
    grounder.reset(rig, get_persona(_PERSONA), _smoke_task(rig), tmp_path)
    model = _FakeModel(*responses)
    grounder._llm = model
    return model


def _nominal_plan(rig: EvalRig, n_frames: int = 21) -> np.ndarray:
    """A stand-in for the harness's nominal continuation from ``rig.q0``."""
    ramp = np.linspace(0.0, 1.0, n_frames)[:, None]
    delta = np.array([0.0, 0.0, 0.0, 0.1, 0.2, -0.1, 0.3])
    return np.asarray(rig.q0, dtype=np.float64)[None] + ramp * delta


def _positions(rig: EvalRig, q: np.ndarray) -> np.ndarray:
    return rig.fk.fk_batch(
        q_to_arm_aa(q, rig.fk.elbow_hinge_axis), rig.spine3_pos, rig.spine3_aa
    )


def _position_rows(rig: EvalRig, q: np.ndarray) -> np.ndarray:
    arm = _positions(rig, q)
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
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    grounder = LlmTrajectoryGrounder(output_space="positions", n_frames=16)
    target = np.asarray(rig.q0, dtype=np.float64).copy()
    target[3:6] += 0.3
    rows = _position_rows(rig, np.linspace(rig.q0, target, 16)).tolist()
    _bind(grounder, rig, tmp_path, _response("frames", rows, 4))
    selector = _Selector()

    result = grounder.ground(
        "lift it higher", np.asarray(rig.q0), _nominal_plan(rig), selector, _GOAL
    )

    assert len(result.candidates) == 4
    assert len(selector.calls) == 1
    assert all(traj.shape == (16, 3, 3) for traj in result.candidates.values())
    assert result.correction_traj.shape == result.candidates[result.chosen_label].shape
    assert (tmp_path / "interpretations_00.json").exists()


def test_anatomical_frames_reproduce_the_requested_angles(tmp_path: Path) -> None:
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    grounder = LlmTrajectoryGrounder(
        output_space="anatomical", n_frames=12, n_interpretations=2
    )
    target = np.asarray(rig.q0, dtype=np.float64).copy()
    target[3:6] += 0.4
    target[6] += 0.5
    rows = feature_rows(np.linspace(rig.q0, target, 12), rig.context)
    _bind(grounder, rig, tmp_path, _response("frames", rows.tolist(), 2))

    result = grounder.ground(
        "bend my elbow more", np.asarray(rig.q0), _nominal_plan(rig), _Selector(), _GOAL
    )

    assert len(result.candidates) == 2
    np.testing.assert_allclose(
        feature_rows(result.candidates[0], rig.context), rows, atol=1e-9
    )


def test_interpolation_passes_through_every_waypoint_row() -> None:
    start = np.zeros(6)
    waypoints = np.array([[1.0] * 6, [2.0] * 6, [3.0] * 6])

    path = _interpolate(start, waypoints, 13)

    assert path.shape == (13, 6)
    for index, knot in enumerate([start, *waypoints]):
        np.testing.assert_allclose(path[index * 4], knot, atol=1e-12)


def test_single_waypoint_lands_the_arm_on_the_waypoint(tmp_path: Path) -> None:
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    grounder = LlmTrajectoryGrounder(n_waypoints=1, n_frames=8, n_interpretations=1)
    target = np.asarray(rig.q0, dtype=np.float64).copy()
    target[3:6] += 0.25
    target[6] += 0.4
    waypoint = _position_rows(rig, target[None])
    _bind(grounder, rig, tmp_path, _response("waypoints", waypoint.tolist(), 1))

    result = grounder.ground(
        "stop there", np.asarray(rig.q0), _nominal_plan(rig), _Selector(), _GOAL
    )

    reached = rig.fk.fk(result.candidates[0][-1], rig.spine3_pos, rig.spine3_aa)
    np.testing.assert_allclose(reached[_ELBOW], waypoint[0, :3], atol=1e-9)
    np.testing.assert_allclose(reached[_WRIST], waypoint[0, 3:], atol=1e-9)


def test_unparseable_response_falls_back_to_the_nominal_plan(tmp_path: Path) -> None:
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    grounder = LlmTrajectoryGrounder()
    _bind(grounder, rig, tmp_path, "sorry, I cannot help with that")
    nominal = _nominal_plan(rig)

    result = grounder.ground("move it", np.asarray(rig.q0), nominal, _Selector(), _GOAL)

    assert len(result.candidates) == 1
    np.testing.assert_allclose(
        result.candidates[0], q_to_arm_aa(nominal, rig.fk.elbow_hinge_axis), atol=1e-12
    )


def test_agent_waypoint_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end with a stubbed interpretation call."""
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    grounder = LlmTrajectoryGrounder(n_waypoints=1, n_frames=8)
    approach = Approach(
        name="agent_waypoint", grounder=grounder, cost_gen=NoCostGen()
    )
    target = np.asarray(rig.q0, dtype=np.float64).copy()
    target[3:6] += 0.2
    rows = _position_rows(rig, target[None]).tolist()
    episode_dir = tmp_path / "episode"
    approach.reset(rig, get_persona(_PERSONA), _smoke_task(rig), episode_dir)
    grounder._llm = _FakeModel(_response("waypoints", rows, 4))

    result = run_episode(
        rig, get_persona(_PERSONA), _smoke_task(rig), approach, episode_dir
    )

    assert (episode_dir / "episode_summary.json").exists()
    assert result["summary"]["goal_results"], "episode recorded no goal results"
    assert result["rows"][0]["n_candidates"] == 4
