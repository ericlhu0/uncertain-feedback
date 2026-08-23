"""Smoke tests for the evaluation harness (benchmarks x approaches x episode)."""

from pathlib import Path

import numpy as np

from evaluation.approaches import (
    Approach,
    BridgePotentialFieldGrounder,
    KeypointGrounder,
    NoCostGen,
    ParameterizedEditGrounder,
)
from evaluation.benchmarks.base import InteractionBenchmark
from evaluation.episode import run_episode
from evaluation.rig import build_rig
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.simulated_users import get_persona

_SMOKE_MPC = (
    Path(__file__).resolve().parents[1] / "evaluation" / "conf" / "mpc_smoke.yaml"
)


def test_benchmark_generates_persona_verbalizer_grid() -> None:
    """Tasks enumerate personas x verbalizers with resolved goal tuples."""
    cfg = load_mpc_config(_SMOKE_MPC)
    bench = InteractionBenchmark(
        name="grid",
        personas=["elbow_contracture", "painful_arc"],
        verbalizers=["vague", "everyday"],
        max_rounds=2,
    )
    tasks = bench.generate_tasks(3, cfg)
    assert len(tasks) == 4
    assert {task.persona for task in tasks} == {"elbow_contracture", "painful_arc"}
    assert all(task.goals == ((-0.18, 0.40, 0.34),) for task in tasks)
    assert all(task.seed == 3 for task in tasks)


def test_bridge_baseline_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end on CPU with the potential-field baseline."""
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    user = get_persona("elbow_contracture")
    bench = InteractionBenchmark(
        name="smoke",
        personas=["elbow_contracture"],
        verbalizers=["joint_resolved"],
        goals=[[-0.18, 0.40, 0.34]],
        max_rounds=1,
    )
    task = bench.generate_tasks(0, rig.cfg)[0]
    approach = Approach(
        name="bridge_baseline",
        grounder=BridgePotentialFieldGrounder(),
        cost_gen=NoCostGen(),
    )
    approach.reset(rig, user, task, tmp_path / "episode")
    result = run_episode(rig, user, task, approach, tmp_path / "episode")
    assert (tmp_path / "episode" / "episode_summary.json").exists()
    assert result["summary"]["goal_results"], "episode recorded no goal results"


def test_keypoint_baseline_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end with a stubbed keypoint interpreter."""
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    user = get_persona("elbow_contracture")
    bench = InteractionBenchmark(
        name="smoke",
        personas=["elbow_contracture"],
        verbalizers=["joint_resolved"],
        goals=[[-0.18, 0.40, 0.34]],
        max_rounds=1,
    )
    task = bench.generate_tasks(0, rig.cfg)[0]
    grounder = KeypointGrounder()
    grounder._interpret = (  # type: ignore[method-assign]
        lambda text, scene: {"joint": "wrist", "keypoint": np.array([0.1, 0.3, 0.2])}
    )
    approach = Approach(
        name="llm_keypoint", grounder=grounder, cost_gen=NoCostGen()
    )
    approach.reset(rig, user, task, tmp_path / "episode")
    result = run_episode(rig, user, task, approach, tmp_path / "episode")
    assert (tmp_path / "episode" / "episode_summary.json").exists()
    assert result["summary"]["goal_results"], "episode recorded no goal results"


def test_edit_baseline_episode_smoke(tmp_path: Path) -> None:
    """The episode loop runs end-to-end on CPU with the edit baseline."""
    rig = build_rig(_SMOKE_MPC, seed=0, load_generator=False)
    user = get_persona("elbow_contracture")
    bench = InteractionBenchmark(
        name="smoke",
        personas=["elbow_contracture"],
        verbalizers=["joint_resolved"],
        goals=[[-0.18, 0.40, 0.34]],
        max_rounds=1,
    )
    task = bench.generate_tasks(0, rig.cfg)[0]
    approach = Approach(
        name="edit_baseline",
        grounder=ParameterizedEditGrounder(),
        cost_gen=NoCostGen(),
    )
    approach.reset(rig, user, task, tmp_path / "episode")
    result = run_episode(rig, user, task, approach, tmp_path / "episode")
    assert (tmp_path / "episode" / "episode_summary.json").exists()
    summary = result["summary"]
    assert summary["persona"] == "elbow_contracture"
    assert summary["goal_results"], "episode recorded no goal results"
