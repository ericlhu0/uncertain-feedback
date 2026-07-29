"""Tests for the multi-round experiment driver."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import uncertain_feedback.experiments.multi_round_experiment as multi_round
from uncertain_feedback.experiments.experiment_pipeline import InitialRolloutResult
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    MpcCostContext,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import get_persona

_CODE = "def cost(q_trajs, context, params):\n    return np.zeros(q_trajs.shape[0])\n"


class _BaseCost:
    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        return np.zeros(q_trajs.shape[0])


class _EvalState:
    def save(self, path: Path) -> None:
        path.write_bytes(b"state")

    def make_rollout_fn(self):
        return None


def test_multi_round_loop_persists_history_and_replaces_costs(
    tmp_path, monkeypatch
) -> None:
    config_path = Path(
        "src/uncertain_feedback/planners/mpc/configs/"
        "mdm_llm_multiround.yaml"
    )
    cfg = load_mpc_config(config_path)
    cfg = replace(
        cfg,
        cartesian=replace(
            cfg.cartesian,
            goals=[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
        ),
    )
    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    generated_context = build_generated_cost_context(
        context,
        current_q=np.zeros((3, 3)),
        mdm_traj=np.zeros((3, 3, 3)),
        q_history=[],
        window=3,
    )
    direct_costs = [
        GeneratedPythonCost(_CODE, {"round": index}, generated_context)
        for index in range(2)
    ]
    combined_cost = GeneratedPythonCost(_CODE, {"combined": True}, generated_context)
    base = _BaseCost()
    mpc = SimpleNamespace(_extra_costs=CompositeTrajectoryCost([base]))
    installed_terms: list[tuple[object, ...]] = []
    initial_calls = 0

    def fake_initial(*args, **_kwargs):
        nonlocal initial_calls
        installed_terms.append(args[4].terms())
        index = initial_calls
        initial_calls += 1
        trajectory = np.zeros((4, 3, 3))
        if index == 2:
            return InitialRolloutResult(trajectory, None, None, None, [])
        return InitialRolloutResult(
            trajectory,
            trigger_step=2,
            trigger_violation=0.1,
            q_feedback=trajectory[2],
            q_history=[trajectory[0], trajectory[1], trajectory[2]],
        )

    generation_calls = 0

    def fake_generation(**_kwargs):
        nonlocal generation_calls
        cost = direct_costs[generation_calls]
        generation_calls += 1
        return SimpleNamespace(
            generated_cost=cost,
            eval_state=_EvalState(),
            generated_context=generated_context,
            summaries={"round": generation_calls},
            images={},
            description="generated cost",
            explanation="why",
            interpretation="preference",
            grounding="threshold",
        )

    combine_round_counts: list[int] = []

    class _Combinator:
        def __init__(self, **kwargs) -> None:
            combine_round_counts.append(len(kwargs["rounds"]))

        def generate(self, install: bool = False):
            assert not install
            return combined_cost

    monkeypatch.setattr(
        multi_round, "artifact_run_dir", lambda *_args: tmp_path / "run"
    )
    monkeypatch.setattr(multi_round, "run_initial_rollout", fake_initial)
    monkeypatch.setattr(
        multi_round,
        "generate_uq_correction",
        lambda *_args, **_kwargs: SimpleNamespace(
            correction_traj=np.zeros((3, 3, 3)),
            uq_result=SimpleNamespace(
                cluster_means={0: np.zeros((3, 3, 3))}, chosen_label=0
            ),
        ),
    )
    monkeypatch.setattr(multi_round, "generate_cost_for_cluster", fake_generation)
    monkeypatch.setattr(multi_round, "CombineCostGenerator", _Combinator)
    monkeypatch.setattr(
        multi_round, "evaluate_cost_conditions", lambda *_args, **_kwargs: {"ok": {}}
    )

    summary = multi_round.run_multi_round_experiment(
        mpc,  # type: ignore[arg-type]
        cfg,
        get_persona("stroke_flexor_synergy"),
        SimpleNamespace(),  # type: ignore[arg-type]
        np.zeros(263),
        np.zeros((3, 3)),
        context,
        tmp_path,
        body_pos=None,
        spine3_pos=None,
        spine3_aa=None,
    )

    assert combine_round_counts == [2]
    assert installed_terms == [(base,), (base, direct_costs[0]), (base, combined_cost)]
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert [round_["index"] for round_ in history] == [0, 1]
    assert summary["feedback_round_count"] == 2
    assert summary["unified_cost"]["params"] == {"combined": True}
