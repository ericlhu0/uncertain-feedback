from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC
from uncertain_feedback.planners.mpc.costs import (
    CombineCostGenerator,
    CompositeTrajectoryCost,
    CostRound,
    GeneratedPythonCost,
    JointLimitCost,
    MpcCostContext,
    build_generated_cost_context,
    replace_generated_costs,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_CODE = (
    "def cost(q_trajs, context, params):\n"
    "    return params['weight'] * np.mean(q_trajs[:, 1:] ** 2, axis=(1, 2, 3))\n"
)


def _context():
    fk = SmplLeftArmFK()
    mpc_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    return build_generated_cost_context(
        mpc_context,
        current_q=np.zeros((3, 3)),
        mdm_traj=np.zeros((3, 3, 3)),
        q_history=[],
        window=3,
    )


def _round(tmp_path: Path, index: int) -> CostRound:
    round_dir = tmp_path / f"round_{index:02d}"
    return CostRound(
        index=index,
        goal=(0.1 + index, 0.2, 0.3),
        feedback_text=f"feedback {index}",
        trigger_step=4 + index,
        round_dir=round_dir,
        state_path=round_dir / "state.pkl",
        cost_code=_CODE,
        params={"weight": float(index + 1)},
        summaries={"round": index},
        image_paths=(round_dir / "image.png",),
    )


def test_replace_generated_costs_drops_all_generated_terms() -> None:
    context = _context()
    base = JointLimitCost(
        slots=(0,), low=np.full((1, 3), -1.0), high=np.full((1, 3), 1.0)
    )
    old_a = GeneratedPythonCost(_CODE, {"weight": 1.0}, context)
    old_b = GeneratedPythonCost(_CODE, {"weight": 2.0}, context)
    new = GeneratedPythonCost(_CODE, {"weight": 3.0}, context)
    composite = CompositeTrajectoryCost([base, old_a, old_b])

    assert replace_generated_costs(composite, new).terms() == (base, new)
    assert replace_generated_costs(composite, None).terms() == (base,)


def test_cost_round_json_round_trip_uses_absolute_paths(tmp_path) -> None:
    round_ = _round(tmp_path, 1)

    restored = CostRound.from_json(round_.to_json())

    assert restored == CostRound(
        **{
            **round_.__dict__,
            "round_dir": round_.round_dir.resolve(),
            "state_path": round_.state_path.resolve(),
            "image_paths": tuple(path.resolve() for path in round_.image_paths),
        }
    )


def test_combine_generator_writes_per_round_scores_and_replaces(
    tmp_path, monkeypatch
) -> None:
    context = _context()
    rounds = [_round(tmp_path, 0), _round(tmp_path, 1)]
    run_dir = tmp_path / "combine"
    base = JointLimitCost(
        slots=(0,), low=np.full((1, 3), -1.0), high=np.full((1, 3), 1.0)
    )
    old = GeneratedPythonCost(_CODE, {"weight": 9.0}, context)
    mpc = SmplLeftArmMPC(
        goals=[np.zeros((3, 3))], extra_costs=CompositeTrajectoryCost([base, old])
    )
    generator = CombineCostGenerator(
        context=context,
        instruction="feedback",
        summaries={},
        run_dir=run_dir,
        use_images=True,
        mpc=mpc,
        rounds=rounds,
    )

    def fake_codex() -> None:
        response = {
            "description": "combined",
            "code": _CODE,
            "params": {"weight": 4.0},
            "explanation": "",
            "recipient_explanation": "",
        }
        (run_dir / "response.json").write_text(json.dumps(response), encoding="utf-8")
        (run_dir / "stage_log.md").write_text(
            "## Evidence synthesis\n", encoding="utf-8"
        )
        (run_dir / "ITERATION_LOG.md").write_text(
            "all rounds checked\n", encoding="utf-8"
        )
        (run_dir / "codex.log").write_text("fake codex\n", encoding="utf-8")

    monkeypatch.setattr(generator, "_save_reference_video", lambda: None)
    monkeypatch.setattr(generator, "_run_codex", fake_codex)
    monkeypatch.setattr(
        generator,
        "_score_rounds",
        lambda *_args: {"per_round": {"0": 1.0, "1": 2.0}, "mean": 1.5},
    )

    combined = generator.generate(install=True)

    assert combined is not None
    assert json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))[
        "per_round"
    ] == {"0": 1.0, "1": 2.0}
    assert mpc._extra_costs.terms() == (base, combined)
