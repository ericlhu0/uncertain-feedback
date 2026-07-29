"""Tests for folding per-round generated costs into one unified cost."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uncertain_feedback.planners.mpc import ArmMPC
from uncertain_feedback.planners.mpc.costs import (
    CombineCostGenerator,
    CompositeTrajectoryCost,
    CostRound,
    EvalState,
    GeneratedPythonCost,
    JointLimitCost,
    MpcCostContext,
    build_generated_cost_context,
    replace_generated_costs,
)
from uncertain_feedback.planners.mpc.costs.generated import GeneratedCostContext
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_CODE = (
    "def cost(q_trajs, context, params):\n"
    "    return params['weight'] * np.mean(q_trajs[:, 1:] ** 2, axis=(1, 2, 3))\n"
)


def _context() -> GeneratedCostContext:
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
    round_dir.mkdir()
    fk = SmplLeftArmFK()
    cost_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    full_cfg = SimpleNamespace(
        steps=3,
        horizon=2,
        n_mpc_samples=4,
        max_angle_delta=0.01,
        seed=0,
        cartesian=SimpleNamespace(goals=[[0.1, 0.2, 0.3]], threshold=0.05),
        constraints={},
        user=f"secret_persona_{index}",
        persona_goals={f"secret_persona_{index}": [[9.0, 9.0, 9.0]]},
    )
    state = EvalState(
        cfg=full_cfg,  # type: ignore[arg-type]
        current_q=np.zeros((3, 3)),
        correction_traj=np.zeros((2, 3, 3)),
        q_history=[],
        window=2,
        cost_context=cost_context,
        base_extra_costs=CompositeTrajectoryCost(),
        body_pos=None,
        spine3_pos=cost_context.spine3_pos,
        spine3_aa=cost_context.spine3_aa,
    )
    state.cfg = full_cfg  # type: ignore[assignment]
    with open(round_dir / "state.pkl", "wb") as file:
        pickle.dump(state, file)
    (round_dir / "image.png").write_bytes(b"image")
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
        trajectory_index=2,
        trigger_reason="text_time" if index == 0 else "discomfort",
        trigger_violation=None if index == 0 else 0.03,
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


def test_cost_round_loads_legacy_history_without_new_trigger_fields(tmp_path) -> None:
    data = _round(tmp_path, 0).to_json()
    data.pop("trajectory_index")
    data.pop("trigger_reason")
    data.pop("trigger_violation")

    restored = CostRound.from_json(data)

    assert restored.trajectory_index == 0
    assert restored.trigger_reason == "discomfort"
    assert restored.trigger_violation is None


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
    mpc = ArmMPC(extra_costs=CompositeTrajectoryCost([base, old]))
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
    task = (run_dir / "TASK.md").read_text(encoding="utf-8")
    assert str(tmp_path) not in task
    assert "/tmp/workspace/inputs/round_00/state.pkl" in task
    staged_state_path = run_dir / "inputs" / "round_00" / "state.pkl"
    staged_payload = staged_state_path.read_bytes()
    assert b"secret_persona_0" not in staged_payload
    assert b"persona_goals" not in staged_payload
    assert not hasattr(EvalState.load(staged_state_path).cfg, "user")


def _bwrap_usable() -> bool:
    """Whether bwrap can create a user namespace on this host.

    Ubuntu's apparmor_restrict_unprivileged_userns denies it to unconfined
    binaries ("setting up uid map: Permission denied"), which no amount of
    bwrap flags can work around.
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return False
    try:
        probe = subprocess.run(
            [bwrap, "--ro-bind", "/", "/", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


_requires_bwrap = pytest.mark.skipif(
    not _bwrap_usable(),
    reason="bwrap cannot create a user namespace on this host",
)


@_requires_bwrap
def test_combine_codex_streams_output_to_console_and_log(tmp_path, capsys) -> None:
    run_dir = tmp_path / "combine"
    run_dir.mkdir()
    message = "combiner reasoning summary"
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(f'print({message!r}, flush=True)')}"
    )
    generator = CombineCostGenerator(
        context=_context(),
        instruction="feedback",
        summaries={},
        run_dir=run_dir,
        codex_cmd=command,
        rounds=[],
    )

    generator._run_codex()

    assert f"[cost-gen][combine] codex output:\n{message}" in capsys.readouterr().out
    assert message in (run_dir / "codex.log").read_text(encoding="utf-8")


@_requires_bwrap
def test_agent_sandbox_hides_oracle_and_prior_runs(tmp_path) -> None:
    secret_session_name = "hidden_persona_session"
    run_dir = tmp_path / secret_session_name / "combine"
    run_dir.mkdir(parents=True)
    marker = run_dir / "visible.txt"
    marker.write_text("visible", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    oracle = repo / "src" / "uncertain_feedback" / "simulated_users" / "personas.py"
    prior_runs = repo / "demo_runner_artifacts"
    script = (
        "from pathlib import Path; "
        "assert Path('/tmp/workspace/visible.txt').is_file(); "
        f"assert not Path({str(oracle)!r}).exists(); "
        f"assert not Path({str(prior_runs)!r}).exists(); "
        f"assert {secret_session_name!r} not in "
        "Path('/proc/self/mountinfo').read_text(); "
        "print('isolated', flush=True)"
    )
    command = f"{shlex.quote(sys.executable)} -c " f"{shlex.quote(script)}"
    generator = CombineCostGenerator(
        context=_context(),
        instruction="feedback",
        summaries={},
        run_dir=run_dir,
        codex_cmd=command,
        rounds=[],
    )

    generator._run_codex()

    assert "isolated" in (run_dir / "codex.log").read_text(encoding="utf-8")
