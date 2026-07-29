"""Tests for the llm / turns / agent cost generators."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import uncertain_feedback.planners.mpc.costs.agent_costs as agent_costs_module
from uncertain_feedback.planners.mpc import ArmMPC
from uncertain_feedback.planners.mpc.config import LlmCostConfig
from uncertain_feedback.planners.mpc.costs import (
    AgentCostGenerator,
    LlmCostGenerator,
    MpcCostContext,
    TurnsCostGenerator,
    artifact_run_dir,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    evaluate_candidate_cost,
    rank_candidate_cost,
    resample_equidistant,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedCostValidationError,
    GeneratedPythonCost,
    extract_json_object,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.utils.plot import ArmVisualizer

_COST_CODE = (
    "def cost(q_trajs, context, params):\n"
    "    future = q_trajs[:, 1:, 0, 0]\n"
    "    return params['weight'] * np.mean(future ** 2, axis=1)\n"
)


def _fake_rollout(_cost: GeneratedPythonCost) -> np.ndarray:
    """A valid ``(T, 3, 3)`` arm-aa rollout so the L2 evaluator has a trajectory.

    Mirrors the live ``rollout_fn`` supplied by ``run.py``; without one the
    evaluator returns ``inf`` (no Cartesian rollout available).
    """
    return np.full((6, 3, 3), 0.05, dtype=np.float64)


def _response(description: str = "fake cost") -> str:
    return json.dumps(
        {
            "description": description,
            "explanation": "dev explanation",
            "recipient_explanation": "user explanation",
            "params": {"weight": 1.0},
            "code": _COST_CODE,
        }
    )


class _FakeLlmModel:
    """Honors both the single-turn and multi-turn call surfaces."""

    def __init__(self, response: str, responses: list[str] | None = None) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.received_images: dict[str, Path] | None = None
        self.full_output_calls: list[tuple[str, object]] = []
        self.converse_calls = 0
        self.last_messages: list[dict[str, Any]] | None = None

    def _next_response(self) -> str:
        return self.responses.pop(0) if self.responses else self.response

    def get_full_output(
        self, text_input: str, image_input: dict[str, Path] | None = None
    ) -> str:
        self.received_images = image_input
        self.full_output_calls.append((text_input, image_input))
        return self._next_response()

    def converse(self, messages: list[dict[str, Any]]) -> str:
        self.converse_calls += 1
        self.last_messages = messages
        return self._next_response()


def _context() -> GeneratedCostContext:
    fk = SmplLeftArmFK()
    mpc_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    return build_generated_cost_context(
        mpc_context,
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
    )


def _factory_kwargs(tmp_path: Path, fake: Any, cfg: LlmCostConfig) -> dict[str, Any]:
    context = _context()
    return {
        "cfg": cfg,
        "context": context,
        "instruction": "raise the elbow",
        "summaries": build_motion_summaries(context),
        "run_dir": artifact_run_dir(tmp_path, cfg.artifact_dir),
        "images": {},
        "llm_model_factory": lambda _name: fake,
    }


def _write_corpus(tmp_path: Path, trajectory: np.ndarray) -> Path:
    corpus_dir = tmp_path / "trajectory_corpus"
    corpus_dir.mkdir()
    np.save(corpus_dir / "traj_000.npy", trajectory)
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "n_frames": len(trajectory),
                    "comfortable_until": len(trajectory),
                    "traj_file": "traj_000.npy",
                }
            ]
        ),
        encoding="utf-8",
    )
    return corpus_dir


def test_create_cost_generator_selects_backend(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    cases = {
        "llm": LlmCostGenerator,
        "turns": TurnsCostGenerator,
        "agent": AgentCostGenerator,
    }
    for backend, klass in cases.items():
        gen = create_cost_generator(
            **_factory_kwargs(tmp_path, fake, LlmCostConfig(backend=backend))
        )
        assert isinstance(gen, klass)


def test_llm_generator_produces_and_installs_cost(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="llm"))
    mpc = ArmMPC()
    gen = create_cost_generator(mpc=mpc, **kwargs)

    cost = gen.generate(install=True)

    assert isinstance(cost, GeneratedPythonCost)
    assert len(mpc._extra_costs.terms()) == 1
    assert (kwargs["run_dir"] / "cost.py").exists()
    assert (kwargs["run_dir"] / "validation.json").exists()


def test_llm_generator_runs_three_focused_stages(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="llm"))
    kwargs["images"] = {"current_cluster_traj_img": Path("current.png")}
    gen = create_cost_generator(**kwargs)

    cost = gen.generate(install=False)

    assert isinstance(cost, GeneratedPythonCost)
    # Three separate calls: interpret, ground, author.
    assert len(fake.full_output_calls) == 3
    interpret, ground, author = (  # pylint: disable=unbalanced-tuple-unpacking
        fake.full_output_calls
    )
    # Only the interpret stage sees images; ground and author are text-only.
    assert interpret[1] == ["current.png"]
    assert ground[1] is None and author[1] is None
    # Only the author stage carries the code contract.
    assert "def cost(q_trajs, context, params):" in author[0]
    assert "def cost(q_trajs, context, params):" not in interpret[0]
    # Each stage's prompt and raw response are persisted for inspection.
    for stage in ("interpret", "ground", "author"):
        assert (kwargs["run_dir"] / f"{stage}_prompt.txt").exists()
        assert (kwargs["run_dir"] / f"{stage}_response.txt").exists()
    stage_log = (kwargs["run_dir"] / "stage_log.md").read_text(encoding="utf-8")
    assert "## interpret" in stage_log
    assert "## ground" in stage_log
    assert "## author" in stage_log
    assert "### Response" in stage_log
    assert (kwargs["run_dir"] / "cost.py").exists()


def test_all_backends_reject_cost_that_penalizes_comfortable_corpus_pose(
    tmp_path,
) -> None:
    trajectory = np.full((2, 3, 3), 0.5, dtype=np.float64)
    corpus_dir = _write_corpus(tmp_path, trajectory)
    fake = _FakeLlmModel(_response())

    for backend in ("llm", "turns", "agent"):
        kwargs = _factory_kwargs(
            tmp_path, fake, LlmCostConfig(backend=backend, use_images=False)
        )
        kwargs["corpus_dir"] = corpus_dir
        generator = create_cost_generator(**kwargs)

        with pytest.raises(
            GeneratedCostValidationError,
            match="corpus entry 0, frame 0",
        ):
            generator.parse_cost(_response())


def test_llm_grounding_prompt_includes_comfortable_corpus_ranges(tmp_path) -> None:
    trajectory = np.zeros((2, 3, 3), dtype=np.float64)
    trajectory[1, 2, 0] = 0.5
    corpus_dir = _write_corpus(tmp_path, trajectory)
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="llm"))
    kwargs["corpus_dir"] = corpus_dir

    assert create_cost_generator(**kwargs).generate(install=False) is not None

    ground_prompt = (kwargs["run_dir"] / "ground_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "previously accepted poses" in ground_prompt
    assert "corpus entry 0" in ground_prompt
    assert '"elbow_flexion"' in ground_prompt


def test_llm_generator_writes_rationale(tmp_path, capsys) -> None:
    interpretation = json.dumps(
        {
            "preference": "keep the elbow more bent",
            "distinguishing_dimension": "elbow flexion",
            "direction": "more bent",
            "secondary": "",
            "goal_conflict": False,
            "evidence": {
                "preference": "images: chosen arm ended with a bent elbow",
                "distinguishing_dimension": "summary: chosen elbow_flexion end=1.2",
                "direction": "instruction: 'raise the elbow'",
            },
        }
    )
    grounding = json.dumps(
        {
            "terms": [
                {
                    "feature": "elbow_flexion",
                    "bound_type": "lower_bound",
                    "values": {"threshold": 1.0},
                    "source": "mdm_traj elbow_flexion minimum=1.0",
                }
            ],
            "goal_safety_check": "The lower bound still permits the goal pose.",
        }
    )
    fake = _FakeLlmModel(
        _response(), responses=[interpretation, grounding, _response()]
    )
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="llm"))
    kwargs["context"] = _ranking_context()
    kwargs["summaries"] = build_motion_summaries(kwargs["context"])

    assert create_cost_generator(**kwargs).generate(install=False) is not None

    rationale = json.loads(
        (kwargs["run_dir"] / "rationale.json").read_text(encoding="utf-8")
    )
    assert rationale["instruction"] == "raise the elbow"
    assert rationale["backend"] == "LlmCostGenerator"
    assert rationale["interpret"]["evidence"]["preference"].startswith("images:")
    assert rationale["ground"]["terms"][0]["source"].startswith("mdm_traj")
    assert rationale["final"]["explanation"] == "dev explanation"
    assert rationale["ranking"]["rank_accuracy"] == 1.0
    assert "[cost-gen] grounding source (elbow_flexion):" in capsys.readouterr().out


def test_llm_generator_tolerates_unstructured_stage_rationale(tmp_path) -> None:
    fake = _FakeLlmModel(_response(), responses=["not json", "[]", _response()])
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="llm"))

    assert create_cost_generator(**kwargs).generate(install=False) is not None

    rationale = json.loads(
        (kwargs["run_dir"] / "rationale.json").read_text(encoding="utf-8")
    )
    assert rationale["interpret"] is None
    assert rationale["ground"] is None


def test_generator_saves_reference_with_correction_video(tmp_path, monkeypatch) -> None:
    saved_rollouts: list[np.ndarray] = []

    def fake_render_rollout_video(
        _self,
        rollout,
        save_path,
        **_kwargs,
    ) -> None:
        saved_rollouts.append(np.asarray(rollout, dtype=np.float64).copy())
        save_path.write_bytes(b"video")

    monkeypatch.setattr(
        ArmVisualizer, "render_rollout_video", fake_render_rollout_video
    )
    fk = SmplLeftArmFK()
    mpc_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    full_correction = np.full((6, 3, 3), 0.2, dtype=np.float64)
    context = build_generated_cost_context(
        mpc_context,
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
        full_correction_traj=full_correction,
    )
    fake = _FakeLlmModel(_response())
    cfg = LlmCostConfig(backend="llm")
    run_dir = artifact_run_dir(tmp_path, cfg.artifact_dir)
    gen = create_cost_generator(
        cfg=cfg,
        context=context,
        instruction="raise the elbow",
        summaries=build_motion_summaries(context),
        run_dir=run_dir,
        images={},
        llm_model_factory=lambda _name: fake,
    )

    assert gen.generate(install=False) is not None

    assert (run_dir / "reference_with_correction.mp4").read_bytes() == b"video"
    assert len(saved_rollouts) == 1
    np.testing.assert_allclose(saved_rollouts[0], context.arm_aa(full_correction))


def test_turns_generator_keeps_state_and_returns_best(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(
        tmp_path,
        fake,
        LlmCostConfig(backend="turns", max_turns=3, use_images=False),
    )
    gen = create_cost_generator(rollout_fn=_fake_rollout, **kwargs)

    cost = gen.generate(install=False)

    assert isinstance(cost, GeneratedPythonCost)
    # A real multi-turn conversation ran and accumulated state.
    assert fake.converse_calls >= 1
    assert fake.last_messages is not None and len(fake.last_messages) > 1
    assert (kwargs["run_dir"] / "turn_0" / "score.json").exists()
    stage_log = (kwargs["run_dir"] / "stage_log.md").read_text(encoding="utf-8")
    assert "## interpret" in stage_log
    assert "## refine turn 0" in stage_log
    assert "### Response" in stage_log


def test_evaluate_candidate_cost_is_finite() -> None:
    context = _context()
    cost = GeneratedPythonCost(code=_COST_CODE, params={"weight": 1.0}, context=context)
    score, rollout = evaluate_candidate_cost(context, cost, _fake_rollout)
    assert np.isfinite(score)
    assert rollout is not None
    # The L2 score compares paths, not timing: dwelling on each frame (same
    # path, 3x slower) scores identically.
    warped_score, _ = evaluate_candidate_cost(
        context, cost, lambda _: np.repeat(rollout, 3, axis=0)  # type: ignore[arg-type]
    )
    assert warped_score == pytest.approx(score, abs=1e-8)
    # No rollout available (e.g. planners without a Cartesian goal) -> inf.
    score, rollout = evaluate_candidate_cost(context, cost)
    assert score == math.inf
    assert rollout is None


def _ranking_context() -> object:
    """Context with revealed preferences: chosen (still) vs original + one rejected."""
    fk = SmplLeftArmFK()
    mpc_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    original = np.linspace(0.0, 0.8, 6)[:, None, None] * np.ones((6, 3, 3))
    rejected = np.linspace(0.0, 0.4, 5)[:, None, None] * np.ones((5, 3, 3))
    return build_generated_cost_context(
        mpc_context,
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=np.zeros((4, 3, 3), dtype=np.float64),
        q_history=[],
        window=5,
        reference_traj=original,
        rejected_trajs=(rejected,),
    )


def _original_tie_context() -> object:
    """Context where the test cost rejects another candidate but ties the original."""
    fk = SmplLeftArmFK()
    mpc_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    chosen_and_original = np.zeros((4, 3, 3), dtype=np.float64)
    rejected = np.linspace(0.0, 0.4, 5)[:, None, None] * np.ones((5, 3, 3))
    return build_generated_cost_context(
        mpc_context,
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=chosen_and_original,
        q_history=[],
        window=5,
        reference_traj=chosen_and_original,
        rejected_trajs=(rejected,),
    )


def _rejected_tie_context() -> object:
    """Context where the test cost prefers chosen to original but ties a negative."""
    fk = SmplLeftArmFK()
    mpc_context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    chosen_and_rejected = np.zeros((4, 3, 3), dtype=np.float64)
    original = np.linspace(0.0, 0.8, 6)[:, None, None] * np.ones((6, 3, 3))
    return build_generated_cost_context(
        mpc_context,
        current_q=np.zeros((3, 3), dtype=np.float64),
        mdm_traj=chosen_and_rejected,
        q_history=[],
        window=5,
        reference_traj=original,
        rejected_trajs=(chosen_and_rejected,),
    )


def test_resample_equidistant_removes_timing() -> None:
    def traj_at(ts: np.ndarray) -> np.ndarray:
        out = np.zeros((len(ts), 3, 3))
        out[:, 0, 0] = ts
        return out

    uniform = traj_at(np.linspace(0.0, 1.0, 40))
    warped = traj_at(np.linspace(0.0, 1.0, 40) ** 2)  # same path, different timing
    np.testing.assert_allclose(
        resample_equidistant(uniform, 20),
        resample_equidistant(warped, 20),
        atol=1e-8,
    )
    still = np.full((7, 3, 3), 0.3)
    out = resample_equidistant(still, 5)
    assert out.shape == (5, 3, 3)
    np.testing.assert_allclose(out, 0.3)


def test_rank_candidate_cost_orders_revealed_preferences() -> None:
    context = _ranking_context()
    cost = GeneratedPythonCost(code=_COST_CODE, params={"weight": 1.0}, context=context)  # type: ignore[arg-type]
    ranking = rank_candidate_cost(context, cost)  # type: ignore[arg-type]
    assert ranking is not None and not ranking.inert
    assert ranking.rank_accuracy == 1.0
    assert ranking.normalized_margin > 0.0
    assert ranking.improves_original_plan is True
    assert set(ranking.costs) == {
        "original_plan",
        "rejected_cluster_0",
        "chosen_correction",
    }
    assert ranking.sort_key < (1.0, 0.0)


def test_rank_candidate_cost_requires_strict_marked_wrong_ordering() -> None:
    context = _rejected_tie_context()
    cost = GeneratedPythonCost(code=_COST_CODE, params={"weight": 1.0}, context=context)  # type: ignore[arg-type]

    ranking = rank_candidate_cost(context, cost)  # type: ignore[arg-type]

    assert ranking is not None and not ranking.inert
    assert ranking.costs["chosen_correction"] == ranking.costs["rejected_cluster_0"]
    assert ranking.rank_accuracy == 0.5


def test_llm_generator_rejects_cost_that_ties_original_plan(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="llm"))
    kwargs["context"] = _original_tie_context()
    kwargs["summaries"] = build_motion_summaries(kwargs["context"])

    cost = create_cost_generator(**kwargs).generate(install=False)

    assert cost is None
    rationale = json.loads(
        (kwargs["run_dir"] / "rationale.json").read_text(encoding="utf-8")
    )
    assert rationale["ranking"]["improves_original_plan"] is False
    assert rationale["ranking"]["costs"]["chosen_correction"] == 0.0
    assert rationale["ranking"]["costs"]["original_plan"] == 0.0
    validation = json.loads(
        (kwargs["run_dir"] / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["ok"] is False
    assert "must be strictly less" in validation["error"]


def test_rank_candidate_cost_flags_inert_and_missing_context() -> None:
    context = _ranking_context()
    inert_code = (
        "def cost(q_trajs, context, params):\n"
        "    return np.zeros(q_trajs.shape[0])\n"
    )
    inert = GeneratedPythonCost(code=inert_code, params={}, context=context)  # type: ignore[arg-type]
    ranking = rank_candidate_cost(context, inert)  # type: ignore[arg-type]
    assert ranking is not None and ranking.inert
    assert ranking.sort_key == (math.inf, math.inf)
    # No reference rollout and no rejected clusters -> no ranking signal.
    bare = _context()
    cost = GeneratedPythonCost(code=_COST_CODE, params={"weight": 1.0}, context=bare)
    assert rank_candidate_cost(bare, cost) is None


def test_turns_generator_selects_by_ranking_when_available(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(
        tmp_path,
        fake,
        LlmCostConfig(backend="turns", max_turns=2, use_images=False),
    )
    kwargs["context"] = _ranking_context()
    kwargs["summaries"] = build_motion_summaries(kwargs["context"])
    gen = create_cost_generator(rollout_fn=_fake_rollout, **kwargs)

    cost = gen.generate(install=False)

    assert isinstance(cost, GeneratedPythonCost)
    payload = json.loads(
        (kwargs["run_dir"] / "turn_0" / "score.json").read_text(encoding="utf-8")
    )
    assert payload["ranking"]["rank_accuracy"] == 1.0
    rationale = json.loads(
        (kwargs["run_dir"] / "rationale.json").read_text(encoding="utf-8")
    )
    assert rationale["ranking"] == payload["ranking"]
    assert rationale["ground"] is None
    # The feedback describes the ranking, not an L2 match to the correction.
    texts = [m["text"] for m in fake.last_messages if m["role"] == "user"]  # type: ignore[union-attr]
    assert any("chosen_correction" in t for t in texts)


def test_agent_generator_errors_when_codex_missing(tmp_path) -> None:
    fake = _FakeLlmModel(_response())

    strict = create_cost_generator(
        **_factory_kwargs(tmp_path, fake, LlmCostConfig(backend="agent", strict=True))
    )
    strict.codex_cmd = "definitely-not-a-real-binary-xyz"  # type: ignore[attr-defined]
    with pytest.raises(GeneratedCostValidationError):
        strict.generate()

    lenient = create_cost_generator(
        **_factory_kwargs(tmp_path, fake, LlmCostConfig(backend="agent"))
    )
    lenient.codex_cmd = "definitely-not-a-real-binary-xyz"  # type: ignore[attr-defined]
    assert lenient.generate() is None


def test_agent_codex_waits_for_natural_exit_when_outputs_exist(
    tmp_path, monkeypatch
) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(
        tmp_path,
        fake,
        LlmCostConfig(backend="agent", use_images=True),
    )
    gen = create_cost_generator(**kwargs)
    run_dir = kwargs["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "response.json").write_text(_response(), encoding="utf-8")
    (run_dir / "stage_log.md").write_text("## Stage 3 response\n", encoding="utf-8")
    (run_dir / "ITERATION_LOG.md").write_text("done\n", encoding="utf-8")

    class _HangingProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def wait(self, timeout=None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise agent_costs_module.subprocess.TimeoutExpired("codex", timeout)
            return 0

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = _HangingProcess()
    monkeypatch.setattr(
        agent_costs_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    gen._run_codex()  # type: ignore[attr-defined]

    assert process.wait_calls == 2
    assert not process.terminated
    assert not process.killed
    assert "required outputs are present" not in (run_dir / "codex.log").read_text(
        encoding="utf-8"
    )


def test_agent_codex_hard_timeout_terminates_child(tmp_path, monkeypatch) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="agent"))
    gen = create_cost_generator(**kwargs)
    kwargs["run_dir"].mkdir(parents=True, exist_ok=True)

    class _HangingProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return -15

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = _HangingProcess()
    monkeypatch.setattr(
        agent_costs_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(agent_costs_module, "_CODEX_TIMEOUT_SECONDS", 0.0)

    with pytest.raises(GeneratedCostValidationError, match="timed out"):
        gen._run_codex()  # type: ignore[attr-defined]

    assert process.terminated
    assert not process.killed


def test_agent_task_requires_stage_log(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    gen = create_cost_generator(
        **_factory_kwargs(tmp_path, fake, LlmCostConfig(backend="agent"))
    )

    task = gen._task_md("prompt body", iterate=False, image_input=None)  # type: ignore[attr-defined]

    assert "stage_log.md" in task
    assert "## Stage 1 response" in task
    assert "## Stage 2 response" in task
    assert "## Stage 3 response" in task


def test_agent_stages_corpus_without_feedback_text(tmp_path) -> None:
    corpus_dir = tmp_path / "trajectory_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "traj_000.npy").write_bytes(b"trajectory")
    (corpus_dir / "traj_000_features.csv").write_text(
        "frame,elbow_flexion\n0,1.0\n", encoding="utf-8"
    )
    original = {
        "index": 0,
        "feedback_text": "hidden canonical feedback",
        "trigger_violation": 0.123,
        "traj_file": "traj_000.npy",
        "features_file": "traj_000_features.csv",
    }
    (corpus_dir / "manifest.json").write_text(json.dumps([original]), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generator = AgentCostGenerator(
        context=_context(),
        instruction="actual instruction",
        summaries={},
        run_dir=run_dir,
        corpus_dir=corpus_dir,
    )

    staged_path = generator._stage_corpus()

    staged = json.loads(
        (run_dir / "inputs" / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )
    assert staged_path == Path("/tmp/workspace/inputs/corpus")
    assert "feedback_text" not in staged[0]
    assert "trigger_violation" not in staged[0]
    assert json.loads((corpus_dir / "manifest.json").read_text())[0] == original
    assert "feedback_text" not in generator._corpus_section()


def test_agent_generator_writes_rationale_from_stage_log(tmp_path, monkeypatch) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(
        tmp_path,
        fake,
        LlmCostConfig(backend="agent", use_images=False),
    )
    kwargs["context"] = _ranking_context()
    kwargs["summaries"] = build_motion_summaries(kwargs["context"])
    gen = create_cost_generator(rollout_fn=_fake_rollout, **kwargs)
    run_dir = kwargs["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "response.json").write_text(_response(), encoding="utf-8")
    (run_dir / "stage_log.md").write_text(
        "## Stage 1 response\n\n```json\n"
        '{"preference": "bend the elbow", "evidence": '
        '{"preference": "images: chosen pose"}}\n'
        "```\n\n## Stage 2 response\n\n```json\n"
        '{"terms": [{"feature": "elbow_flexion", "source": '
        '"summary: end=1.2"}]}\n'
        "```\n\n## Stage 3 response\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gen, "_run_codex", lambda: None)
    monkeypatch.setattr(gen, "_save_reference_video", lambda: None)

    assert gen.generate(install=False) is not None

    rationale = json.loads((run_dir / "rationale.json").read_text(encoding="utf-8"))
    assert rationale["interpret"]["preference"] == "bend the elbow"
    assert rationale["ground"]["terms"][0]["source"] == "summary: end=1.2"
    assert rationale["ranking"]["rank_accuracy"] == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"value": 1}\n```', {"value": 1}),
        ('prefix {"value": 2} suffix', {"value": 2}),
        ("garbage", None),
        ("[1, 2, 3]", None),
    ],
)
def test_extract_json_object(text: str, expected: dict | None) -> None:
    assert extract_json_object(text) == expected
