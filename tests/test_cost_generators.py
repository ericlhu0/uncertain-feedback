"""Tests for the llm / turns / agent cost generators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC
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
    GeneratedCostValidationError,
    GeneratedPythonCost,
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

    def __init__(self, response: str) -> None:
        self.response = response
        self.received_images = None
        self.full_output_calls: list[tuple[str, object]] = []
        self.converse_calls = 0
        self.last_messages = None

    def get_full_output(self, text_input: str, image_input=None) -> str:
        self.received_images = image_input
        self.full_output_calls.append((text_input, image_input))
        return self.response

    def converse(self, messages) -> str:
        self.converse_calls += 1
        self.last_messages = messages
        return self.response


def _context() -> object:
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


def _factory_kwargs(tmp_path, fake, cfg):
    context = _context()
    return dict(
        cfg=cfg,
        context=context,
        instruction="raise the elbow",
        summaries=build_motion_summaries(context),
        run_dir=artifact_run_dir(tmp_path, cfg.artifact_dir),
        images={},
        llm_model_factory=lambda _name: fake,
    )


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
    mpc = SmplLeftArmMPC(goals=[np.zeros((3, 3))])
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
    interpret, ground, author = fake.full_output_calls
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
    np.testing.assert_allclose(saved_rollouts[0], full_correction)


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
    cost = GeneratedPythonCost(
        code=_COST_CODE, params={"weight": 1.0}, context=context
    )
    score, rollout = evaluate_candidate_cost(context, cost, _fake_rollout)
    assert np.isfinite(score)
    assert rollout is not None
    # The L2 score compares paths, not timing: dwelling on each frame (same
    # path, 3x slower) scores identically.
    warped_score, _ = evaluate_candidate_cost(
        context, cost, lambda _: np.repeat(rollout, 3, axis=0)
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
    cost = GeneratedPythonCost(
        code=_COST_CODE, params={"weight": 1.0}, context=context
    )
    ranking = rank_candidate_cost(context, cost)
    assert ranking is not None and not ranking.inert
    assert ranking.rank_accuracy == 1.0
    assert ranking.normalized_margin > 0.0
    assert set(ranking.costs) == {
        "original_plan",
        "rejected_cluster_0",
        "chosen_correction",
    }
    assert ranking.sort_key < (1.0, 0.0)


def test_rank_candidate_cost_flags_inert_and_missing_context() -> None:
    context = _ranking_context()
    inert_code = (
        "def cost(q_trajs, context, params):\n"
        "    return np.zeros(q_trajs.shape[0])\n"
    )
    inert = GeneratedPythonCost(code=inert_code, params={}, context=context)
    ranking = rank_candidate_cost(context, inert)
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
    # The feedback describes the ranking, not an L2 match to the correction.
    texts = [m["text"] for m in fake.last_messages if m["role"] == "user"]
    assert any("chosen_correction" in t for t in texts)


def test_agent_generator_errors_when_codex_missing(tmp_path) -> None:
    fake = _FakeLlmModel(_response())

    strict = create_cost_generator(
        **_factory_kwargs(
            tmp_path, fake, LlmCostConfig(backend="agent", strict=True)
        )
    )
    strict.codex_cmd = "definitely-not-a-real-binary-xyz"
    with pytest.raises(GeneratedCostValidationError):
        strict.generate()

    lenient = create_cost_generator(
        **_factory_kwargs(tmp_path, fake, LlmCostConfig(backend="agent"))
    )
    lenient.codex_cmd = "definitely-not-a-real-binary-xyz"
    assert lenient.generate() is None


def test_agent_task_requires_stage_log(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    gen = create_cost_generator(
        **_factory_kwargs(tmp_path, fake, LlmCostConfig(backend="agent"))
    )

    task = gen._task_md("prompt body", iterate=False, image_input=None)

    assert "stage_log.md" in task
    assert "## Stage 1 response" in task
    assert "## Stage 2 response" in task
    assert "## Stage 3 response" in task
