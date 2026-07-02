"""Tests for the llm / staged / turns / agent cost generators."""

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
    StagedCostGenerator,
    TurnsCostGenerator,
    artifact_run_dir,
    build_generated_cost_context,
    build_motion_summaries,
    create_cost_generator,
    evaluate_candidate_cost,
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
        "staged": StagedCostGenerator,
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


def test_staged_generator_runs_three_focused_stages(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(tmp_path, fake, LlmCostConfig(backend="staged"))
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


def test_evaluate_candidate_cost_is_finite() -> None:
    context = _context()
    cost = GeneratedPythonCost(
        code=_COST_CODE, params={"weight": 1.0}, context=context
    )
    score, rollout = evaluate_candidate_cost(context, cost, _fake_rollout)
    assert np.isfinite(score)
    assert rollout is not None
    # No rollout available (e.g. planners without a Cartesian goal) -> inf.
    score, rollout = evaluate_candidate_cost(context, cost)
    assert score == math.inf
    assert rollout is None


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
