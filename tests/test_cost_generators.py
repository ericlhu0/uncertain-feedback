"""Tests for the llm / turns / agent cost generators."""

from __future__ import annotations

import json

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
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_COST_CODE = (
    "def cost(q_trajs, context, params):\n"
    "    future = q_trajs[:, 1:, 0, 0]\n"
    "    return params['weight'] * np.mean(future ** 2, axis=1)\n"
)


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
        self.converse_calls = 0
        self.last_messages = None

    def get_full_output(self, text_input: str, image_input=None) -> str:
        self.received_images = image_input
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


def test_turns_generator_keeps_state_and_returns_best(tmp_path) -> None:
    fake = _FakeLlmModel(_response())
    kwargs = _factory_kwargs(
        tmp_path, fake, LlmCostConfig(backend="turns", max_turns=3)
    )
    gen = create_cost_generator(**kwargs)

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
    assert np.isfinite(evaluate_candidate_cost(context, cost))


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
