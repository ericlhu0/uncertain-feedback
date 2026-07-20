"""Tests for the visual free-form verbalizer."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, cast

import numpy as np
import pytest

from uncertain_feedback.llm.openai_model import OpenAIModel
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import ATTRIBUTED_FEATURES, CorrectionIntent
from uncertain_feedback.simulated_users.visual import PROMPT, VisualVerbalizer


class _StubModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        images = list(image_input) if isinstance(image_input, list) else []
        self.calls.append((text_input, images))
        return "please lower my arm a little"


@pytest.fixture(name="context")
def _context() -> MpcCostContext:
    fk = SmplLeftArmFK()
    return MpcCostContext(fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3))


def _intent(elevation_delta: float) -> CorrectionIntent:
    deltas = {name: 0.0 for name in ATTRIBUTED_FEATURES}
    deltas["shoulder_elevation"] = elevation_delta
    return CorrectionIntent(
        join_index=2,
        feature_deltas=deltas,
        wrist_offset=np.zeros(3),
        elbow_offset=np.zeros(3),
    )


def test_prompt_contains_no_joint_nomenclature() -> None:
    lowered = PROMPT.lower()
    for word in (
        "elbow",
        "wrist",
        "shoulder",
        "joint",
        "flexion",
        "abduction",
        "elevation",
        "radian",
        "degree",
    ):
        assert word not in lowered


def test_termination_gate_short_circuits(context: MpcCostContext, tmp_path) -> None:
    stub = _StubModel()
    verbalizer = VisualVerbalizer(cast(OpenAIModel, stub), tmp_path)
    result = verbalizer.verbalize(
        _intent(0.05), np.zeros(7), np.zeros((5, 7)), context, "ep", 0
    )
    assert result is None
    assert not stub.calls
    assert not list(Path(tmp_path).glob("*"))


def test_cache_round_trip(context: MpcCostContext, tmp_path) -> None:
    stub = _StubModel()
    verbalizer = VisualVerbalizer(cast(OpenAIModel, stub), tmp_path)
    oracle = np.zeros((5, 7), dtype=np.float64)
    first = verbalizer.verbalize(
        _intent(0.4), np.zeros(7), oracle, context, "ep", 1, window=2
    )
    assert first is not None
    assert first.text == "please lower my arm a little"
    assert first.form == "visual"
    assert len(stub.calls) == 1
    prompt, images = stub.calls[0]
    assert prompt == PROMPT
    assert len(images) == 2
    assert (tmp_path / "ep_round1.txt").exists()

    second = verbalizer.verbalize(
        _intent(0.4), np.zeros(7), oracle, context, "ep", 1, window=2
    )
    assert second is not None
    assert second.text == first.text
    assert len(stub.calls) == 1
