from __future__ import annotations

from pathlib import Path

import pytest

from uncertain_feedback.planners.mpc.costs.prompts import (
    PROMPTS,
    build_llm_cost_prompt,
)


def test_default_prompt_is_registered() -> None:
    assert "default" in PROMPTS


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_every_prompt_fills_placeholders_and_shares_contract(name: str) -> None:
    out = build_llm_cost_prompt(
        "raise my left arm", {"k": 1}, [Path("overlay.png")], prompt=name
    )
    # Placeholders substituted.
    assert "raise my left arm" in out
    assert '"k": 1' in out
    for placeholder in ("{instruction}", "{image_section}", "{summaries}"):
        assert placeholder not in out
    # Shared technical contract present in every variant.
    assert "def cost(q_trajs, context, params):" in out
    assert "Return only JSON" in out
    # Image attachment line shown when an image is provided.
    assert "An image of the trajectory is attached." in out


def test_image_section_omitted_without_images() -> None:
    out = build_llm_cost_prompt("x", {}, [], prompt="default")
    assert "An image of the trajectory is attached." not in out


def test_unknown_prompt_raises() -> None:
    with pytest.raises(ValueError, match="Unknown llm_cost prompt"):
        build_llm_cost_prompt("x", {}, [], prompt="does-not-exist")
