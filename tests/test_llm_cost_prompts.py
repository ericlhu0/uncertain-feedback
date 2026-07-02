from __future__ import annotations

from pathlib import Path

import pytest

from uncertain_feedback.planners.mpc.costs.prompts import (
    IMAGE_PLACEHOLDERS,
    PROMPTS,
    build_llm_cost_prompt,
)

ALL_IMAGES = {key: Path(f"{key}.png") for key in IMAGE_PLACEHOLDERS}


def _requested(name: str) -> list[str]:
    """Image placeholders present in a template head, in IMAGE_PLACEHOLDERS order."""
    return [key for key in IMAGE_PLACEHOLDERS if "{" + key + "}" in PROMPTS[name]]


def test_prompts_are_registered() -> None:
    assert {"1"} <= set(PROMPTS)


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_every_prompt_fills_placeholders_and_shares_contract(name: str) -> None:
    text, attached = build_llm_cost_prompt(
        "raise my left arm", {"k": 1}, ALL_IMAGES, prompt=name
    )
    # Instruction + summaries substituted.
    assert "raise my left arm" in text
    assert '"k": 1' in text
    assert "{instruction}" not in text
    assert "{summaries}" not in text
    # No image placeholder left unsubstituted.
    for key in IMAGE_PLACEHOLDERS:
        assert "{" + key + "}" not in text
    # Shared technical contract present in every variant.
    assert "def cost(q_trajs, context, params):" in text
    assert "Return only JSON" in text
    # Exactly the images the template head requests are attached (placeholders are
    # listed in IMAGE_PLACEHOLDERS order in every template), with matching prose.
    requested = _requested(name)
    assert [p.name for p in attached] == [f"{key}.png" for key in requested]
    for key in requested:
        assert IMAGE_PLACEHOLDERS[key] in text


def test_unavailable_image_dropped_without_attachment() -> None:
    # Template "1" requests all three images; provide only the current one.
    text, attached = build_llm_cost_prompt(
        "x", {}, {"current_cluster_traj_img": Path("current.png")}, prompt="1"
    )
    assert [p.name for p in attached] == ["current.png"]
    assert IMAGE_PLACEHOLDERS["current_cluster_traj_img"] in text
    # The unavailable images are dropped whole — their self-describing lines carry the
    # image-specific guidance, so its absence proves no dangling "use the grey arms"
    # instruction survives in the prompt, and no brace is left unsubstituted.
    assert IMAGE_PLACEHOLDERS["other_clusters_traj_img"] not in text
    assert IMAGE_PLACEHOLDERS["reference_traj_img"] not in text
    assert "{other_clusters_traj_img}" not in text
    assert "{reference_traj_img}" not in text


def test_no_images_attached_when_none_available() -> None:
    text, attached = build_llm_cost_prompt("x", {}, {}, prompt="1")
    assert attached == []
    for key in IMAGE_PLACEHOLDERS:
        assert "{" + key + "}" not in text
        assert IMAGE_PLACEHOLDERS[key] not in text


def test_unknown_prompt_raises() -> None:
    with pytest.raises(ValueError, match="Unknown llm_cost prompt"):
        build_llm_cost_prompt("x", {}, {}, prompt="does-not-exist")
