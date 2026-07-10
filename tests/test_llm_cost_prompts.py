from __future__ import annotations

from pathlib import Path

from uncertain_feedback.planners.mpc.costs.prompts import (
    IMAGE_PLACEHOLDERS,
    build_author_prompt,
    build_ground_prompt,
    build_interpret_prompt,
    build_refine_prompt,
    build_staged_task_body,
    compact_summaries,
)

ALL_IMAGES = {key: Path(f"{key}.png") for key in IMAGE_PLACEHOLDERS}


def test_interpret_prompt_fills_placeholders_without_code_contract() -> None:
    text, attached = build_interpret_prompt(
        "raise my left arm",
        {"mdm_traj": {"joint_features": {"elbow_flexion": {"end": 1.0}}}},
        ALL_IMAGES,
    )
    # Instruction + summaries substituted.
    assert "raise my left arm" in text
    assert '"elbow_flexion"' in text
    assert "{instruction}" not in text
    assert "{summaries}" not in text
    # No image placeholder left unsubstituted.
    for key in IMAGE_PLACEHOLDERS:
        assert "{" + key + "}" not in text
    # Interpretation sees images, but no code-writing contract.
    assert "def cost(q_trajs, context, params):" not in text
    assert "Return only JSON" not in text
    assert [p.name for p in attached] == [f"{key}.png" for key in IMAGE_PLACEHOLDERS]
    for key in IMAGE_PLACEHOLDERS:
        assert IMAGE_PLACEHOLDERS[key] in text


def test_unavailable_image_dropped_without_attachment() -> None:
    text, attached = build_interpret_prompt(
        "x", {}, {"current_cluster_traj_img": Path("current.png")}
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
    text, attached = build_interpret_prompt("x", {}, {})
    assert attached == []
    for key in IMAGE_PLACEHOLDERS:
        assert "{" + key + "}" not in text
        assert IMAGE_PLACEHOLDERS[key] not in text


def test_ground_and_author_split_numeric_work_from_code_contract() -> None:
    ground = build_ground_prompt(
        '{"preference": "keep the elbow more bent"}',
        {"mdm_traj": {"joint_features": {"elbow_flexion": {"end": 1.0}}}},
    )
    assert "keep the elbow more bent" in ground
    assert '"elbow_flexion"' in ground
    assert "def cost(q_trajs, context, params):" not in ground

    author = build_author_prompt('{"terms": []}')
    assert '{"terms": []}' in author
    assert "def cost(q_trajs, context, params):" in author
    assert "Return only JSON" in author


def test_refine_prompt_keeps_interpretation_fixed_and_can_author() -> None:
    text = build_refine_prompt(
        '{"preference": "keep the elbow more bent"}',
        {"mdm_traj": {"joint_features": {"elbow_flexion": {"end": 1.0}}}},
    )
    assert "do NOT re-interpret it" in text
    assert "keep the elbow more bent" in text
    assert "def cost(q_trajs, context, params):" in text
    assert "Return only JSON" in text


def test_staged_task_body_inlines_all_stages_for_agent() -> None:
    text, attached = build_staged_task_body(
        "raise my left arm",
        {"mdm_traj": {"joint_features": {"elbow_flexion": {"end": 1.0}}}},
        {"current_cluster_traj_img": Path("current.png")},
    )
    assert "## Stage 1" in text
    assert "## Stage 2" in text
    assert "## Stage 3" in text
    assert "raise my left arm" in text
    assert "def cost(q_trajs, context, params):" in text
    assert [p.name for p in attached] == ["current.png"]


def test_compact_summaries_drop_raw_joint_arrays() -> None:
    slim = compact_summaries(
        {
            "mdm_traj": {
                "joint_features": {"elbow_flexion": {"end": 1.0}},
                "positions": {"wrist": {"start": [0, 0, 0], "end": [1, 1, 1]}},
                "joint_angles": [[[0.0]]],
            }
        }
    )
    assert slim == {
        "mdm_traj": {
            "joint_features": {"elbow_flexion": {"end": 1.0}},
            "end_positions": {"wrist": [1, 1, 1]},
        }
    }
