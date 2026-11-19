"""Tests for cost-generation prompt assembly and image attachment."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

from uncertain_feedback.cost_generation.prompts import (
    IMAGE_PLACEHOLDERS,
    build_author_prompt,
    build_combine_task_body,
    build_ground_prompt,
    build_interpret_prompt,
    build_refine_prompt,
    build_staged_task_body,
    compact_summaries,
)

ALL_IMAGES = {key: Path(f"{key}.png") for key in IMAGE_PLACEHOLDERS}
CHOSEN_IMAGES = [
    "current_cluster_traj_img",
    "other_clusters_traj_img",
    "reference_traj_img",
]


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
    assert "def path_margins(features, params):" not in text
    assert "Return only JSON" not in text
    assert '"evidence"' in text
    assert (
        "Only marked-wrong candidates are shown; unmarked candidates carry no "
        "preference signal. The spoken instruction defines the overall correction. "
        "The candidate "
        "selection is used only to disambiguate its joint-space interpretation "
        "and preferred degree."
    ) in text
    assert "The GOLD STAR is the original left-wrist task-space goal." in text
    assert (
        "A candidate is a short correction motion, not necessarily a complete "
        "trajectory to that goal."
    ) in text
    assert 'every entry in "rejected_ends" maps one statistic' in text
    assert "marked-wrong rollout that produced it" in text
    assert (
        '"chosen_minus_original_plan" shows whether that feature can distinguish'
        in text
    )
    assert "Large chosen-versus-marked-wrong separation alone is not enough" in text
    assert [p.name for p in attached] == [f"{key}.png" for key in CHOSEN_IMAGES]
    for key in CHOSEN_IMAGES:
        assert IMAGE_PLACEHOLDERS[key] in text
    assert IMAGE_PLACEHOLDERS["interrupted_plan_img"] not in text
    assert "candidate the person explicitly marked as wrong" in text


def test_language_only_interpret_and_ground_frame_plan_as_interrupted() -> None:
    text, attached = build_interpret_prompt(
        "keep my arm closer to my body",
        {"reference": {"joint_features": {}}, "current": {}, "recent": {}},
        ALL_IMAGES,
        language_only=True,
    )
    assert [p.name for p in attached] == ["interrupted_plan_img.png"]
    assert IMAGE_PLACEHOLDERS["interrupted_plan_img"] in text
    for key in CHOSEN_IMAGES:
        assert IMAGE_PLACEHOLDERS[key] not in text
        assert "{" + key + "}" not in text
    assert "the path the person CHOSE" not in text
    assert "correcting AWAY from" in text
    ground = build_ground_prompt("{}", {}, language_only=True)
    assert "INTERRUPTED PLAN" in ground
    assert 'strictly higher than the "recent" history' in ground
    assert "exceed them by strictly less than" not in ground


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
    assert attached == []  # pylint: disable=use-implicit-booleaness-not-comparison
    for key, placeholder in IMAGE_PLACEHOLDERS.items():
        assert "{" + key + "}" not in text
        assert placeholder not in text


def test_ground_and_author_split_numeric_work_from_code_contract() -> None:
    ground = build_ground_prompt(
        '{"preference": "keep the elbow more bent"}',
        {"mdm_traj": {"joint_features": {"elbow_flexion": {"end": 1.0}}}},
    )
    assert "keep the elbow more bent" in ground
    assert '"elbow_flexion"' in ground
    assert "def path_margins(features, params):" not in ground
    assert "exceed them by strictly less than" in ground
    assert "does not mean the original plan must stay inside the bounds" in ground
    # A chosen correction is preferred, not accepted: satisfying it is the default.
    assert "That is the default placement, not an absolute rule" in ground
    assert "MUST already be satisfied" not in ground

    author = build_author_prompt('{"terms": []}')
    assert '{"terms": []}' in author
    assert "def path_margins(features, params):" in author
    assert "Return only JSON" in author


def test_refine_prompt_keeps_interpretation_fixed_and_can_author() -> None:
    text = build_refine_prompt(
        '{"preference": "keep the elbow more bent"}',
        {"mdm_traj": {"joint_features": {"elbow_flexion": {"end": 1.0}}}},
    )
    assert "do NOT re-interpret it" in text
    assert "keep the elbow more bent" in text
    assert "def path_margins(features, params):" in text
    assert "Return only JSON" in text
    assert "exceed them by strictly less than" in text


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
    assert "def path_margins(features, params):" in text
    assert '"evidence"' in text
    assert [p.name for p in attached] == ["current.png"]


def test_compact_summaries_drop_raw_joint_arrays() -> None:
    slim = compact_summaries(
        {
            "mdm_traj": {
                "joint_features": {"elbow_flexion": {"end": 1.0}},
                "positions": {"wrist": {"start": [0, 0, 0], "end": [1, 1, 1]}},
                "joint_angles": [[[0.0]]],
            },
            "candidate_comparison": {
                "elbow_flexion": {
                    "chosen_end": 1.0,
                    "rejected_ends": {"rejected_cluster_0": 0.5},
                }
            },
        }
    )
    assert slim == {
        "mdm_traj": {
            "joint_features": {"elbow_flexion": {"end": 1.0}},
            "end_positions": {"wrist": [1, 1, 1]},
        },
        "candidate_comparison": {
            "elbow_flexion": {
                "chosen_end": 1.0,
                "rejected_ends": {"rejected_cluster_0": 0.5},
            }
        },
    }


def test_combine_prompt_replays_every_round_and_requires_anchors(tmp_path) -> None:
    rounds = [
        {
            "index": index,
            "goal": [0.1 * index, 0.2, 0.3],
            "feedback_text": f"feedback {index}",
            "trigger_step": index + 2,
            "state_path": str(tmp_path / f"round_{index}" / "state.pkl"),
            "summaries": {"trigger": {"joint_features": {"shoulder_elevation": index}}},
            "cost_code": f"def cost_{index}(): pass",
            "params": {"threshold": index},
            "image_paths": [str(tmp_path / f"round_{index}.png")],
        }
        for index in range(2)
    ]

    text, images = build_combine_task_body(rounds)

    assert "feedback 0" in text and "feedback 1" in text
    assert "def cost_0" in text and "def cost_1" in text
    assert "np.interp" in text
    assert "def path_margins(features, params):" in text
    assert images == [tmp_path / "round_0.png", tmp_path / "round_1.png"]


def _combine_body() -> str:
    rounds = [
        {
            "index": 0,
            "goal": [0.0, 0.2, 0.3],
            "feedback_text": "feedback",
            "trigger_step": 2,
            "state_path": "state.pkl",
            "summaries": {"trigger": {}},
            "cost_code": "def cost_0(): pass",
            "params": {},
            "image_paths": [],
        }
    ]
    text, _ = build_combine_task_body(rounds)
    return text


def test_explicit_relation_may_produce_coupled_cost_in_one_round() -> None:
    interpret, _ = build_interpret_prompt("x", {}, {})
    ground = build_ground_prompt("{}", {})
    # Interpret can emit a relationship from an explicit spoken coupling...
    assert '"relationship"' in interpret
    assert "bend my elbow more as you raise my arm" in interpret
    # ...and ground turns a non-null relationship into a pose-dependent bound.
    assert 'when stage one returned a non-null "relationship"' in ground
    assert '"bounded_dimension"' in ground and '"conditioning_dimension"' in ground


def test_trajectory_correlation_alone_forbids_global_coupling() -> None:
    interpret, _ = build_interpret_prompt("x", {}, {})
    assert (
        "Correlated joint changes in one selected trajectory are insufficient"
        in interpret
    )


def test_associated_changes_preserved_but_not_immediate_cost_terms() -> None:
    interpret, _ = build_interpret_prompt("x", {}, {})
    ground = build_ground_prompt("{}", {})
    assert '"associated_changes"' in interpret
    assert (
        'Do NOT turn stage one\'s "associated_changes" or "unresolved_explanations" '
        "into additional bounds" in ground
    )


def test_interpretation_is_local_and_provisional_by_default() -> None:
    interpret, _ = build_interpret_prompt("x", {}, {})
    assert "simplest LOCAL explanation of this one correction" in interpret
    assert "not automatically a global mobility restriction" in interpret


def test_combine_consults_literature_but_rounds_decide() -> None:
    combine = " ".join(_combine_body().split())
    assert "consult the biomechanics and clinical literature" in combine
    assert "any differentiable function of the five features" in combine
    # Accepted motion stays inside the limits; a chosen correction only has to
    # keep beating what it replaced.
    assert "may be stricter than an earlier chosen correction" in combine
    assert "must stay inside every limit" in combine
    assert "the rounds decide between them" in combine


def test_primary_interpret_fields_and_output_contract_intact() -> None:
    interpret, _ = build_interpret_prompt("x", {}, {})
    for field in (
        "preference",
        "distinguishing_dimension",
        "direction",
        "secondary",
        "goal_conflict",
        "evidence",
    ):
        assert f'"{field}"' in interpret
    # Final generated-cost schema (author stage) is unchanged.
    author = build_author_prompt('{"terms": []}')
    assert "def path_margins(features, params):" in author
    assert "Return only JSON" in author


def test_contract_states_margins_and_scopes_speed_and_position() -> None:
    author = " ".join(build_author_prompt('{"terms": []}').split())
    assert "def speed_margins(features, params):" in author
    assert "POSITIVE where the limit is violated" in author
    assert "Never clamp, square, weight, sum or average a margin" in author
    # Speed features come with units and signs.
    assert (
        'features["wrist_speed"] speed of the hand relative to the torso, m/s' in author
    )
    assert "negative while it straightens" in author
    # Scoped rules replace the blanket velocity and Cartesian bans.
    assert "Use speed features ONLY when the instruction is about speed" in author
    assert "Do NOT use Cartesian or world-space positions" in author
    assert "Do not include smoothness, velocity, or acceleration costs" not in author
    interpret, _ = build_interpret_prompt("x", {}, {})
    assert '"speed": an object or null' in interpret
    ground = build_ground_prompt("{}", {})
    assert '"speed_cap"' in ground and "wrist_speed" in ground
