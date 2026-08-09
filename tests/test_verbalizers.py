"""Tests for the specificity-level verbalizers."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import numpy as np

from uncertain_feedback.simulated_users import (
    ATTRIBUTED_FEATURES,
    CorrectionIntent,
    verbalize_everyday,
    verbalize_joint_resolved,
    verbalize_motion_directive,
    verbalize_vague,
)
from uncertain_feedback.simulated_users.verbalizers import VAGUE_PHRASE


def _intent(
    deltas: dict[str, float] | None = None,
    wrist: tuple[float, float, float] = (0.0, 0.0, 0.0),
    elbow: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CorrectionIntent:
    feature_deltas = {name: 0.0 for name in ATTRIBUTED_FEATURES}
    feature_deltas.update(deltas or {})
    return CorrectionIntent(
        join_index=0,
        feature_deltas=feature_deltas,
        wrist_offset=np.asarray(wrist, dtype=np.float64),
        elbow_offset=np.asarray(elbow, dtype=np.float64),
    )


def test_all_verbalizers_return_none_below_dead_band() -> None:
    quiet = _intent({"shoulder_elevation": 0.1}, wrist=(0.0, 0.3, 0.0))
    rng = np.random.default_rng(0)
    assert verbalize_vague(quiet) is None
    assert verbalize_joint_resolved(quiet) is None
    assert verbalize_everyday(quiet, rng) is None
    assert verbalize_motion_directive(quiet) is None


def test_vague_is_fixed_phrase() -> None:
    utterance = verbalize_vague(_intent({"shoulder_elevation": 0.3}))
    assert utterance is not None
    assert utterance.text == VAGUE_PHRASE
    assert utterance.form == "vague"


def test_joint_resolved_names_top_two_features() -> None:
    utterance = verbalize_joint_resolved(
        _intent({"elbow_flexion": -0.4, "shoulder_elevation": 0.2})
    )
    assert utterance is not None
    assert utterance.text == "bend my elbow more and keep my arm lower"
    assert utterance.form == "joint_resolved"


def test_joint_resolved_single_feature() -> None:
    utterance = verbalize_joint_resolved(_intent({"shoulder_abduction_adduction": 0.3}))
    assert utterance is not None
    assert utterance.text == "keep my arm closer to my body"


def test_everyday_is_seed_deterministic() -> None:
    intent = _intent(
        {"shoulder_elevation": 0.3}, wrist=(0.0, 0.2, 0.0), elbow=(0.0, 0.1, 0.0)
    )
    first = verbalize_everyday(intent, np.random.default_rng(7))
    second = verbalize_everyday(intent, np.random.default_rng(7))
    assert first is not None and second is not None
    assert first == second


def test_everyday_dead_band_gates_referents() -> None:
    # Offsets below 0.05 m leave only the joint-resolved form eligible.
    intent = _intent(
        {"shoulder_elevation": 0.3}, wrist=(0.0, 0.02, 0.0), elbow=(0.0, 0.01, 0.0)
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        utterance = verbalize_everyday(intent, rng)
        assert utterance is not None
        assert utterance.form == "joint_resolved"
        assert utterance.text == "keep my arm lower"


def test_motion_directive_offset_up_means_move_down() -> None:
    # Offsets are nominal - oracle, so an upward wrist offset commands "down".
    utterance = verbalize_motion_directive(
        _intent({"shoulder_elevation": 0.3}, wrist=(0.0, 0.3, 0.0))
    )
    assert utterance is not None
    assert utterance.text == "move my left arm down a lot"
    assert utterance.form == "motion_directive"


def test_motion_directive_picks_dominant_referent() -> None:
    utterance = verbalize_motion_directive(
        _intent(
            {"elbow_flexion": 0.3}, wrist=(0.0, 0.0, 0.06), elbow=(0.16, 0.0, 0.0)
        )
    )
    assert utterance is not None
    assert utterance.text == "move my left elbow closer to my body"


def test_motion_directive_composes_comparable_axes() -> None:
    utterance = verbalize_motion_directive(
        _intent({"shoulder_elevation": 0.3}, wrist=(0.2, -0.15, 0.0))
    )
    assert utterance is not None
    assert utterance.text == "move my left arm closer to my body and up a lot"


def test_motion_directive_small_offset_says_a_bit() -> None:
    utterance = verbalize_motion_directive(
        _intent({"shoulder_elevation": 0.2}, wrist=(0.0, 0.04, 0.0))
    )
    assert utterance is not None
    assert utterance.text == "move my left arm down a bit"


def test_everyday_offset_up_means_move_down() -> None:
    # Offsets are nominal - oracle, so an upward wrist offset phrases "down".
    intent = _intent({"shoulder_elevation": 0.3}, wrist=(0.0, 0.3, 0.0))
    rng = np.random.default_rng(0)
    arm_texts = []
    for _ in range(50):
        utterance = verbalize_everyday(intent, rng)
        assert utterance is not None
        if utterance.form == "arm":
            arm_texts.append(utterance.text)
    assert arm_texts
    for text in arm_texts:
        assert text == "move my arm down a lot"
