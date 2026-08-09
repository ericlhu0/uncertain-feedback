"""Verbalizers: one function per utterance-specificity level.

Each verbalizer phrases the same level-invariant :class:`CorrectionIntent` at
a different specificity — ``vague`` (a fixed complaint), ``everyday`` (a
sampled body-part phrase), ``motion_directive`` (a deterministic imperative
over the dominant referent in egocentric direction words), and
``joint_resolved`` (the divergent joint features named directly). All return
``None`` exactly when :func:`has_feedback_content` is false, so episode
termination never depends on the level. Intent deltas/offsets are
nominal − oracle; phrases point the opposite way (offset up → move down).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from uncertain_feedback.simulated_users.attribution import (
    CorrectionIntent,
    has_feedback_content,
)

FEATURE_DEAD_BAND = 0.15
OFFSET_DEAD_BAND = 0.05
_MAGNITUDE_SMALL = 0.12
_MAGNITUDE_LARGE = 0.25
_JOINT_RESOLVED_PRIOR = 0.15

VAGUE_PHRASE = "stop, that hurts — not like that"

# (feature, sign of nominal − oracle delta) → corrective request.
JOINT_PHRASES: dict[tuple[str, int], str] = {
    ("elbow_flexion", 1): "straighten my elbow more",
    ("elbow_flexion", -1): "bend my elbow more",
    ("shoulder_flexion_extension", 1): "keep my arm further back",
    ("shoulder_flexion_extension", -1): "bring my arm forward more",
    ("shoulder_abduction_adduction", 1): "keep my arm closer to my body",
    ("shoulder_abduction_adduction", -1): "move my arm out to the side more",
    ("shoulder_elevation", 1): "keep my arm lower",
    ("shoulder_elevation", -1): "raise my arm higher",
}

# Per-axis corrective direction words on the verified world axes (x lateral,
# +x away from the torso for the left arm; y up; z forward). The word points
# opposite the offset because the offset is nominal − oracle.
_AXIS_DIRECTION_WORDS = (
    ("in toward my body", "out to the side"),
    ("down", "up"),
    ("back", "forward"),
)

# Egocentric variant for the motion-directive level.
_MOTION_DIRECTIVE_WORDS = (
    ("closer to my body", "out to the side"),
    ("down", "up"),
    ("back", "forward"),
)


@dataclass(frozen=True)
class Utterance:
    """One synthesized correction utterance and the form it was drawn as."""

    text: str
    form: str


def _joint_resolved_phrases(intent: CorrectionIntent) -> list[tuple[str, float]]:
    """Return (phrase, |delta|) for every feature delta above the dead-band."""
    ranked = sorted(
        intent.feature_deltas.items(), key=lambda item: abs(item[1]), reverse=True
    )
    return [
        (JOINT_PHRASES[(name, 1 if delta > 0 else -1)], abs(delta))
        for name, delta in ranked
        if abs(delta) > FEATURE_DEAD_BAND
    ]


def _direction_words(
    offset: np.ndarray,
    lexicon: tuple[tuple[str, str], ...] = _AXIS_DIRECTION_WORDS,
) -> str:
    """Corrective direction words from the offset's dominant axis components."""
    components = np.asarray(offset, dtype=np.float64)
    order = np.argsort(-np.abs(components))
    dominant = abs(components[order[0]])
    words = []
    for axis in order[:2]:
        if abs(components[axis]) < dominant / 2.0:
            break
        negative, positive = lexicon[axis]
        words.append(negative if components[axis] > 0 else positive)
    return " and ".join(words)


def _offset_phrase(
    referent: str,
    offset: np.ndarray,
    lexicon: tuple[tuple[str, str], ...] = _AXIS_DIRECTION_WORDS,
) -> str:
    phrase = f"move my {referent} {_direction_words(offset, lexicon)}"
    norm = float(np.linalg.norm(offset))
    if norm < _MAGNITUDE_SMALL:
        return f"{phrase} a bit"
    if norm >= _MAGNITUDE_LARGE:
        return f"{phrase} a lot"
    return phrase


def verbalize_vague(intent: CorrectionIntent) -> Utterance | None:
    """A fixed complaint carrying no direction beyond discomfort."""
    if not has_feedback_content(intent):
        return None
    return Utterance(VAGUE_PHRASE, "vague")


def verbalize_joint_resolved(intent: CorrectionIntent) -> Utterance | None:
    """The top two above-dead-band joint features named with their directions."""
    if not has_feedback_content(intent):
        return None
    phrases = [phrase for phrase, _ in _joint_resolved_phrases(intent)[:2]]
    return Utterance(" and ".join(phrases), "joint_resolved")


def verbalize_everyday(
    intent: CorrectionIntent, rng: np.random.Generator
) -> Utterance | None:
    """A sampled body-part phrase at everyday granularity.

    Candidate forms are the arm phrase (from the wrist offset), the elbow
    phrase (from the elbow offset), and the top joint-resolved phrase; each is
    dropped below its dead-band and the rest are sampled with weight
    proportional to contrast magnitude times a form prior.
    """
    if not has_feedback_content(intent):
        return None
    candidates: list[tuple[str, str, float]] = []
    for referent, offset in (
        ("arm", intent.wrist_offset),
        ("elbow", intent.elbow_offset),
    ):
        norm = float(np.linalg.norm(offset))
        if norm > OFFSET_DEAD_BAND:
            candidates.append((referent, _offset_phrase(referent, offset), norm))
    joint_phrase, joint_magnitude = _joint_resolved_phrases(intent)[0]
    candidates.append(
        ("joint_resolved", joint_phrase, joint_magnitude * _JOINT_RESOLVED_PRIOR)
    )
    weights = np.array([weight for _, _, weight in candidates], dtype=np.float64)
    index = int(rng.choice(len(candidates), p=weights / weights.sum()))
    form, text, _ = candidates[index]
    return Utterance(text, form)


def verbalize_motion_directive(intent: CorrectionIntent) -> Utterance | None:
    """A deterministic imperative over the dominant referent's offset.

    The referent (elbow or arm) with the larger Cartesian contrast is named
    with egocentric direction words, always yielding a directive whenever the
    intent has feedback content.
    """
    if not has_feedback_content(intent):
        return None
    referent, offset = max(
        (("elbow", intent.elbow_offset), ("arm", intent.wrist_offset)),
        key=lambda candidate: float(np.linalg.norm(candidate[1])),
    )
    text = _offset_phrase(f"left {referent}", offset, _MOTION_DIRECTIVE_WORDS)
    return Utterance(text, "motion_directive")


# Name → callable for config-driven selection. "everyday" is bound to an rng
# and "visual" (a class, registered by visual.py) is constructed at episode
# setup, so values are loosely typed.
VERBALIZERS: dict[str, Callable[..., object]] = {
    "vague": verbalize_vague,
    "everyday": verbalize_everyday,
    "motion_directive": verbalize_motion_directive,
    "joint_resolved": verbalize_joint_resolved,
}
