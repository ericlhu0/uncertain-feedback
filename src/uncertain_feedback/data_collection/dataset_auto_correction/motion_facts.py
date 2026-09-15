"""What a correction clip's window actually does, and which words may claim it.

One measurement of a clip shared by both ends of the caption pipeline: the
captioner (``captioning.py``) puts it in the prompt and verifies every line the
VLM returns against it, and the dataset build (``build_dataset.py``
``--consistent_captions``) re-checks the stored lines before encoding. Both read
the same axes through the same word table, so a line the captioner accepted is
not dropped later for a different reason.

The measured window is frame ``n_prefix - 1`` (the last pinned frame, where the
correction starts) to the clip's last frame — exactly the window
:func:`~...captioning.render_window` draws — in the SMPL world frame, where ``+x``
is the person's left, ``+y`` up and ``+z`` their front. Since the moved limb is
the *left* arm, ``+x`` is also away from the body, so "out" and "left" are one
axis direction and "in" and "right" the other.

Axes below the dead band carry no direction at all: a caption asserting one on
them describes a motion the clip does not make.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    ArmFeatureContext,
    arm_aa_from_state,
    arm_feature_series,
)
from uncertain_feedback.planners.mpc.kinematics import ELBOW_CHAIN_IDX, WRIST_CHAIN_IDX

# Travel (m) and flexion change (rad) below which an axis has no direction.
DEAD_BAND_M = 0.02
DEAD_BAND_RAD = 0.05

# Nouns that turn a side word into the name of a landmark ("your left hip") rather
# than a direction of travel, which is the difference between a caption that
# asserts +x and one that merely points at something.
_BODY_NOUN = (
    r"(?:hand|hands|arm|arms|forearm|forearms|elbow|elbows|wrist|wrists"
    r"|shoulder|shoulders|hip|hips|side|leg|legs|knee|knees|thigh|thighs"
    r"|chest|ribs|waist|lap|ear|cheek|face|head|foot|feet)"
)

_UP = re.compile(
    r"\b(up|upward|upwards|lift|lifts|lifted|raise|raises|raised|rise|rises"
    r"|rising|higher)\b",
    re.IGNORECASE,
)
_DOWN = re.compile(
    r"\b(down|downward|downwards|lower|lowers|lowered|drop|drops|dropped)\b",
    re.IGNORECASE,
)
# "out in front" is depth and "straighten out" is the elbow, neither is width;
# "left" is a direction only when it does not name a landmark's side.
_OUT = re.compile(
    r"\b(outward|outwards|away)\b"
    r"|(?<!straighten )(?<!straightens )(?<!straightened )\bout\b"
    r"(?!\s+in\s+front)"
    r"|\bleft\b(?!\s+" + _BODY_NOUN + r")",
    re.IGNORECASE,
)
_IN = re.compile(
    r"\b(inward|inwards|across|tuck|tucks|tucked)\b"
    r"|\bright\b(?!\s+" + _BODY_NOUN + r")"
    r"|\bin\b(?!\s+(?:front|a|the))"
    r"|\btowards?\s+(?:my|your|the)\s+(?:body|chest|midline|ribs|waist|side)\b",
    re.IGNORECASE,
)
# "back" is only depth when it is neither the body part nor "back down"/"back up".
_FORWARD = re.compile(r"\b(forward|forwards|ahead)\b|\bin\s+front\b", re.IGNORECASE)
_BACK = re.compile(
    r"\b(backward|backwards|behind)\b"
    r"|(?<!my )(?<!your )\bback\b(?!\s+(?:up|down|in|out))",
    re.IGNORECASE,
)
_BEND = re.compile(
    r"\b(bend|bends|bent|bending|flex|flexes|flexed|curl|curls|curled"
    r"|fold|folds|folded)\b",
    re.IGNORECASE,
)
_STRAIGHTEN = re.compile(
    r"\b(straight|straighten|straightens|straightened|extend|extends|extended"
    r"|unbend|unfold|unfolds|unfolded)\b",
    re.IGNORECASE,
)

_ELBOW_PART = re.compile(r"\belbows?\b", re.IGNORECASE)
_HAND_PART = re.compile(
    r"\b(hand|hands|wrist|wrists|arm|arms|forearm|forearms|palm|palms|fingers)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Axis:
    """One quantity a caption can assert a direction on.

    ``part`` is the body part a direction word must be talking about for it to
    count on this axis, decided by which part the word sits closest to in the
    sentence; ``None`` means the words name the axis on their own.
    """

    quantity: str
    part: str | None
    positive: re.Pattern[str]
    negative: re.Pattern[str]


AXES: tuple[_Axis, ...] = (
    _Axis("wrist_dy", "hand", _UP, _DOWN),
    _Axis("wrist_dx", "hand", _OUT, _IN),
    _Axis("wrist_dz", "hand", _FORWARD, _BACK),
    _Axis("elbow_dy", "elbow", _UP, _DOWN),
    _Axis("elbow_dx", "elbow", _OUT, _IN),
    _Axis("elbow_dz", "elbow", _FORWARD, _BACK),
    _Axis("flexion", None, _BEND, _STRAIGHTEN),
)


@dataclass(frozen=True)
class MotionFacts:
    """Measured travel of one clip's described window, in metres and radians."""

    wrist_dx: float
    wrist_dy: float
    wrist_dz: float
    elbow_dx: float
    elbow_dy: float
    elbow_dz: float
    flexion: float

    def signs(self) -> dict[str, int]:
        """``+1`` / ``-1`` per axis, ``0`` where the motion is inside the dead band."""
        return {
            "wrist_dx": _sign(self.wrist_dx, DEAD_BAND_M),
            "wrist_dy": _sign(self.wrist_dy, DEAD_BAND_M),
            "wrist_dz": _sign(self.wrist_dz, DEAD_BAND_M),
            "elbow_dx": _sign(self.elbow_dx, DEAD_BAND_M),
            "elbow_dy": _sign(self.elbow_dy, DEAD_BAND_M),
            "elbow_dz": _sign(self.elbow_dz, DEAD_BAND_M),
            "flexion": _sign(self.flexion, DEAD_BAND_RAD),
        }

    @property
    def text(self) -> str:
        """The facts as the one line of ground truth the captioner is given."""
        return (
            f"wrist: {_vertical(self.wrist_dy)}, {_lateral(self.wrist_dx)}, "
            f"{_depth(self.wrist_dz)}; "
            f"elbow: {_vertical(self.elbow_dy)}, {_lateral(self.elbow_dx)}, "
            f"{_depth(self.elbow_dz)}; {_flexion(self.flexion)}"
        )


def motion_facts(
    clip: np.ndarray, context: ArmFeatureContext, n_prefix: int
) -> MotionFacts:
    """Measure *clip*'s described window: wrist, elbow and elbow-flexion travel."""
    positions = context.fk.fk_batch(
        arm_aa_from_state(clip, context), context.spine3_pos, context.spine3_aa
    )
    flexion = arm_feature_series(clip, context)["elbow_flexion"]
    start = n_prefix - 1
    wrist = positions[-1, WRIST_CHAIN_IDX] - positions[start, WRIST_CHAIN_IDX]
    elbow = positions[-1, ELBOW_CHAIN_IDX] - positions[start, ELBOW_CHAIN_IDX]
    return MotionFacts(
        wrist_dx=float(wrist[0]),
        wrist_dy=float(wrist[1]),
        wrist_dz=float(wrist[2]),
        elbow_dx=float(elbow[0]),
        elbow_dy=float(elbow[1]),
        elbow_dz=float(elbow[2]),
        flexion=float(flexion[-1] - flexion[start]),
    )


def asserted(caption: str) -> dict[str, int]:
    """Every axis *caption* claims a direction on, as ``quantity -> +1 / -1``.

    A direction word claims the axis of the part named most recently before it,
    so "move my hand toward my right and my elbow forward" is hand-inward and
    elbow-forward — nearest-by-distance would give "right" to the elbow, which is
    in the other clause. Words on both sides of one axis cancel: the line is then
    hedged rather than wrong.
    """
    parts = [(match.start(), "elbow") for match in _ELBOW_PART.finditer(caption)]
    parts += [(match.start(), "hand") for match in _HAND_PART.finditer(caption)]
    claims: dict[str, int] = {}
    for axis in AXES:
        positive = _claims(caption, axis.positive, axis.part, parts)
        negative = _claims(caption, axis.negative, axis.part, parts)
        if positive != negative:
            claims[axis.quantity] = 1 if positive else -1
    return claims


def caption_conflicts(caption: str, facts: MotionFacts) -> list[str]:
    """The axes on which *caption*'s direction words disagree with *facts*.

    A dead-band axis has sign ``0``, so asserting any direction on it conflicts:
    the clip does not move there and the line would teach the word on noise.
    """
    signs = facts.signs()
    return [
        quantity
        for quantity, claim in asserted(caption).items()
        if claim != signs[quantity]
    ]


def _claims(
    caption: str,
    pattern: re.Pattern[str],
    part: str | None,
    parts: list[tuple[int, str]],
) -> bool:
    """Whether *pattern* matches *caption* while talking about *part*."""
    matches = list(pattern.finditer(caption))
    if part is None:
        return bool(matches)
    return any(_nearest_part(match.start(), parts) == part for match in matches)


def _nearest_part(index: int, parts: list[tuple[int, str]]) -> str:
    """Which body part the word at *index* is about; the hand when none is named.

    English names the part before the direction ("raise my elbow"), so the last
    part mentioned before the word wins; a word ahead of every mention ("Lower
    your hand") takes the first one that follows.
    """
    before = [part for part in parts if part[0] < index]
    if before:
        return max(before, key=lambda part: part[0])[1]
    if parts:
        return min(parts, key=lambda part: part[0])[1]
    return "hand"


def _sign(delta: float, dead_band: float) -> int:
    """The direction of one axis of travel, dead-banded."""
    if abs(delta) <= dead_band:
        return 0
    return 1 if delta > 0 else -1


def _vertical(delta: float) -> str:
    """The vertical axis in words."""
    if abs(delta) <= DEAD_BAND_M:
        return "~0 up/down"
    return f"{abs(delta) * 100:.0f} cm {'up' if delta > 0 else 'down'}"


def _lateral(delta: float) -> str:
    """The width axis in words, glossed so "left" and "out" are one direction."""
    if abs(delta) <= DEAD_BAND_M:
        return "~0 left/right"
    side = (
        "to the person's left (outward, away from the body)"
        if delta > 0
        else "to the person's right (inward, toward the body)"
    )
    return f"{abs(delta) * 100:.0f} cm {side}"


def _depth(delta: float) -> str:
    """The depth axis in words."""
    if abs(delta) <= DEAD_BAND_M:
        return "~0 forward/back"
    return f"{abs(delta) * 100:.0f} cm {'forward' if delta > 0 else 'back'}"


def _flexion(delta: float) -> str:
    """The elbow angle in words."""
    if abs(delta) <= DEAD_BAND_RAD:
        return "elbow bend unchanged"
    degrees = abs(np.degrees(delta))
    if delta > 0:
        return f"elbow bends {degrees:.0f} degrees more"
    return f"elbow straightens {degrees:.0f} degrees"
