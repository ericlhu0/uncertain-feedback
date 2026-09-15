"""Short single-axis captions written straight from a clip's measured motion.

An alternative to the VLM captioner for stage (b) of the correction-clip pipeline.
Every line is generated from :class:`~...motion_facts.MotionFacts` by template, so
it is true of the clip by construction, and — unlike the VLM captioner's lines,
which average two axes and 10-14 words — it names one axis in at most ten words. The diffusion model conditions on a frozen CLIP text embedding, which is
close to a bag of words, so a multi-axis sentence splits its weight over every
direction it names while the test-time instructions are mostly single-axis.

Per motion: one imperative per axis outside the dead band in both grammatical
persons, one two-clause line joining the two axes that move most, and two synonym
paraphrases of the axis that moves most (also in both persons). Axes are ranked by
travel in dead-band units, so metres and radians compare. The elbow's depth axis
has no template — there is no short natural imperative for it — so it never
contributes a line and is never the dominant axis.

The direction words come from the same ``AXES`` table :mod:`motion_facts` checks
captions against, and a word is attributed to the part named last before it, so
the two-clause line puts its part before its direction word ("move my hand down",
not "lower my hand") — otherwise the second clause's direction would be read as
the first clause's part. Its clauses also avoid a bare "in", whose table entry
reads "in and" as the landmark phrase "in a ..." and so claims nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from uncertain_feedback.data_collection.dataset_auto_correction.motion_facts import (
    DEAD_BAND_M,
    DEAD_BAND_RAD,
    MotionFacts,
    asserted,
)


@dataclass(frozen=True)
class _Template:
    """The lines one direction of one axis contributes, in the first person.

    ``clause`` is the two-clause form: the part comes before the direction word so
    the word is attributed to it wherever the clause sits in the line. ``part`` is
    what the clause names, so a second clause about the same part can say "it".
    """

    imperative: str
    clause: str
    part: str
    paraphrases: tuple[str, str]


_TEMPLATES: dict[tuple[str, int], _Template] = {
    ("wrist_dy", 1): _Template(
        "Lift my hand up.",
        "move my hand up",
        "hand",
        ("Raise my hand.", "Move my hand upward."),
    ),
    ("wrist_dy", -1): _Template(
        "Lower my hand.",
        "move my hand down",
        "hand",
        ("Bring my hand down.", "Drop my hand lower."),
    ),
    ("wrist_dx", 1): _Template(
        "Move my hand out to the left.",
        "move my hand left",
        "hand",
        ("Move my hand outward.", "Move my hand away from my body."),
    ),
    ("wrist_dx", -1): _Template(
        "Move my hand in to the right.",
        "move my hand right",
        "hand",
        ("Bring my hand inward.", "Move my hand across my body."),
    ),
    ("wrist_dz", 1): _Template(
        "Move my hand forward.",
        "move my hand forward",
        "hand",
        ("Move my hand ahead.", "Reach my hand out in front."),
    ),
    ("wrist_dz", -1): _Template(
        "Move my hand back.",
        "move my hand back",
        "hand",
        ("Move my hand backward.", "Pull my hand behind me."),
    ),
    ("elbow_dy", 1): _Template(
        "Raise my elbow.",
        "move my elbow up",
        "elbow",
        ("Lift my elbow up.", "Move my elbow higher."),
    ),
    ("elbow_dy", -1): _Template(
        "Lower my elbow.",
        "move my elbow down",
        "elbow",
        ("Drop my elbow down.", "Move my elbow lower."),
    ),
    ("elbow_dx", 1): _Template(
        "Move my elbow outward.",
        "move my elbow out",
        "elbow",
        ("Move my elbow away.", "Swing my elbow out."),
    ),
    ("elbow_dx", -1): _Template(
        "Bring my elbow in.",
        "move my elbow inward",
        "elbow",
        ("Tuck my elbow in.", "Move my elbow across."),
    ),
    ("flexion", 1): _Template(
        "Bend my elbow more.",
        "bend my elbow more",
        "elbow",
        ("Flex my elbow.", "Curl my elbow more."),
    ),
    ("flexion", -1): _Template(
        "Straighten my elbow.",
        "straighten my elbow",
        "elbow",
        ("Extend my elbow.", "Unbend my elbow."),
    ),
}

_TEMPLATED: tuple[str, ...] = (
    "wrist_dy",
    "wrist_dx",
    "wrist_dz",
    "elbow_dy",
    "elbow_dx",
    "flexion",
)

_OWN_PART = re.compile(r"\bmy (?:hand|elbow)\b")
_MY = re.compile(r"\bmy\b")
_ME = re.compile(r"\bme\b")


def templated_captions(facts: MotionFacts) -> list[str]:
    """Every line *facts* generates, ordered imperatives then paraphrases.

    A clip whose every templated axis is inside the dead band has nothing true to
    say, but a motion with no text is not trainable, so it names the axis that
    moves most anyway — the same fallback the VLM path takes when every line
    contradicts.
    """
    signs = facts.signs()
    ranked = sorted(_TEMPLATED, key=lambda quantity: -_scaled(facts, quantity))
    active = [quantity for quantity in ranked if signs[quantity] != 0]
    if not active:
        active = ranked[:1]
        signs = {**signs, active[0]: 1 if _value(facts, active[0]) > 0 else -1}

    lines: list[str] = []
    for quantity in active:
        lines += _both_persons(_TEMPLATES[(quantity, signs[quantity])].imperative)
    if len(active) > 1:
        lines.append(
            _two_clause(
                _TEMPLATES[(active[0], signs[active[0]])],
                _TEMPLATES[(active[1], signs[active[1]])],
            )
        )
    for paraphrase in _TEMPLATES[(active[0], signs[active[0]])].paraphrases:
        lines += _both_persons(paraphrase)
    return lines


def second_person(line: str) -> str:
    """*line* addressed to the person whose arm it is rather than asked about."""
    return _MY.sub("your", _ME.sub("you", line))


def templated_report(records: list[tuple[list[str], MotionFacts]]) -> str:
    """What the generated text looks like over every encoded motion.

    The line ratios are what the loader draws from — one line per motion per epoch
    — so they are the text prior the model learns, and they are reported against
    the motion ratios they are supposed to mirror.
    """
    lines = [line for caption_lines, _ in records for line in caption_lines]
    first = sum(1 for line in lines if line != second_person(line))
    return (
        "\n=== templated captions ===\n"
        f"{len(records)} motions, {len(lines)} lines "
        f"({len(lines) / len(records):.1f} per motion), "
        f"{len(set(lines))} unique, "
        f"max {max(len(line.split()) for line in lines)} words\n"
        f"registers: {first} first person / {len(lines) - first} second person\n"
        f"up:down = {_ratio(records, 'wrist_dy')}\n"
        f"left:right = {_ratio(records, 'wrist_dx')}\n"
    )


def _both_persons(line: str) -> list[str]:
    """*line* and its opposite-person twin."""
    return [line, second_person(line)]


def _two_clause(first: _Template, second: _Template) -> str:
    """One line asking for both axes, the second clause pronominalized if it can be."""
    tail = second.clause
    if second.part == first.part:
        tail = _OWN_PART.sub("it", tail, count=1)
    return f"{first.clause[0].upper()}{first.clause[1:]} and {tail}."


def _value(facts: MotionFacts, quantity: str) -> float:
    """One axis's travel."""
    return float(getattr(facts, quantity))


def _scaled(facts: MotionFacts, quantity: str) -> float:
    """One axis's travel in dead-band units, so metres and radians compare."""
    band = DEAD_BAND_RAD if quantity == "flexion" else DEAD_BAND_M
    return abs(_value(facts, quantity)) / band


def _ratio(records: list[tuple[list[str], MotionFacts]], quantity: str) -> str:
    """The positive:negative line ratio on one axis, against the motion ratio."""
    claims = [asserted(line).get(quantity, 0) for lines, _ in records for line in lines]
    signs = [facts.signs()[quantity] for _, facts in records]
    positive_lines = sum(1 for claim in claims if claim > 0)
    negative_lines = sum(1 for claim in claims if claim < 0)
    positive = sum(1 for sign in signs if sign > 0)
    negative = sum(1 for sign in signs if sign < 0)
    return (
        f"{positive_lines}:{negative_lines} lines "
        f"({positive_lines / max(negative_lines, 1):.2f}) over "
        f"{positive}:{negative} motions ({positive / max(negative, 1):.2f})"
    )
