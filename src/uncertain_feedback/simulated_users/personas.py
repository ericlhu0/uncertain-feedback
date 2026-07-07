"""Clinically motivated simulated-user personas.

Each persona encodes a documented movement restriction as hidden bounds over the
shared joint features (radians; degree equivalents in the descriptions). The
feature values are the coarse SMPL-space proxies from ``GeneratedCostContext``,
not clinical goniometry, so thresholds are chosen to be behaviorally meaningful
in this model rather than exact clinical norms.
"""

from __future__ import annotations

from uncertain_feedback.simulated_users.base import (
    CoupledBound,
    HiddenBound,
    SimulatedUser,
)

ADHESIVE_CAPSULITIS = SimulatedUser(
    name="adhesive_capsulitis",
    description=(
        "Frozen shoulder (adhesive capsulitis): painful restriction of "
        "glenohumeral elevation; comfortable abduction limited to ~63 deg, "
        "below the ~90 deg typical of stage-2 capsulitis."
    ),
    feedback_text="keep my arm closer to my body",
    bounds=(
        HiddenBound(
            feature="shoulder_abduction_adduction",
            bound_type="upper_bound",
            high=1.1,
        ),
    ),
)

ELBOW_CONTRACTURE = SimulatedUser(
    name="elbow_contracture",
    description=(
        "Elbow flexion contracture: the elbow cannot fully extend; a ~30 deg "
        "extension deficit is a common clinically significant contracture."
    ),
    feedback_text="don't straighten my elbow all the way",
    bounds=(
        HiddenBound(
            feature="elbow_flexion",
            bound_type="lower_bound",
            low=0.5,
        ),
    ),
)

PAINFUL_ARC = SimulatedUser(
    name="painful_arc",
    description=(
        "Subacromial painful arc: pain specifically in the ~60-120 deg "
        "abduction range; comfortable below and above it."
    ),
    feedback_text="lifting my arm out to the side partway up hurts",
    bounds=(
        HiddenBound(
            feature="shoulder_abduction_adduction",
            bound_type="avoid_band",
            low=1.05,
            high=2.1,
        ),
    ),
)

STROKE_FLEXOR_SYNERGY = SimulatedUser(
    name="stroke_flexor_synergy",
    description=(
        "Post-stroke flexor synergy: shoulder abduction couples with "
        "involuntary elbow flexion; the higher the arm is raised, the more the "
        "elbow must stay bent. Pose-dependent bound: required elbow flexion "
        "rises linearly from 0 at ~34 deg abduction to ~69 deg bend at ~92 deg "
        "abduction."
    ),
    feedback_text="keep my elbow bent while you lift my arm",
    bounds=(
        CoupledBound(
            feature="elbow_flexion",
            bound_type="lower_bound",
            cond_feature="shoulder_abduction_adduction",
            intercept=-0.72,
            slope=1.2,
        ),
    ),
)

PERSONAS: dict[str, SimulatedUser] = {
    user.name: user
    for user in (
        ADHESIVE_CAPSULITIS,
        ELBOW_CONTRACTURE,
        PAINFUL_ARC,
        STROKE_FLEXOR_SYNERGY,
    )
}


def get_persona(name: str) -> SimulatedUser:
    """Return a registered persona by name."""
    try:
        return PERSONAS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown persona {name!r}; expected one of {sorted(PERSONAS)}."
        ) from exc
