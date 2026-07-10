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
    JointBoxLimit,
    SimulatedUser,
)

# Anatomical box limits shared by every persona (radians, per axis-angle
# component of the controlled slots). The left_shoulder slot drives the
# clavicle under the repo FK convention, so it gets a tight box around the
# seated demo pose neutral of ~[-0.26, 0.05, -0.44] (demo_pose.pt and
# demo_pose_v3.pt decode to nearly the same clavicle; real clavicle range of
# motion is ~±20-30 deg); the upper-arm (left_elbow slot) and forearm
# (left_wrist slot) rotations get generous ball/hinge-joint boxes.
DEFAULT_ARM_JOINT_LIMITS = (
    JointBoxLimit(
        joint="left_shoulder", low=(-0.7, -0.4, -0.85), high=(0.15, 0.4, 0.05)
    ),
    JointBoxLimit(joint="left_elbow", low=(-2.0, -2.0, -2.0), high=(2.0, 2.0, 2.0)),
    JointBoxLimit(joint="left_wrist", low=(-2.2, -2.2, -2.2), high=(2.2, 2.2, 2.2)),
)

UNRESTRICTED = SimulatedUser(
    name="unrestricted",
    description="No movement restrictions.",
    feedback_text="",
    bounds=(),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

ADHESIVE_CAPSULITIS = SimulatedUser(
    name="adhesive_capsulitis",
    description=(
        "Frozen shoulder (adhesive capsulitis): painful restriction of "
        "glenohumeral elevation in every plane; comfortable elevation "
        "limited to ~72 deg, below the ~90 deg typical of stage-2 "
        "capsulitis."
    ),
    feedback_text="keep my arm down and close to my body",
    bounds=(
        HiddenBound(
            feature="shoulder_elevation",
            bound_type="upper_bound",
            high=1.25,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
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
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

PAINFUL_ARC = SimulatedUser(
    name="painful_arc",
    description=(
        "Subacromial painful arc: pain specifically in the ~60-120 deg "
        "elevation range in any plane; comfortable below and above it."
    ),
    feedback_text="lifting my arm partway up hurts",
    bounds=(
        HiddenBound(
            feature="shoulder_elevation",
            bound_type="avoid_band",
            low=1.05,
            high=2.1,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

STROKE_FLEXOR_SYNERGY = SimulatedUser(
    name="stroke_flexor_synergy",
    description=(
        "Post-stroke flexor synergy: raising the arm couples with "
        "involuntary elbow flexion; the higher the arm is elevated (in any "
        "plane), the more the elbow must stay bent. Pose-dependent bound: "
        "required elbow flexion rises linearly from 0 at ~66 deg elevation "
        "to ~90 deg bend at ~96 deg elevation."
    ),
    feedback_text="keep my elbow bent, don't straighten it while you lift my arm",
    bounds=(
        CoupledBound(
            feature="elbow_flexion",
            bound_type="lower_bound",
            cond_feature="shoulder_elevation",
            intercept=-3.45,
            slope=3.0,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

CROSS_BODY_PAIN = SimulatedUser(
    name="cross_body_pain",
    description=(
        "Acromioclavicular joint pain: moving the arm across the chest while "
        "it is elevated compresses the AC joint and hurts (the cross-body "
        "adduction test), and the pain grows the farther across and the "
        "higher the arm is carried; reaching up in front or out to the side, "
        "or carrying the arm across low, is comfortable. Pose-dependent "
        "bound: the tolerable elevation drops linearly as the upper arm "
        "adducts past the midline (~2.2 rad allowed at neutral, ~0 by "
        "~0.5 rad of adduction)."
    ),
    feedback_text="bring my arm up in front of me, don't drag it across my chest",
    bounds=(
        CoupledBound(
            feature="shoulder_elevation",
            bound_type="upper_bound",
            cond_feature="shoulder_abduction_adduction",
            intercept=2.2,
            slope=4.5,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

PERSONAS: dict[str, SimulatedUser] = {
    user.name: user
    for user in (
        UNRESTRICTED,
        ADHESIVE_CAPSULITIS,
        ELBOW_CONTRACTURE,
        PAINFUL_ARC,
        STROKE_FLEXOR_SYNERGY,
        CROSS_BODY_PAIN,
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
