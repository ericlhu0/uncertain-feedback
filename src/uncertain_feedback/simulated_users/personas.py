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
        "Post-stroke flexor synergy during unsupported active or active-assisted "
        "movement: shoulder-abduction effort couples with involuntary elbow "
        "flexion. This elevation-based bound is a pose-only proxy for antigravity "
        "effort and does not apply to a fully passive, relaxed arm. Required "
        "elbow flexion rises linearly from 0 at ~66 deg elevation "
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

OUT_OF_SYNERGY_REACH_PREFERENCE = SimulatedUser(
    name="out_of_synergy_reach_preference",
    description=(
        "Soft rehabilitation preference, not a passive joint limit: the user "
        "wants to practice an upward/outward reach outside the post-stroke "
        "flexor synergy, combining greater shoulder elevation with progressively "
        "more elbow extension. The inverse coupling is a simplified task goal, "
        "not a universal requirement for every elevated-arm activity."
    ),
    feedback_text="as you lift my arm, straighten my elbow more",
    bounds=(
        CoupledBound(
            feature="elbow_flexion",
            bound_type="upper_bound",
            cond_feature="shoulder_elevation",
            intercept=2.4,
            slope=-0.65,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

TRICEPS_LONG_HEAD_CONTRACTURE = SimulatedUser(
    name="triceps_long_head_contracture",
    description=(
        "Shortened long head of the triceps, as may contribute to an elbow "
        "extension contracture in arthrogryposis or after spastic elbow "
        "extension: shoulder elevation and elbow flexion lengthen this "
        "two-joint muscle together, so the maximum comfortable elbow flexion "
        "decreases as the arm is raised."
    ),
    feedback_text="I can't bend my elbow that much while my arm is raised",
    bounds=(
        CoupledBound(
            feature="elbow_flexion",
            bound_type="upper_bound",
            cond_feature="shoulder_elevation",
            intercept=2.6,
            slope=-0.65,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

BICEPS_LONG_HEAD_CONTRACTURE = SimulatedUser(
    name="biceps_long_head_contracture",
    description=(
        "Shortened long head of the biceps: shoulder extension and elbow "
        "extension lengthen this two-joint muscle together, so the minimum "
        "comfortable elbow flexion increases as the upper arm moves behind "
        "the torso."
    ),
    feedback_text="don't straighten my elbow when you move my arm behind me",
    bounds=(
        CoupledBound(
            feature="elbow_flexion",
            bound_type="lower_bound",
            cond_feature="shoulder_flexion_extension",
            intercept=0.2,
            slope=-1.0,
        ),
    ),
    joint_limits=DEFAULT_ARM_JOINT_LIMITS,
)

BRACHIAL_PLEXUS_MECHANOSENSITIVITY = SimulatedUser(
    name="brachial_plexus_mechanosensitivity",
    description=(
        "Symptom-limited neural mechanosensitivity after brachial-plexus or "
        "peripheral-nerve injury: shoulder abduction and elbow extension "
        "increase neural loading together, so progressively more elbow flexion "
        "is required as the arm moves laterally. This is a coarse proxy because "
        "neck, wrist, forearm, and scapular pose also affect neural loading."
    ),
    feedback_text="keep my elbow bent when you move my arm out to the side",
    bounds=(
        CoupledBound(
            feature="elbow_flexion",
            bound_type="lower_bound",
            cond_feature="shoulder_abduction_adduction",
            intercept=-0.3,
            slope=1.0,
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
        OUT_OF_SYNERGY_REACH_PREFERENCE,
        TRICEPS_LONG_HEAD_CONTRACTURE,
        BICEPS_LONG_HEAD_CONTRACTURE,
        BRACHIAL_PLEXUS_MECHANOSENSITIVITY,
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
