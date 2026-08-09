"""Tests for simulated users: hidden bounds, features, and violations."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import dataclasses

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import MOTION_FPS, feature_series
from uncertain_feedback.simulated_users.personas import (
    BICEPS_LONG_HEAD_CONTRACTURE,
    BRACHIAL_PLEXUS_MECHANOSENSITIVITY,
    CROSS_BODY_PAIN,
    MORNING_SHOULDER_STIFFNESS,
    OUT_OF_SYNERGY_REACH_PREFERENCE,
    PAINFUL_ARC,
    SPASTIC_ELBOW_FLEXORS,
    TRICEPS_LONG_HEAD_CONTRACTURE,
    UNRESTRICTED,
)


def test_painful_arc_uses_plane_agnostic_elevation() -> None:
    bound = PAINFUL_ARC.bounds[0]

    assert bound.feature == "shoulder_elevation"
    assert bound.bound_type == "avoid_band"
    assert bound.low == 1.05  # type: ignore[union-attr]
    assert bound.high == 2.1  # type: ignore[union-attr]

    features = {
        "shoulder_elevation": np.array([0.5, 1.57, 2.5]),
        "shoulder_abduction_adduction": np.array([0.0, 0.0, 0.0]),
    }

    violations = PAINFUL_ARC.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_cross_body_pain_allowance_drops_with_adduction() -> None:
    bound = CROSS_BODY_PAIN.bounds[0]

    assert bound.feature == "shoulder_elevation"
    assert bound.cond_feature == "shoulder_abduction_adduction"  # type: ignore[union-attr]

    features = {
        "shoulder_elevation": np.array([1.5, 1.5, 1.5, 0.3]),
        "shoulder_abduction_adduction": np.array([0.4, -0.2, -0.4, -0.4]),
    }

    violations = CROSS_BODY_PAIN.violation_series(features)

    assert violations[0] == 0.0  # elevated but lateral: unrestricted
    assert violations[1] > 0.0  # elevated at moderate adduction: hurts
    assert violations[2] > violations[1]  # deeper across: hurts more
    assert violations[3] == 0.0  # deep across but carried low: comfortable


def test_triceps_contracture_allows_less_elbow_flexion_when_elevated() -> None:
    features = {
        "elbow_flexion": np.array([1.5, 1.5, 0.8]),
        "shoulder_elevation": np.array([0.5, 2.0, 2.0]),
    }

    violations = TRICEPS_LONG_HEAD_CONTRACTURE.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_out_of_synergy_preference_favors_extension_when_elevated() -> None:
    features = {
        "elbow_flexion": np.array([1.5, 1.5, 0.8]),
        "shoulder_elevation": np.array([0.5, 2.0, 2.0]),
    }

    violations = OUT_OF_SYNERGY_REACH_PREFERENCE.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_biceps_contracture_allows_less_extension_behind_the_body() -> None:
    features = {
        "elbow_flexion": np.array([0.3, 0.3, 0.8]),
        "shoulder_flexion_extension": np.array([0.3, -0.5, -0.5]),
    }

    violations = BICEPS_LONG_HEAD_CONTRACTURE.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_neural_mechanosensitivity_requires_flexion_during_abduction() -> None:
    features = {
        "elbow_flexion": np.array([0.4, 0.4, 1.0]),
        "shoulder_abduction_adduction": np.array([0.2, 1.0, 1.0]),
    }

    violations = BRACHIAL_PLEXUS_MECHANOSENSITIVITY.violation_series(features)

    assert violations[0] == 0.0
    assert violations[1] > 0.0
    assert violations[2] == 0.0


def test_morning_stiffness_only_restricts_in_the_morning() -> None:
    elevations = np.array([1.5, 0.8])

    morning = MORNING_SHOULDER_STIFFNESS.violation_series(
        {"shoulder_elevation": elevations, "time_of_day": np.array([8.0, 8.0])}
    )
    afternoon = MORNING_SHOULDER_STIFFNESS.violation_series(
        {"shoulder_elevation": elevations, "time_of_day": np.array([14.0, 14.0])}
    )

    assert morning[0] > 0.0  # elevated in the morning: stiff
    assert morning[1] == 0.0  # carried low in the morning: fine
    assert np.all(afternoon == 0.0)  # same poses after loosening up: fine


def test_spastic_elbow_tolerates_fast_extension_only_when_flexed() -> None:
    features = {
        "elbow_flexion": np.array([0.4, 0.4, 2.0]),
        "elbow_flexion_velocity": np.array([-0.2, -1.0, -1.0]),
    }

    violations = SPASTIC_ELBOW_FLEXORS.violation_series(features)

    assert violations[0] == 0.0  # slow extension near full extension: fine
    assert violations[1] > 0.0  # fast extension near full extension: catch
    assert violations[2] == 0.0  # same speed while deeply flexed: fine


def test_feature_series_adds_velocities_and_session_clock() -> None:
    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    q = np.zeros((5, 7), dtype=np.float64)
    q[:, 6] = np.linspace(0.2, 1.0, 5)

    features = feature_series(context, q)

    assert "time_of_day" not in features
    np.testing.assert_allclose(
        features["elbow_flexion_velocity"],
        np.gradient(features["elbow_flexion"]) * MOTION_FPS,
    )
    assert np.all(features["elbow_flexion_velocity"] > 0.0)
    assert np.all(feature_series(context, q[:1])["elbow_flexion_velocity"] == 0.0)

    timed = feature_series(dataclasses.replace(context, time_of_day=8.5), q)
    np.testing.assert_allclose(timed["time_of_day"], 8.5)


def test_limit_cost_penalizes_out_of_box_rollouts() -> None:
    cost = UNRESTRICTED.limit_cost(weight=1.0)
    assert {limit.joint for limit in UNRESTRICTED.joint_limits} == {
        "left_elbow",
        "left_wrist",
    }
    elbow = UNRESTRICTED.joint_limits[0]

    inside = np.zeros((1, 2, 3, 3))
    outside = np.zeros((1, 2, 3, 3))
    outside[:, :, 1, 1] = elbow.high[1] + 0.3

    assert cost(inside)[0] == 0.0
    assert np.isclose(cost(outside)[0], 0.3)


def test_intent_aligned_chooser_picks_comfortable_aligned_candidate() -> None:
    from uncertain_feedback.simulated_users.attribution import CorrectionIntent
    from uncertain_feedback.simulated_users.chooser import choose_correction
    from uncertain_feedback.simulated_users.personas import ADHESIVE_CAPSULITIS

    fk = SmplLeftArmFK()
    context = MpcCostContext(
        fk=fk, spine3_pos=fk.tpose_spine3_pos, spine3_aa=np.zeros(3)
    )
    intent = CorrectionIntent(
        join_index=0,
        feature_deltas={
            "elbow_flexion": -0.4,
            "shoulder_flexion_extension": 0.0,
            "shoulder_abduction_adduction": 0.0,
            "shoulder_elevation": 0.0,
        },
        wrist_offset=np.zeros(3),
        elbow_offset=np.zeros(3),
    )
    frames = 6
    ramp = np.linspace(0.0, 1.0, frames)
    aligned = np.zeros((frames, 7))
    aligned[:, 6] = 0.5 * ramp  # bends the elbow hinge
    orthogonal = np.zeros((frames, 7))
    orthogonal[:, 4] = 0.2 * ramp  # small upper-arm rotation, no elbow bend
    painful = np.zeros((frames, 7))
    painful[:, 6] = 0.5 * ramp
    painful[:, 3] = 2.6 * ramp  # exceeds the anatomical joint box
    candidates = {0: aligned, 1: orthogonal, 2: painful}
    oracle = np.zeros((4, 7))

    choice = choose_correction(
        ADHESIVE_CAPSULITIS,
        context,
        candidates,
        oracle,
        mode="intent_aligned",
        intent=intent,
    )
    assert choice.label == 0
    assert choice.acceptable[2] is False
    assert choice.alignment[0] > choice.alignment[1]

    rng = np.random.default_rng(0)
    for _ in range(5):
        random_choice = choose_correction(
            ADHESIVE_CAPSULITIS,
            context,
            candidates,
            oracle,
            mode="random",
            rng=rng,
        )
        assert random_choice.label in (0, 1)
