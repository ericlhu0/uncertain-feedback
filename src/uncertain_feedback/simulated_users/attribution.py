"""Deterministic feedback attribution for the simulated user.

When the robot's motion triggers discomfort, the simulated user contrasts the
robot's nominal continuation against the window of its internal oracle path
(the MPC + hidden-cost rollout) it is currently closest to, and summarizes the
difference as signed joint-feature deltas plus Cartesian referent offsets. The
resulting :class:`CorrectionIntent` is the level-invariant input every
verbalizer phrases; dead-bands live in the verbalizers, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    arm_aa_from_state,
    arm_feature_series,
    canonical_arm_q,
)
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.kinematics import Q_DIM, SmplLeftArmFK

ATTRIBUTED_FEATURES = (
    "elbow_flexion",
    "shoulder_flexion_extension",
    "shoulder_abduction_adduction",
    "shoulder_elevation",
)

_ELBOW_CHAIN_IDX = 3
_WRIST_CHAIN_IDX = 4
_PELVIS_SMPL_IDX = 0
_HEAD_SMPL_IDX = 15


@dataclass(frozen=True)
class CorrectionIntent:
    """Signed contrast between the robot's nominal plan and the oracle window.

    ``feature_deltas`` are nominal − oracle window means in radians;
    ``wrist_offset`` and ``elbow_offset`` are nominal − oracle window-mean
    positions in world meters.
    """

    join_index: int
    feature_deltas: dict[str, float]
    wrist_offset: np.ndarray
    elbow_offset: np.ndarray


def _window_positions(context: MpcCostContext, window_q: np.ndarray) -> np.ndarray:
    arm_aa = arm_aa_from_state(window_q, context).reshape(-1, 3, 3)
    return context.fk.fk_batch(arm_aa, context.spine3_pos, context.spine3_aa)


def attribute_correction(
    oracle_path: np.ndarray,
    nominal_plan: np.ndarray,
    q_trigger: np.ndarray,
    context: MpcCostContext,
    min_join: int = 0,
) -> CorrectionIntent:
    """Attribute the trigger to a contrast against the nearest oracle window.

    The join point is the oracle waypoint at index ``>= min_join`` nearest to
    ``q_trigger`` in canonical q space; the oracle window starting there is
    compared against the whole ``nominal_plan`` (both clamped at the path end)
    by window-mean feature and referent-position differences.
    """
    oracle_q = canonical_arm_q(oracle_path, context).reshape(-1, Q_DIM)
    nominal_q = canonical_arm_q(nominal_plan, context).reshape(-1, Q_DIM)
    trigger_q = canonical_arm_q(q_trigger, context).reshape(Q_DIM)

    tail = oracle_q[min_join:]
    join = min_join + int(np.argmin(np.linalg.norm(tail - trigger_q, axis=-1)))
    window = oracle_q[join : join + nominal_q.shape[0]]

    nominal_features = arm_feature_series(nominal_q, context)
    oracle_features = arm_feature_series(window, context)
    deltas = {
        name: float(np.mean(nominal_features[name]) - np.mean(oracle_features[name]))
        for name in ATTRIBUTED_FEATURES
    }

    nominal_pos = _window_positions(context, nominal_q)
    oracle_pos = _window_positions(context, window)
    offsets = nominal_pos.mean(axis=0) - oracle_pos.mean(axis=0)
    return CorrectionIntent(
        join_index=join,
        feature_deltas=deltas,
        wrist_offset=offsets[_WRIST_CHAIN_IDX],
        elbow_offset=offsets[_ELBOW_CHAIN_IDX],
    )


def has_feedback_content(
    intent: CorrectionIntent, feature_dead_band: float = 0.15
) -> bool:
    """Level-invariant termination check: any feature delta above the dead-band."""
    return any(
        abs(delta) > feature_dead_band for delta in intent.feature_deltas.values()
    )


def assert_axis_conventions(fk: SmplLeftArmFK) -> None:
    """Verify the world-frame axis conventions the verbalizers' words assume.

    Y is up (T-pose head above pelvis) and +x is lateral for the left arm
    (T-pose wrist lateral of the elbow); z is the remaining depth axis.
    """
    arm = fk.tpose_joints
    if not arm[_WRIST_CHAIN_IDX, 0] > arm[_ELBOW_CHAIN_IDX, 0]:
        raise AssertionError("Expected T-pose left wrist lateral of elbow along +x.")
    body = fk.tpose_all_joints
    if not body[_HEAD_SMPL_IDX, 1] > body[_PELVIS_SMPL_IDX, 1]:
        raise AssertionError("Expected T-pose head above pelvis along +y.")
