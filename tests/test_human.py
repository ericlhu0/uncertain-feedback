"""Tests for the immutable Human: construction, history, immutability, geometry."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import pickle

import numpy as np
import pytest

from uncertain_feedback.consts import MDM_START_POSE_PATH
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import HML_STATS_DIR
from uncertain_feedback.planners.mpc.arm_features import (
    arm_feature_series,
    arm_q_from_features,
)
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    Q_CLAVICLE,
    SmplLeftArmFK,
    q_reaching_wrist,
    q_to_arm_aa,
)

_ARM = np.array(
    [
        [-0.2578, 0.1121, -0.4418],
        [0.6868, -0.18, -0.1100],
        [0.0356, -1.23, 0.7515],
    ]
)


def _arm_states() -> np.ndarray:
    q = np.zeros((4, 7))
    q[:, 3:6] = [[0.3, 0.1, -0.2], [0.8, -0.4, 0.1], [-0.2, 0.5, 0.6], [0.0, 0.0, 0.9]]
    q[:, 6] = [0.2, 0.9, 1.4, 0.6]
    return q


def test_tpose_human_is_the_fk_template() -> None:
    human = Human()
    fk = SmplLeftArmFK()

    np.testing.assert_array_equal(human.posture, fk.tpose_all_joints)
    np.testing.assert_array_equal(human.spine3_pos, fk.tpose_spine3_pos)
    np.testing.assert_array_equal(human.spine3_aa, np.zeros(3))
    np.testing.assert_array_equal(human.q, fk.arm_aa_to_q(np.zeros((3, 3))))
    assert human.history.shape == (1, 7)
    assert human.hml_pose is None


def test_arm_sets_the_start_state() -> None:
    human = Human(arm=_ARM)

    np.testing.assert_array_equal(human.q, SmplLeftArmFK().arm_aa_to_q(_ARM))
    with pytest.raises(ValueError):
        Human(arm=np.zeros((4, 3)))


def test_step_rewind_and_reset_return_new_humans() -> None:
    start = Human(arm=_ARM)
    frames = _arm_states()

    one = start.step(frames[0])
    many = one.step(frames[1:])

    assert start.history.shape == (1, 7)
    assert one.history.shape == (2, 7)
    np.testing.assert_array_equal(many.history[1:], frames)
    np.testing.assert_array_equal(many.q, frames[-1])
    np.testing.assert_array_equal(many.rewind(0).history, start.history)
    np.testing.assert_array_equal(many.rewind(2).q, frames[1])
    with pytest.raises(IndexError):
        many.rewind(5)

    reset = many.reset_human_with_q(frames[2])
    np.testing.assert_array_equal(reset.history, frames[2:3])
    assert reset.fk is start.fk
    np.testing.assert_array_equal(reset.posture, start.posture)


def test_human_cannot_be_mutated() -> None:
    human = Human(arm=_ARM).step(_arm_states())
    before = human.history

    with pytest.raises(AttributeError):
        human.history = before  # type: ignore[misc]
    with pytest.raises(AttributeError):
        human.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del human.q
    snapshot = human.history
    snapshot[0] = 99.0
    np.testing.assert_array_equal(human.history, before)
    # pylint: disable=protected-access
    with pytest.raises(ValueError):
        human._history[0] = 99.0
    with pytest.raises(ValueError):
        human._history.setflags(write=True)

    restored = pickle.loads(pickle.dumps(human))
    np.testing.assert_array_equal(restored.history, before)
    np.testing.assert_array_equal(restored.posture, human.posture)
    with pytest.raises(ValueError):
        restored._history.setflags(write=True)


def test_geometry_matches_the_kinematics_helpers() -> None:
    human = Human(arm=_ARM)
    q = _arm_states()
    context = MpcCostContext(
        fk=human.fk, spine3_pos=human.spine3_pos, spine3_aa=human.spine3_aa
    )

    positions = human.fk_positions_from_q(q)
    expected = human.fk.fk_batch(
        q_to_arm_aa(q, human.fk.elbow_hinge_axis), human.spine3_pos, human.spine3_aa
    )
    np.testing.assert_allclose(positions, expected)
    assert human.fk_positions_from_q(q.reshape((2, 2, 7))).shape == (2, 2, 5, 3)
    np.testing.assert_allclose(human.wrist_from_q(q), expected[:, 4] - human.spine3_pos)
    np.testing.assert_allclose(
        human.fk_positions_from_q(human.ik_q_from_positions(positions)),
        positions,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        human.q_from_arm_aa(human.arm_aa_from_q(q)), q, atol=1e-12
    )

    features = human.features_from_q(q)
    for name, values in arm_feature_series(q, context).items():
        np.testing.assert_allclose(features[name], values)
    rows = np.stack(list(features.values()), axis=-1)
    np.testing.assert_allclose(
        human.q_from_features(rows),
        arm_q_from_features(rows, human.q[Q_CLAVICLE], context),
    )

    target = human.wrist_from_q(q[1])
    np.testing.assert_allclose(
        human.q_from_wrist(target),
        q_reaching_wrist(
            human.fk,
            target + human.spine3_pos,
            human.q,
            human.spine3_pos,
            human.spine3_aa,
        ),
    )


@pytest.mark.skipif(
    not (HML_STATS_DIR / "Mean.npy").exists(),
    reason="HumanML3D normalization statistics not available",
)
def test_pose_file_decodes_a_consistent_body() -> None:
    human = Human(pose=MDM_START_POSE_PATH)

    assert human.hml_pose is not None and human.hml_pose.shape == (263,)
    np.testing.assert_array_equal(human.spine3_pos, human.posture[9])
    assert np.linalg.norm(human.fk.collar_aa) > 0.0
    np.testing.assert_allclose(
        human.fk_positions_from_q(human.q),
        human.posture[LEFT_ARM_CHAIN_INDICES],
        atol=1e-6,
    )
