# pylint: disable=duplicate-code
"""Tests for hml_smpl_conversion.py — HumanML3D 263-dim ↔ SMPL body_pose
conversion.

Tests for ``positions_to_smpl_body_pose`` and ``smpl_body_pose_to_arm_aa``
using only numpy/scipy and the SMPL neutral model PKL.
Skipped automatically when SMPL_NEUTRAL.pkl is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    ARM_BODY_POSE_INDICES,
    positions_to_smpl_body_pose,
    smpl_body_pose_to_arm_aa,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

_SMPL_PKL = (
    Path(__file__).parent.parent
    / "src"
    / "uncertain_feedback"
    / "motion_generators"
    / "mdm"
    / "motion-diffusion-model"
    / "body_models"
    / "smpl"
    / "SMPL_NEUTRAL.pkl"
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fk() -> SmplLeftArmFK:
    """Return a shared SmplLeftArmFK instance for the test module."""
    return SmplLeftArmFK()


# ---------------------------------------------------------------------------
# Group A — pure numpy/scipy tests (no MDM, no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SMPL_PKL.exists(),
    reason="SMPL_NEUTRAL.pkl not available",
)
class TestPositionsToSmplBodyPose:
    """Unit tests for positions_to_smpl_body_pose."""

    def test_output_shape(
        self, fk: SmplLeftArmFK  # pylint: disable=redefined-outer-name
    ) -> None:
        """Check that the output shape is (21, 3)."""
        result = positions_to_smpl_body_pose(fk.tpose_all_joints, fk.tpose_all_joints)
        assert result.shape == (21, 3)

    def test_tpose_gives_zero_body_pose(
        self, fk: SmplLeftArmFK  # pylint: disable=redefined-outer-name
    ) -> None:
        """T-pose positions require no rotation; body_pose should be all
        zeros."""
        result = positions_to_smpl_body_pose(fk.tpose_all_joints, fk.tpose_all_joints)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_fk_roundtrip_random_arm(
        self, fk: SmplLeftArmFK  # pylint: disable=redefined-outer-name
    ) -> None:
        """IK(FK(arm_aa)) → recovered arm positions match original arm
        positions."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            arm_aa = rng.uniform(-0.5, 0.5, (4, 3))
            positions = fk.full_body_positions(arm_aa)  # (22, 3)

            body_pose = positions_to_smpl_body_pose(positions, fk.tpose_all_joints)
            recovered_arm_aa = smpl_body_pose_to_arm_aa(body_pose)

            original_arm_pos = fk.fk(arm_aa)  # (5, 3)
            recovered_arm_pos = fk.fk(recovered_arm_aa)  # (5, 3)
            np.testing.assert_allclose(recovered_arm_pos, original_arm_pos, atol=1e-4)

    def test_fk_roundtrip_realistic_pose(
        self, fk: SmplLeftArmFK  # pylint: disable=redefined-outer-name
    ) -> None:
        """Roundtrip for a realistic arm pose from the MPC demo.

        Minimum-rotation IK does not preserve twist, so recovered axis-
        angles will generally differ from the originals.  The invariant
        we check is that FK(recovered_aa) reproduces the same joint
        positions as FK(arm_aa).
        """
        arm_aa = np.array(
            [
                [0.3, 0.3, 0.3],  # left_collar
                [0.0, -1.45, 0.0],  # left_shoulder
                [0.0, 0.0, 0.4],  # left_elbow
                [0.0, 0.0, 0.0],  # left_wrist
            ]
        )
        positions = fk.full_body_positions(arm_aa)
        body_pose = positions_to_smpl_body_pose(positions, fk.tpose_all_joints)
        recovered_arm_aa = smpl_body_pose_to_arm_aa(body_pose)

        # Joint positions must match — axis-angles may differ (twist ambiguity).
        np.testing.assert_allclose(fk.fk(recovered_arm_aa), fk.fk(arm_aa), atol=1e-4)


class TestSmplBodyPoseToArmAa:
    """Unit tests for smpl_body_pose_to_arm_aa."""

    def test_output_shape(self) -> None:
        """Check that the output shape is (4, 3) for a single frame."""
        body_pose = np.zeros((21, 3))
        result = smpl_body_pose_to_arm_aa(body_pose)
        assert result.shape == (4, 3)

    def test_zeros_in_zeros_out(self) -> None:
        """Zero body_pose should yield zero arm axis-angles."""
        body_pose = np.zeros((21, 3))
        result = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(result, 0.0)

    def test_batched_shape(self) -> None:
        """Check that batched input gives (N, 4, 3) output."""
        body_pose = np.zeros((10, 21, 3))
        result = smpl_body_pose_to_arm_aa(body_pose)
        assert result.shape == (10, 4, 3)

    def test_collar_index(self) -> None:
        """smpl_body_pose_to_arm_aa[0] should be left_collar
        (body_pose[12])."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[0]] = [0.1, 0.2, 0.3]  # collar
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(arm_aa[1:], 0.0)

    def test_shoulder_index(self) -> None:
        """Check that shoulder joint is correctly extracted."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[1]] = [0.4, 0.5, 0.6]  # shoulder
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[1], [0.4, 0.5, 0.6])

    def test_elbow_index(self) -> None:
        """Check that elbow joint is correctly extracted."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[2]] = [0.7, 0.8, 0.9]  # elbow
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[2], [0.7, 0.8, 0.9])

    def test_wrist_index(self) -> None:
        """Check that wrist joint is correctly extracted."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[3]] = [1.0, 1.1, 1.2]  # wrist
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[3], [1.0, 1.1, 1.2])

    def test_collar_vs_wrist_differ(self) -> None:
        """Distinct joints should produce distinct outputs."""
        collar_bp = np.zeros((21, 3))
        collar_bp[ARM_BODY_POSE_INDICES[0]] = [0.3, 0.0, 0.0]
        wrist_bp = np.zeros((21, 3))
        wrist_bp[ARM_BODY_POSE_INDICES[3]] = [0.3, 0.0, 0.0]
        assert not np.allclose(
            smpl_body_pose_to_arm_aa(collar_bp),
            smpl_body_pose_to_arm_aa(wrist_bp),
        )


