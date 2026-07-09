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
    COLLAR_BODY_POSE_INDEX,
    positions_to_smpl_body_pose,
    smpl_arm_aa_to_hml263_frame,
    smpl_body_pose_to_arm_aa,
    smpl_body_pose_to_collar_aa,
    smpl_body_pose_to_positions,
)
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    SmplLeftArmFK,
)

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
            arm_aa_4 = rng.uniform(-0.5, 0.5, (4, 3))
            collar_aa = arm_aa_4[0]
            arm_aa = arm_aa_4[1:]
            fk.collar_aa = collar_aa
            positions = fk.full_body_positions(arm_aa)  # (22, 3)

            body_pose = positions_to_smpl_body_pose(positions, fk.tpose_all_joints)
            recovered_arm_aa = smpl_body_pose_to_arm_aa(body_pose)
            recovered_collar_aa = smpl_body_pose_to_collar_aa(body_pose)

            original_arm_pos = fk.fk(arm_aa)  # (5, 3)
            fk.collar_aa = recovered_collar_aa
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
        collar_aa = np.array([0.3, 0.3, 0.3])
        arm_aa = np.array(
            [
                [0.0, -1.45, 0.0],  # left_shoulder
                [0.0, 0.0, 0.4],  # left_elbow
                [0.0, 0.0, 0.0],  # left_wrist
            ]
        )
        fk.collar_aa = collar_aa
        positions = fk.full_body_positions(arm_aa)
        body_pose = positions_to_smpl_body_pose(positions, fk.tpose_all_joints)
        recovered_arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        recovered_collar_aa = smpl_body_pose_to_collar_aa(body_pose)

        # Joint positions must match — axis-angles may differ (twist ambiguity).
        original_pos = fk.fk(arm_aa)
        fk.collar_aa = recovered_collar_aa
        np.testing.assert_allclose(
            fk.fk(recovered_arm_aa),
            original_pos,
            atol=1e-4,
        )

    def test_fixed_base_projection_preserves_target_bone_directions(
        self, fk: SmplLeftArmFK  # pylint: disable=redefined-outer-name
    ) -> None:
        """Project MDM XYZ into a different fixed collar/spine base."""
        mdm_spine_aa = np.array([0.2, -0.1, 0.05])
        mdm_collar_aa = np.array([0.15, -0.05, 0.1])
        target_arm_aa = np.array(
            [
                [0.0, -1.0, 0.2],
                [0.1, 0.2, 0.7],
                [-0.1, 0.05, 0.2],
            ],
            dtype=np.float64,
        )
        fk.collar_aa = mdm_collar_aa
        mdm_positions = fk.full_body_positions(
            target_arm_aa,
            fk.tpose_spine3_pos,
            mdm_spine_aa,
        )

        fixed_spine_aa = np.array([-0.1, 0.15, 0.05])
        fixed_collar_aa = np.array([-0.2, 0.05, -0.1])
        fk.collar_aa = fixed_collar_aa
        projected_arm_aa = fk.arm_aa_from_positions(
            mdm_positions,
            spine3_aa=fixed_spine_aa,
        )
        projected_positions = fk.fk(
            projected_arm_aa,
            fk.tpose_spine3_pos,
            fixed_spine_aa,
        )

        mdm_chain = mdm_positions[LEFT_ARM_CHAIN_INDICES]
        mdm_controlled_bones = np.diff(mdm_chain, axis=0)[1:]
        projected_controlled_bones = np.diff(projected_positions, axis=0)[1:]
        mdm_dirs = mdm_controlled_bones / np.linalg.norm(
            mdm_controlled_bones, axis=1, keepdims=True
        )
        projected_dirs = projected_controlled_bones / np.linalg.norm(
            projected_controlled_bones, axis=1, keepdims=True
        )
        np.testing.assert_allclose(projected_dirs, mdm_dirs, atol=1e-5)


class TestSmplBodyPoseToArmAa:
    """Unit tests for smpl_body_pose_to_arm_aa."""

    def test_output_shape(self) -> None:
        """Check that the output shape is (3, 3) for a single frame."""
        body_pose = np.zeros((21, 3))
        result = smpl_body_pose_to_arm_aa(body_pose)
        assert result.shape == (3, 3)

    def test_zeros_in_zeros_out(self) -> None:
        """Zero body_pose should yield zero arm axis-angles."""
        body_pose = np.zeros((21, 3))
        result = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(result, 0.0)

    def test_batched_shape(self) -> None:
        """Check that batched input gives (N, 3, 3) output."""
        body_pose = np.zeros((10, 21, 3))
        result = smpl_body_pose_to_arm_aa(body_pose)
        assert result.shape == (10, 3, 3)

    def test_collar_is_separate(self) -> None:
        """Collar is no longer part of MPC-controlled arm_aa."""
        body_pose = np.zeros((21, 3))
        body_pose[COLLAR_BODY_POSE_INDEX] = [0.1, 0.2, 0.3]  # collar
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        collar_aa = smpl_body_pose_to_collar_aa(body_pose)
        np.testing.assert_allclose(arm_aa, 0.0)
        np.testing.assert_allclose(collar_aa, [0.1, 0.2, 0.3])

    def test_shoulder_index(self) -> None:
        """Check that shoulder joint is correctly extracted."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[0]] = [0.4, 0.5, 0.6]  # shoulder
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[0], [0.4, 0.5, 0.6])

    def test_elbow_index(self) -> None:
        """Check that elbow joint is correctly extracted."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[1]] = [0.7, 0.8, 0.9]  # elbow
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[1], [0.7, 0.8, 0.9])

    def test_wrist_index(self) -> None:
        """Check that wrist joint is correctly extracted."""
        body_pose = np.zeros((21, 3))
        body_pose[ARM_BODY_POSE_INDICES[2]] = [1.0, 1.1, 1.2]  # wrist
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        np.testing.assert_allclose(arm_aa[2], [1.0, 1.1, 1.2])

    def test_collar_vs_wrist_differ(self) -> None:
        """Distinct joints should produce distinct outputs."""
        collar_bp = np.zeros((21, 3))
        collar_bp[COLLAR_BODY_POSE_INDEX] = [0.3, 0.0, 0.0]
        wrist_bp = np.zeros((21, 3))
        wrist_bp[ARM_BODY_POSE_INDICES[2]] = [0.3, 0.0, 0.0]
        assert not np.allclose(
            smpl_body_pose_to_arm_aa(collar_bp),
            smpl_body_pose_to_arm_aa(wrist_bp),
        )


@pytest.mark.skipif(
    not _SMPL_PKL.exists(),
    reason="SMPL_NEUTRAL.pkl not available",
)
class TestSmplArmAaToHml263FrameRoundtrip:
    """The patched arm must survive an official encode → decode roundtrip.

    Encodes a T-pose base frame with the official HumanML3D pipeline, patches
    in a non-trivial arm configuration, then decodes the result back to
    positions (``recover_from_ric``) and checks the controlled arm bone
    directions against repo FK of the patched body_pose.  Twist is not
    preserved by the position pipeline, so directions — not axis-angles — are
    the invariant; and the official encoder re-faces the body to Z+ using
    hips + shoulders (so moving the arm shifts the canonical heading), so
    directions are compared after yaw-aligning on the hip line.
    """

    @staticmethod
    def _hip_yaw(positions: np.ndarray) -> float:
        across = positions[2] - positions[1]  # r_hip - l_hip
        return float(np.arctan2(across[0], across[2]))

    def test_patched_arm_survives_official_roundtrip(
        self, fk: SmplLeftArmFK  # pylint: disable=redefined-outer-name
    ) -> None:
        # pylint: disable=import-outside-toplevel
        import torch

        from uncertain_feedback.data_collection.smpl_to_hml263 import (
            positions_to_hml263,
        )
        from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
            _import_recover_from_ric,
        )

        recover_from_ric = _import_recover_from_ric()
        mean, std = np.zeros(263), np.ones(263)  # identity normalization

        base_positions = fk.tpose_all_joints  # (22, 3)
        base_frame = positions_to_hml263(
            np.repeat(base_positions[None], 2, axis=0), mean, std
        )[0].astype(np.float64)

        arm_aa = np.array(
            [[0.0, -0.8, 0.3], [0.0, 0.5, 0.0], [0.1, 0.0, 0.2]], dtype=np.float64
        )
        out = smpl_arm_aa_to_hml263_frame(base_frame, arm_aa, mean, std, fk)

        decoded_positions = (
            recover_from_ric(
                torch.tensor(out, dtype=torch.float32).unsqueeze(0), 22
            )[0]
            .numpy()
            .astype(np.float64)
        )

        # Expected: base positions decoded → IK → arm slots patched → FK.
        base_decoded = (
            recover_from_ric(
                torch.tensor(base_frame, dtype=torch.float32).unsqueeze(0), 22
            )[0]
            .numpy()
            .astype(np.float64)
        )
        body_pose = positions_to_smpl_body_pose(base_decoded, fk.tpose_all_joints)
        body_pose[ARM_BODY_POSE_INDICES] = arm_aa
        expected_positions = smpl_body_pose_to_positions(
            body_pose, fk.tpose_all_joints, root_pos=base_decoded[0]
        )

        from scipy.spatial.transform import Rotation

        yaw_delta = self._hip_yaw(expected_positions) - self._hip_yaw(
            decoded_positions
        )
        unyaw = Rotation.from_euler("y", yaw_delta)
        for parent_j, child_j in [(13, 16), (16, 18), (18, 20)]:
            got = unyaw.apply(
                decoded_positions[child_j] - decoded_positions[parent_j]
            )
            want = expected_positions[child_j] - expected_positions[parent_j]
            got = got / np.linalg.norm(got)
            want = want / np.linalg.norm(want)
            angle = np.degrees(np.arccos(np.clip(got @ want, -1.0, 1.0)))
            assert angle < 3.0, f"bone {parent_j}->{child_j} off by {angle:.2f} deg"
