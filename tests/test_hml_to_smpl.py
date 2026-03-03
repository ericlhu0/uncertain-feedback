# pylint: disable=duplicate-code
"""Tests for hml_smpl_conversion.py — HumanML3D 263-dim ↔ SMPL body_pose
conversion.

Group A  (no MDM, no GPU):
    Tests for ``positions_to_smpl_body_pose`` and ``smpl_body_pose_to_arm_aa``
    using only numpy/scipy and the SMPL neutral model PKL.

Group B  (requires MDM model + CUDA):
    Tests for the full ``hml263_to_smpl_body_pose`` pipeline.
    Skipped automatically when CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    ARM_BODY_POSE_INDICES,
    positions_to_smpl_body_pose,
    smpl_body_pose_to_arm_aa,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

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


# ---------------------------------------------------------------------------
# Group B — full pipeline tests (requires MDM model + CUDA)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not available")


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available — skipping MDM integration tests",
)
class TestHml263ToSmplBodyPose:
    """Integration tests for hml263_to_smpl_body_pose.

    Requires GPU and the MDM model weights to be present. Run with: uv
    run --extra develop pytest -m slow
    """

    @pytest.fixture(scope="class")
    def mdm_resources(
        self,
        fk: SmplLeftArmFK,  # pylint: disable=redefined-outer-name,unused-argument
    ):
        """Load MDM model + dataset once for all tests in this class."""
        # pylint: disable=import-outside-toplevel,too-many-locals
        import os
        import sys as _sys
        from pathlib import Path as _Path

        from uncertain_feedback.consts import MDM_MODEL_WEIGHTS_PATH, MDM_ROOT
        from uncertain_feedback.motion_generators.mdm.mdm_parser_util import edit_args

        _mdm_dir = MDM_ROOT / "motion-diffusion-model"
        if str(_mdm_dir) not in _sys.path:
            _sys.path.insert(0, str(_mdm_dir))

        os.chdir(_mdm_dir)

        # pylint: disable=import-error
        from data_loaders.get_data import get_dataset_loader
        from utils import dist_util
        from utils.fixseed import fixseed
        from utils.model_util import create_model_and_diffusion, load_saved_model
        from utils.sampler_util import ClassifierFreeSampleModel

        _orig_argv = _sys.argv
        _sys.argv = [
            "test",
            "--model_path",
            str(MDM_MODEL_WEIGHTS_PATH),
            "--text_condition",
            "",
        ]
        try:
            args = edit_args()  # type: ignore[no-untyped-call]
        finally:
            _sys.argv = _orig_argv

        args.batch_size = 1
        args.num_samples = 1
        args.num_repetitions = 1
        args.no_video = True

        fixseed(args.seed)
        dist_util.setup_dist(args.device)

        data = get_dataset_loader(
            name="humanml",
            batch_size=1,
            num_frames=196,
            split="test",
            hml_mode="text_only",
            fixed_len=0,
            pred_len=0,
            device=dist_util.dev(),
        )
        model, _ = create_model_and_diffusion(args, data)
        load_saved_model(model, args.model_path, use_avg=args.use_ema)
        model = ClassifierFreeSampleModel(model)
        model.to(dist_util.dev())
        model.eval()

        sitting_pose = torch.load(
            MDM_ROOT / "sitting_pose.pt", map_location=dist_util.dev()
        )  # (263, 1)
        return {"dataset": data, "model": model, "sitting_pose": sitting_pose}

    def test_output_shape(
        self,
        fk: SmplLeftArmFK,  # pylint: disable=redefined-outer-name
        mdm_resources: dict,
    ) -> None:
        """Check that the output shape is (1, 21, 3) for a single sitting
        frame."""
        from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (  # pylint: disable=import-outside-toplevel
            hml263_to_smpl_body_pose,
        )

        sitting = mdm_resources["sitting_pose"].squeeze(-1).unsqueeze(0)  # (1, 263)
        result = hml263_to_smpl_body_pose(
            sitting,
            mdm_resources["dataset"],
            mdm_resources["model"],
            fk.tpose_all_joints,
        )
        assert result.shape == (1, 21, 3)

    def test_arm_aa_shape_and_nonzero(
        self,
        fk: SmplLeftArmFK,  # pylint: disable=redefined-outer-name
        mdm_resources: dict,
    ) -> None:
        """Check arm axis-angles are non-zero for the sitting pose."""
        from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (  # pylint: disable=import-outside-toplevel
            hml263_to_smpl_body_pose,
        )

        sitting = mdm_resources["sitting_pose"].squeeze(-1).unsqueeze(0)  # (1, 263)
        body_pose = hml263_to_smpl_body_pose(
            sitting,
            mdm_resources["dataset"],
            mdm_resources["model"],
            fk.tpose_all_joints,
        )
        arm_aa = smpl_body_pose_to_arm_aa(body_pose[0])
        assert arm_aa.shape == (4, 3)
        # Sitting pose has the arm in a non-trivial position (not T-pose).
        assert not np.allclose(arm_aa, 0.0, atol=0.01)
