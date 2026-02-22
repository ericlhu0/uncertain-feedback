"""Forward kinematics for the SMPL left arm.

Loads T-pose bone offsets from the SMPL neutral model and computes joint
positions from axis-angle joint rotations.  Follows the same FK convention
as the MDM Skeleton class: the accumulated world rotation at each joint
(including that joint's own local rotation) transforms the outgoing bone.

Joint chain:
    spine3 (9) → left_collar (13) → left_shoulder (16) → left_elbow (18) → left_wrist (20)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Default path to the SMPL neutral model inside the MDM submodule
_SMPL_PKL_DEFAULT = (
    Path(__file__).parent.parent.parent
    / "motion_generators"
    / "mdm"
    / "motion-diffusion-model"
    / "body_models"
    / "smpl"
    / "SMPL_NEUTRAL.pkl"
)

# ---------------------------------------------------------------------------
# SMPL skeleton topology (22 joints, 0-21; hands excluded)
# ---------------------------------------------------------------------------

# Parent index for each of the 22 joints (-1 = root)
SMPL_PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# All (parent, child) bone pairs for the 22-joint skeleton
SMPL_BONE_PAIRS_22 = [(p, c) for c, p in enumerate(SMPL_PARENTS_22) if p >= 0]

# Left arm joints in the 22-joint skeleton (collar through wrist)
LEFT_ARM_JOINT_INDICES_22 = [13, 16, 18, 20]

# Bones that belong to the left arm (including the spine3→collar connection)
LEFT_ARM_BONE_PAIRS_22 = [(9, 13), (13, 16), (16, 18), (18, 20)]

# SMPL joint chain for the left arm FK (spine3 is the anchor)
LEFT_ARM_CHAIN_INDICES = [9, 13, 16, 18, 20]
LEFT_ARM_CHAIN_NAMES = ["spine3", "left_collar", "left_shoulder", "left_elbow", "left_wrist"]


class SmplLeftArmFK:
    """Forward kinematics for the SMPL left arm.

    Loads T-pose data from the SMPL neutral PKL file once at construction time.
    Subsequent FK calls are pure numpy/scipy operations.

    Args:
        smpl_pkl_path: Path to ``SMPL_NEUTRAL.pkl``.  Defaults to the copy
                       inside the MDM submodule.
    """

    def __init__(self, smpl_pkl_path: str | Path | None = None) -> None:
        pkl_path = Path(smpl_pkl_path) if smpl_pkl_path is not None else _SMPL_PKL_DEFAULT
        self._bone_offsets, self._tpose_joints, self._tpose_22 = self._load_from_pkl(pkl_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_pkl(
        pkl_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (bone_offsets, arm_chain_tpose, all_22_tpose).

        - ``bone_offsets``:     ``(4, 3)`` parent→child vectors for the arm.
        - ``arm_chain_tpose``:  ``(5, 3)`` T-pose positions of the 5 arm-chain joints.
        - ``all_22_tpose``:     ``(22, 3)`` T-pose positions of all 22 body joints.
        """
        with open(pkl_path, "rb") as f:
            dd = pickle.load(f, encoding="latin1")

        J_reg = dd["J_regressor"]
        if hasattr(J_reg, "todense"):
            J_reg = np.array(J_reg.todense())
        else:
            J_reg = np.array(J_reg)

        v = np.array(dd["v_template"])
        joints = J_reg @ v  # (24, 3) T-pose joint positions

        # Arm chain subset
        chain = LEFT_ARM_CHAIN_INDICES
        tpose_chain = joints[chain]           # (5, 3)
        bone_offsets = np.diff(tpose_chain, axis=0)  # (4, 3)

        # Full 22-joint subset (exclude hands at 22, 23)
        tpose_22 = joints[:22].copy()         # (22, 3)

        return bone_offsets, tpose_chain, tpose_22

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tpose_spine3_pos(self) -> np.ndarray:
        """T-pose world position of the spine3 joint ``(3,)``."""
        return self._tpose_joints[0].copy()

    @property
    def tpose_joints(self) -> np.ndarray:
        """T-pose positions of the 5 arm-chain joints ``(5, 3)``."""
        return self._tpose_joints.copy()

    @property
    def tpose_all_joints(self) -> np.ndarray:
        """T-pose positions of all 22 body joints ``(22, 3)``."""
        return self._tpose_22.copy()

    # ------------------------------------------------------------------
    # FK — arm only
    # ------------------------------------------------------------------

    def fk(
        self,
        arm_aa: np.ndarray,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint positions for the left arm.

        FK convention (matches MDM Skeleton class): the accumulated world
        rotation at joint *i* — which includes joint *i*'s own local rotation —
        transforms the bone that ends at joint *i*.

        Args:
            arm_aa:     ``(4, 3)`` axis-angle for
                        [left_collar, left_shoulder, left_elbow, left_wrist].
            spine3_pos: ``(3,)`` world position of spine3.  Defaults to the
                        SMPL T-pose spine3 position.
            spine3_aa:  ``(3,)`` world axis-angle of spine3.  Defaults to
                        identity (all zeros).

        Returns:
            ``(5, 3)`` world positions of
            [spine3, left_collar, left_shoulder, left_elbow, left_wrist].
        """
        arm_aa = np.asarray(arm_aa, dtype=np.float64)
        spine3_pos = (
            np.asarray(spine3_pos, dtype=np.float64)
            if spine3_pos is not None
            else self._tpose_joints[0]
        )
        spine3_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3)
        )

        positions = np.empty((5, 3), dtype=np.float64)
        positions[0] = spine3_pos

        T = Rotation.from_rotvec(spine3_aa)
        for i in range(4):
            T = T * Rotation.from_rotvec(arm_aa[i])
            positions[i + 1] = positions[i] + T.apply(self._bone_offsets[i])

        return positions

    def fk_batch(
        self,
        arm_aa: np.ndarray,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Batched FK over N arm configurations.

        Args:
            arm_aa:     ``(N, 4, 3)`` axis-angle arrays.
            spine3_pos: ``(3,)`` — same for all samples.
            spine3_aa:  ``(3,)`` — same for all samples.

        Returns:
            ``(N, 5, 3)`` world positions.
        """
        arm_aa = np.asarray(arm_aa, dtype=np.float64)
        N = arm_aa.shape[0]
        out = np.empty((N, 5, 3), dtype=np.float64)
        for i in range(N):
            out[i] = self.fk(arm_aa[i], spine3_pos, spine3_aa)
        return out

    # ------------------------------------------------------------------
    # FK — full body
    # ------------------------------------------------------------------

    def full_body_positions(
        self,
        arm_aa: np.ndarray,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return all 22 joint positions with the left arm updated by ``arm_aa``.

        All non-arm joints remain at their SMPL T-pose positions.  The five
        arm-chain joints (spine3, collar, shoulder, elbow, wrist) are
        recomputed via FK.

        Args:
            arm_aa:     ``(4, 3)`` axis-angle for the 4 controlled arm joints.
            spine3_pos: ``(3,)`` spine3 world position.
            spine3_aa:  ``(3,)`` spine3 world axis-angle.

        Returns:
            ``(22, 3)`` world positions for all 22 body joints.
        """
        all_pos = self._tpose_22.copy()
        arm_pos = self.fk(arm_aa, spine3_pos, spine3_aa)  # (5, 3)
        # Map arm chain back into the 22-joint array
        for local_i, global_i in enumerate(LEFT_ARM_CHAIN_INDICES):
            if global_i < 22:
                all_pos[global_i] = arm_pos[local_i]
        return all_pos
