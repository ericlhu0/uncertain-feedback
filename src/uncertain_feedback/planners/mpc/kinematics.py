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
SMPL_PARENTS_22 = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]

# All (parent, child) bone pairs for the 22-joint skeleton
SMPL_BONE_PAIRS_22 = [(p, c) for c, p in enumerate(SMPL_PARENTS_22) if p >= 0]

# Left arm joints in the 22-joint skeleton (collar through wrist)
LEFT_ARM_JOINT_INDICES_22 = [13, 16, 18, 20]

# Names of the 3 MPC-controlled joints (shoulder, elbow, wrist)
LEFT_ARM_NAMES = [
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

# Bones that belong to the left arm (including the spine3→collar connection)
LEFT_ARM_BONE_PAIRS_22 = [(9, 13), (13, 16), (16, 18), (18, 20)]

# SMPL joint chain for the left arm FK (spine3 is the anchor)
LEFT_ARM_CHAIN_INDICES = [9, 13, 16, 18, 20]
LEFT_ARM_CHAIN_NAMES = [
    "spine3",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

# Number of joints MPC controls (shoulder, elbow, wrist)
_N_JOINTS = 3


def _compose_rotvec(rotvec: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compose axis-angle rotations element-wise: R_new = R_delta ∘ R_q.

    Args:
        rotvec: ``(..., 3)`` current axis-angle vectors.
        delta:  ``(..., 3)`` delta axis-angle vectors.

    Returns:
        ``(..., 3)`` composed axis-angle vectors.
    """
    flat_q = rotvec.reshape(-1, 3)
    flat_d = delta.reshape(-1, 3)
    composed = (Rotation.from_rotvec(flat_d) * Rotation.from_rotvec(flat_q)).as_rotvec()
    return composed.reshape(rotvec.shape)


class SmplLeftArmFK:
    """Forward kinematics for the SMPL left arm.

    Loads T-pose data from the SMPL neutral PKL file once at construction time.
    Subsequent FK calls are pure numpy/scipy operations.

    The collar rotation is stored as :attr:`collar_aa` (default: zeros for
    T-pose) and is applied automatically by all FK methods.  Set it once after
    loading the initial body pose:

        fk = SmplLeftArmFK()
        fk.collar_aa = fixed_collar_aa  # from decode_pose_with_collar()

    Args:
        smpl_pkl_path: Path to ``SMPL_NEUTRAL.pkl``.  Defaults to the copy
                       inside the MDM submodule.
    """

    def __init__(self, smpl_pkl_path: str | Path | None = None) -> None:
        pkl_path = (
            Path(smpl_pkl_path) if smpl_pkl_path is not None else _SMPL_PKL_DEFAULT
        )
        self._bone_offsets, self._tpose_joints, self._tpose_22 = self._load_from_pkl(
            pkl_path
        )
        self.collar_aa: np.ndarray = np.zeros(3, dtype=np.float64)

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

        j_reg = dd.get("j_regressor") or dd["J_regressor"]
        if hasattr(j_reg, "todense"):
            j_reg = np.array(j_reg.todense())
        else:
            j_reg = np.array(j_reg)

        v = np.array(dd["v_template"])
        joints = j_reg @ v  # (24, 3) T-pose joint positions

        # Arm chain subset
        chain = LEFT_ARM_CHAIN_INDICES
        tpose_chain = joints[chain]  # (5, 3)
        bone_offsets = np.diff(tpose_chain, axis=0)  # (4, 3)

        # Full 22-joint subset (exclude hands at 22, 23)
        tpose_22 = joints[:22].copy()  # (22, 3)

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

        The collar rotation is taken from :attr:`collar_aa`.

        Args:
            arm_aa:     ``(3, 3)`` axis-angle for
                        [left_shoulder, left_elbow, left_wrist].
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

        # Build full 4-joint array: [collar, shoulder, elbow, wrist]
        full_aa = np.concatenate([self.collar_aa[None], arm_aa], axis=0)

        positions = np.empty((5, 3), dtype=np.float64)
        positions[0] = spine3_pos

        t_rot = Rotation.from_rotvec(spine3_aa)
        for i in range(4):
            t_rot = t_rot * Rotation.from_rotvec(full_aa[i])
            positions[i + 1] = positions[i] + t_rot.apply(self._bone_offsets[i])

        return positions

    def fk_batch(
        self,
        arm_aa: np.ndarray,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Batched FK over N arm configurations.

        Args:
            arm_aa:     ``(N, 3, 3)`` axis-angle arrays.
            spine3_pos: ``(3,)`` — same for all samples.
            spine3_aa:  ``(3,)`` — same for all samples.

        Returns:
            ``(N, 5, 3)`` world positions.
        """
        arm_aa = np.asarray(arm_aa, dtype=np.float64)
        n_configs = arm_aa.shape[0]
        out = np.empty((n_configs, 5, 3), dtype=np.float64)
        for i in range(n_configs):
            out[i] = self.fk(arm_aa[i], spine3_pos, spine3_aa)
        return out

    def arm_aa_from_positions(
        self,
        positions: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project SMPL XYZ arm positions into the fixed MPC arm frame.

        MDM positions may include body/spine/collar rotations that the MPC does
        not control.  This helper preserves the MDM arm bone directions, but
        expresses the controlled shoulder/elbow/wrist rotations relative to the
        fixed spine3 and collar rotations used by the MPC.

        The collar rotation is taken from :attr:`collar_aa`.

        Args:
            positions: ``(22, 3)`` full SMPL joint positions or ``(5, 3)``
                arm-chain positions for
                ``[spine3, left_collar, left_shoulder, left_elbow, left_wrist]``.
            spine3_aa: Fixed MPC spine3 world axis-angle.  Defaults to identity.

        Returns:
            ``(3, 3)`` local axis-angles for
            ``[left_shoulder, left_elbow, left_wrist]``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[-2:] == (22, 3):
            arm_positions = positions[LEFT_ARM_CHAIN_INDICES]
        elif positions.shape[-2:] == (5, 3):
            arm_positions = positions
        else:
            raise ValueError(
                "positions must have shape (22, 3) or (5, 3), "
                f"got {positions.shape}"
            )

        spine3_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3, dtype=np.float64)
        )

        parent_world_rot = Rotation.from_rotvec(spine3_aa) * Rotation.from_rotvec(
            self.collar_aa
        )
        controlled = np.zeros((3, 3), dtype=np.float64)

        # Controlled joints are shoulder, elbow, wrist.  Each local rotation
        # maps the corresponding outgoing T-pose bone into the MDM bone
        # direction, then becomes the parent frame for the next joint.
        for out_i, bone_i in enumerate(range(1, 4)):
            actual_bone = arm_positions[bone_i + 1] - arm_positions[bone_i]
            tpose_bone = self._bone_offsets[bone_i]
            if np.linalg.norm(actual_bone) < 1e-8:
                child_world_rot = parent_world_rot
                local_rot = Rotation.identity()
            else:
                child_world_rot, _ = Rotation.align_vectors(
                    [actual_bone],
                    [tpose_bone],
                )
                local_rot = parent_world_rot.inv() * child_world_rot
            controlled[out_i] = local_rot.as_rotvec()
            parent_world_rot = child_world_rot

        return controlled

    def arm_aa_from_positions_batch(
        self,
        positions: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Batched version of :meth:`arm_aa_from_positions`.

        Args:
            positions: ``(..., 22, 3)`` or ``(..., 5, 3)`` positions.
            spine3_aa: Fixed MPC spine3 world axis-angle.

        Returns:
            ``(..., 3, 3)`` arm axis-angles.
        """
        positions = np.asarray(positions, dtype=np.float64)
        leading = positions.shape[:-2]
        flat = positions.reshape((-1, *positions.shape[-2:]))
        out = np.stack(
            [
                self.arm_aa_from_positions(frame, spine3_aa)
                for frame in flat
            ],
            axis=0,
        )
        return out.reshape((*leading, 3, 3))

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
        recomputed via FK.  The collar rotation is taken from :attr:`collar_aa`.

        Args:
            arm_aa:     ``(3, 3)`` axis-angle for [left_shoulder, left_elbow,
                        left_wrist].
            spine3_pos: ``(3,)`` spine3 world position.
            spine3_aa:  ``(3,)`` spine3 world axis-angle.

        Returns:
            ``(22, 3)`` world positions for all 22 body joints.
        """
        all_pos = self._tpose_22.copy()
        arm_pos = self.fk(arm_aa, spine3_pos, spine3_aa)  # (5, 3)
        for local_i, global_i in enumerate(LEFT_ARM_CHAIN_INDICES):
            if global_i < 22:
                all_pos[global_i] = arm_pos[local_i]
        return all_pos
