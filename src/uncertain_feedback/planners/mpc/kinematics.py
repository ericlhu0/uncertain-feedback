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

# Anatomical names of the three controlled rotation slots.  Under this FK
# convention each slot rotates the bone arriving at the named joint.
LEFT_ARM_NAMES = [
    "left_clavicle",
    "left_shoulder",
    "left_elbow",
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

# Number of axis-angle slots consumed by the arm FK.
_N_JOINTS = 3

Q_DIM = 7
Q_CLAVICLE = slice(0, 3)
Q_SHOULDER = slice(3, 6)
Q_ELBOW = 6


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


def _rate_limited_step(
    current_q: np.ndarray,
    target_q: np.ndarray,
    max_delta: float,
) -> tuple[np.ndarray, bool]:
    """Take a geodesic step from ``current_q`` toward ``target_q``, per-joint
    angle-capped.

    Each joint rotates along the shortest SO(3) path toward its target by at
    most ``max_delta`` radians.  Joints already within ``max_delta`` land exactly
    on the target.  Used to follow an MDM trajectory at a bounded angular speed
    (rate limiting) so large frame-to-frame jumps are traversed smoothly.

    Args:
        current_q: ``(3, 3)`` current axis-angle joint angles.
        target_q:  ``(3, 3)`` target axis-angle joint angles.
        max_delta: Maximum per-joint rotation (radians) for this step.

    Returns:
        Tuple of:

        - ``next_q`` ``(3, 3)``: stepped joint angles.
        - ``reached`` ``bool``: ``True`` when every joint was within
          ``max_delta`` of the target (i.e. the target is fully reached).
    """
    current_q = np.asarray(current_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    cur = Rotation.from_rotvec(current_q)
    rel = (Rotation.from_rotvec(target_q) * cur.inv()).as_rotvec()  # (3, 3)
    angles = np.linalg.norm(rel, axis=1)  # (3,)
    scale = np.minimum(1.0, max_delta / np.maximum(angles, 1e-12))
    delta = rel * scale[:, np.newaxis]
    next_q = _compose_rotvec(current_q, delta)
    reached = bool(np.all(angles <= max_delta))
    return next_q, reached


def q_to_arm_aa(q: np.ndarray, hinge_axis: np.ndarray) -> np.ndarray:
    """Convert ``(..., 7)`` planner states to ``(..., 3, 3)`` arm rotations."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1:] != (Q_DIM,):
        raise ValueError(f"q must end in shape ({Q_DIM},), got {q.shape}")
    hinge_axis = np.asarray(hinge_axis, dtype=np.float64)
    return np.stack(
        (
            q[..., Q_CLAVICLE],
            q[..., Q_SHOULDER],
            q[..., Q_ELBOW, np.newaxis] * hinge_axis,
        ),
        axis=-2,
    )


def _compose_q(q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compose 7-DOF arm states with tangent-space action deltas."""
    q = np.asarray(q, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    out = np.empty_like(q)
    out[..., Q_CLAVICLE] = _compose_rotvec(q[..., Q_CLAVICLE], delta[..., Q_CLAVICLE])
    out[..., Q_SHOULDER] = _compose_rotvec(q[..., Q_SHOULDER], delta[..., Q_SHOULDER])
    out[..., Q_ELBOW] = q[..., Q_ELBOW] + delta[..., Q_ELBOW]
    return out


def _rate_limited_step_q(
    current_q: np.ndarray,
    target_q: np.ndarray,
    max_delta: float,
) -> tuple[np.ndarray, bool]:
    """Step toward a 7-DOF target with one angular cap per joint block."""
    current_q = np.asarray(current_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    next_q = np.empty_like(current_q)
    reached = True
    for block in (Q_CLAVICLE, Q_SHOULDER):
        current = Rotation.from_rotvec(current_q[block])
        relative = (Rotation.from_rotvec(target_q[block]) * current.inv()).as_rotvec()
        angle = float(np.linalg.norm(relative))
        scale = min(1.0, max_delta / max(angle, 1e-12))
        next_q[block] = _compose_rotvec(current_q[block], relative * scale)
        reached = reached and angle <= max_delta
    elbow_delta = float(target_q[Q_ELBOW] - current_q[Q_ELBOW])
    next_q[Q_ELBOW] = current_q[Q_ELBOW] + np.clip(elbow_delta, -max_delta, max_delta)
    reached = reached and abs(elbow_delta) <= max_delta
    return next_q, reached


_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _signed_angle_about_axis(
    v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray
) -> float:
    """Signed angle rotating ``v_from`` onto ``v_to`` about ``axis`` (radians).

    ``v_from`` and ``v_to`` are assumed perpendicular to the unit ``axis``.
    """
    s = float(np.dot(np.cross(v_from, v_to), axis))
    c = float(np.dot(v_from, v_to))
    return float(np.arctan2(s, c))


def _canonical_hinge_axis(
    tpose_upper_axis: np.ndarray, tpose_forearm_axis: np.ndarray
) -> np.ndarray:
    """Canonical elbow hinge = T-pose flexion-plane normal (upper × forearm).

    Perpendicular to the upper-arm axis by construction.  Referencing the
    recovered shoulder internal/external rotation to the T-pose flexion plane
    keeps the T-pose a zero-twist neutral (identity body_pose).  The SMPL T-pose
    arm carries a ~7.5° bend, so this cross product is well-defined; only if a
    skeleton's arm were exactly collinear does it fall back to the component of
    world ``+z`` orthogonal to the upper-arm axis.  See
    ``.claude/POSE_REPRESENTATION_AUDIT.md``.
    """
    axis = np.cross(tpose_upper_axis, tpose_forearm_axis)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        axis = np.cross(tpose_upper_axis, _WORLD_UP)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.cross(tpose_upper_axis, np.array([0.0, 1.0, 0.0]))
            norm = np.linalg.norm(axis)
    return axis / norm


def anatomical_elbow_wrist_slots(
    shoulder_pos: np.ndarray,
    elbow_pos: np.ndarray,
    wrist_pos: np.ndarray,
    parent_world_rot: Rotation,
    tpose_upper_axis: np.ndarray,
    tpose_forearm_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Anatomically-constrained elbow + wrist slot rotations for the left arm.

    Given the observed shoulder/elbow/wrist world positions, reallocate the two
    axial DOFs so the reconstruction is anatomically faithful while preserving
    positions exactly (both bone directions are reproduced by construction):

    1. **Elbow slot** — aligns the T-pose upper-arm axis to the observed
       upper-arm direction ``u``, then twists about ``u`` so the canonical hinge
       axis lands on the observed flexion-plane normal ``n = normalize(u × f)``.
       That twist *is* the recovered shoulder internal/external rotation.
       Degenerate when the arm is straight (``‖u × f‖ ≈ 0``); falls back to the
       minimal (zero-twist) alignment.
    2. **Wrist slot** — the elbow-relative shortest-arc rotation onto the forearm
       direction ``f``: a pure hinge with zero pronation, referenced to the
       (stable) elbow frame rather than the near-antiparallel T-pose forearm.

    Slot semantics follow the repo FK convention (a joint's rotation transforms
    the bone arriving at it): the returned ``elbow_slot`` orients the upper arm
    (shoulder→elbow) and the ``wrist_slot`` orients the forearm (elbow→wrist).
    ``parent_world_rot`` is the world rotation of the slot upstream of the elbow
    slot (the shoulder-slot world rotation).

    Args:
        shoulder_pos:       ``(3,)`` world position of the shoulder.
        elbow_pos:          ``(3,)`` world position of the elbow.
        wrist_pos:          ``(3,)`` world position of the wrist.
        parent_world_rot:   Shoulder-slot world rotation (elbow slot's parent).
        tpose_upper_axis:   ``(3,)`` T-pose shoulder→elbow bone (any length).
        tpose_forearm_axis: ``(3,)`` T-pose elbow→wrist bone (any length).

    Returns:
        Tuple ``(elbow_slot_aa, wrist_slot_aa)`` of local axis-angle vectors.
    """
    tpose_upper_axis = np.asarray(tpose_upper_axis, dtype=np.float64)
    tpose_upper_axis = tpose_upper_axis / np.linalg.norm(tpose_upper_axis)
    tpose_forearm_axis = np.asarray(tpose_forearm_axis, dtype=np.float64)
    tpose_forearm_axis = tpose_forearm_axis / np.linalg.norm(tpose_forearm_axis)

    u = np.asarray(elbow_pos, dtype=np.float64) - np.asarray(
        shoulder_pos, dtype=np.float64
    )
    f = np.asarray(wrist_pos, dtype=np.float64) - np.asarray(
        elbow_pos, dtype=np.float64
    )
    u = u / np.linalg.norm(u)
    f = f / np.linalg.norm(f)

    r_up, _ = Rotation.align_vectors([u], [tpose_upper_axis])

    normal = np.cross(u, f)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-8:
        elbow_world = r_up
    else:
        normal = normal / normal_len
        hinge_now = r_up.apply(
            _canonical_hinge_axis(tpose_upper_axis, tpose_forearm_axis)
        )
        theta = _signed_angle_about_axis(hinge_now, normal, u)
        elbow_world = Rotation.from_rotvec(theta * u) * r_up

    rest_forearm = elbow_world.apply(tpose_forearm_axis)
    wrist_delta, _ = Rotation.align_vectors([f], [rest_forearm])
    wrist_world = wrist_delta * elbow_world

    elbow_local = parent_world_rot.inv() * elbow_world
    wrist_local = elbow_world.inv() * wrist_world
    return elbow_local.as_rotvec(), wrist_local.as_rotvec()


class SmplLeftArmFK:
    """Forward kinematics for the SMPL left arm.

    Loads T-pose data from the SMPL neutral PKL file once at construction time.
    Subsequent FK calls are pure numpy/scipy operations.

    The collar rotation is stored as :attr:`collar_aa` (default: zeros for
    T-pose) and is applied automatically by all FK methods.  Set it once after
    loading the initial body pose:

        fk = SmplLeftArmFK()
        fk.collar_aa = fixed_collar_aa  # from decode_pose()

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
        self._hinge_axis = _canonical_hinge_axis(
            self._bone_offsets[2], self._bone_offsets[3]
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

    @property
    def elbow_hinge_axis(self) -> np.ndarray:
        """Canonical elbow-flexion axis in the shoulder-local frame."""
        return self._hinge_axis.copy()

    def scale_arm_lengths(
        self, clavicle: float, upper_arm: float, forearm: float
    ) -> None:
        """Rescale the arm bones to measured segment lengths (metres).

        Bone directions are unchanged (so the elbow hinge axis and all joint
        angles keep their meaning); only the lengths move, and the T-pose joint
        positions are recomputed to stay consistent. Idempotent — lengths are
        absolute, not relative scales. Used by envs that measure the person, so
        the whole run plans on their proportions instead of SMPL neutral's.
        """
        for i, length in zip((1, 2, 3), (clavicle, upper_arm, forearm)):
            self._bone_offsets[i] *= length / float(
                np.linalg.norm(self._bone_offsets[i])
            )
        self._tpose_joints[1:] = self._tpose_joints[0] + np.cumsum(
            self._bone_offsets, axis=0
        )
        for local_i, global_i in enumerate(LEFT_ARM_CHAIN_INDICES):
            self._tpose_22[global_i] = self._tpose_joints[local_i]

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

        positions = np.empty((5, 3), dtype=np.float64)
        positions[0] = spine3_pos

        rotations = self.bone_world_rotations(arm_aa, spine3_aa)
        for i in range(4):
            positions[i + 1] = positions[i] + rotations[i].apply(self._bone_offsets[i])

        return positions

    def bone_world_rotations(
        self,
        arm_aa: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> list[Rotation]:
        """World rotations transforming each arm-chain bone, in FK convention.

        Entry *i* is the accumulated rotation applied to ``bone_offsets[i]``, so
        the four entries carry the [spine3→collar, clavicle, upper-arm, forearm]
        bones. Unlike a frame built from joint positions, these have no
        up-reference to flip and carry the bone's roll about its own axis.

        Args:
            arm_aa:    ``(3, 3)`` axis-angle for [shoulder, elbow, wrist].
            spine3_aa: ``(3,)`` world axis-angle of spine3 (default identity).
        """
        arm_aa = np.asarray(arm_aa, dtype=np.float64)
        spine3_aa = (
            np.asarray(spine3_aa, dtype=np.float64)
            if spine3_aa is not None
            else np.zeros(3)
        )
        # Build full 4-joint array: [collar, shoulder, elbow, wrist]
        full_aa = np.concatenate([self.collar_aa[None], arm_aa], axis=0)

        rotations: list[Rotation] = []
        t_rot = Rotation.from_rotvec(spine3_aa)
        for i in range(4):
            t_rot = t_rot * Rotation.from_rotvec(full_aa[i])
            rotations.append(t_rot)
        return rotations

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
                "positions must have shape (22, 3) or (5, 3), " f"got {positions.shape}"
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

        # Shoulder slot: minimum rotation mapping the T-pose clavicle bone onto
        # the actual clavicle bone (collar→shoulder), then expressed relative to
        # the fixed parent frame.
        actual_clavicle = arm_positions[2] - arm_positions[1]
        if np.linalg.norm(actual_clavicle) < 1e-8:
            shoulder_world = parent_world_rot
        else:
            shoulder_world, _ = Rotation.align_vectors(
                [actual_clavicle],
                [self._bone_offsets[1]],
            )
            controlled[0] = (parent_world_rot.inv() * shoulder_world).as_rotvec()

        # Elbow + wrist slots: anatomical reparameterization — the elbow slot
        # carries the recovered shoulder rotation (upper-arm orientation) and the
        # wrist slot is a pure forearm hinge.  Positions are preserved.
        controlled[1], controlled[2] = anatomical_elbow_wrist_slots(
            arm_positions[2],
            arm_positions[3],
            arm_positions[4],
            shoulder_world,
            self._bone_offsets[2],
            self._bone_offsets[3],
        )

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
            [self.arm_aa_from_positions(frame, spine3_aa) for frame in flat],
            axis=0,
        )
        return out.reshape((*leading, 3, 3))

    def arm_aa_to_q(
        self,
        arm_aa: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert one arm pose to the planner's 7-DOF representation.

        Hinge-constrained input is extracted directly so planner states round-trip
        exactly.  Raw input with an off-hinge elbow rotation is first decoded from
        its FK positions, preserving every arm joint position while reallocating
        axial rotation to the anatomical shoulder slot.
        """
        arm_aa = np.asarray(arm_aa, dtype=np.float64)
        if arm_aa.shape != (_N_JOINTS, 3):
            raise ValueError(f"arm_aa must have shape (3, 3), got {arm_aa.shape}")
        elbow_angle = float(np.dot(arm_aa[2], self._hinge_axis))
        off_hinge = arm_aa[2] - elbow_angle * self._hinge_axis
        if np.linalg.norm(off_hinge) > 1e-10:
            arm_aa = self.arm_aa_from_positions(
                self.fk(arm_aa, spine3_aa=spine3_aa), spine3_aa
            )
            elbow_angle = float(np.dot(arm_aa[2], self._hinge_axis))
        return np.concatenate((arm_aa[0], arm_aa[1], [elbow_angle]))

    def arm_aa_to_q_batch(
        self,
        arm_aa: np.ndarray,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized-leading-dimension version of :meth:`arm_aa_to_q`."""
        arm_aa = np.asarray(arm_aa, dtype=np.float64)
        if arm_aa.shape[-2:] != (_N_JOINTS, 3):
            raise ValueError(f"arm_aa must end in shape (3, 3), got {arm_aa.shape}")
        leading = arm_aa.shape[:-2]
        flat = arm_aa.reshape((-1, _N_JOINTS, 3))
        out = np.stack([self.arm_aa_to_q(frame, spine3_aa) for frame in flat], axis=0)
        return out.reshape((*leading, Q_DIM))

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


def _shortest_arc_rotvecs(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimal rotations taking unit vectors ``v_from`` onto ``v_to``, batched.

    Both inputs are ``(..., 3)`` and broadcast against each other. Antiparallel
    pairs (axis undefined) rotate by pi about an arbitrary perpendicular.
    """
    v_from, v_to = np.broadcast_arrays(v_from, v_to)
    axis = np.cross(v_from, v_to)
    sin = np.linalg.norm(axis, axis=-1)
    cos = np.sum(v_from * v_to, axis=-1)
    angle = np.arctan2(sin, cos)
    direction = axis / np.maximum(sin, 1e-12)[..., None]
    anti = (sin < 1e-12) & (cos < 0.0)
    if np.any(anti):
        perp = np.cross(v_from, _WORLD_UP)
        fallback = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        weak = np.linalg.norm(perp, axis=-1, keepdims=True) < 1e-8
        perp = np.where(weak, fallback, perp)
        perp = perp / np.linalg.norm(perp, axis=-1, keepdims=True)
        direction = np.where(anti[..., None], perp, direction)
    return direction * angle[..., None]


def project_forearm_frames(
    fk: SmplLeftArmFK,
    ee_pos: np.ndarray,
    forearm_rot: np.ndarray,
    grasp_offset: np.ndarray,
    q_ref: np.ndarray,
    spine3_pos: np.ndarray | None = None,
    spine3_aa: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project grasp-implied forearm frames onto the arm manifold, batched.

    A rigid grasp maps a robot end-effector pose to a forearm frame, but an
    arbitrary ee pose is generally not one the attached arm can produce: with
    the clavicle frozen the shoulder is fixed, the elbow must sit at upper-arm
    length from it, and the forearm cannot roll about its own axis. The
    projection is anchored on the gripper *position* — the component of the
    grasp a robot holding the forearm couples firmly, while orientation slips —
    so it predicts what the coupled arm actually does: the elbow goes to the
    nearest point that keeps both the upper-arm length and the rigid
    elbow-to-gripper distance (the intersection circle of the two spheres),
    the forearm rotation is minimally re-anchored so the gripper stays at the
    commanded ee position, and forearm roll is dropped. The residual is how far
    the implied frame was from the manifold — the motion the grasp cannot
    transmit.

    The slot reconstruction mirrors :func:`anatomical_elbow_wrist_slots`
    (recovered shoulder rotation on the elbow slot, pure-hinge wrist slot),
    vectorized over the batch; the clavicle slot is ``q_ref``'s throughout.

    Args:
        ee_pos:       ``(..., 3)`` commanded gripper positions, SMPL frame.
        forearm_rot:  ``(..., 3, 3)`` implied forearm bone rotations
                      (ee rotation composed with the inverse grasp rotation),
                      SMPL frame.
        grasp_offset: ``(3,)`` gripper origin in the forearm frame
                      (:attr:`MeasuredGrasp.position` — forearm-local, so the
                      same coordinates in SMPL and pybullet usage).
        q_ref:        ``(7,)`` current configuration (clavicle source).
        spine3_pos:   ``(3,)`` torso anchor shared with the planner.
        spine3_aa:    ``(3,)`` torso rotation shared with the planner.

    Returns:
        Tuple of ``(..., 3, 3)`` arm axis-angles, ``(..., 3)`` projected wrist
        positions, and ``(...,)`` residuals (elbow projection metres +
        untransmitted rotation radians).
    """
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    batch = ee_pos.shape[:-1]
    flat_ee = ee_pos.reshape(-1, 3)
    flat_rot = Rotation.from_matrix(
        np.asarray(forearm_rot, dtype=np.float64).reshape(-1, 3, 3)
    )
    grasp_offset = np.asarray(grasp_offset, dtype=np.float64)
    grasp_dist = float(np.linalg.norm(grasp_offset))
    q_ref = np.asarray(q_ref, dtype=np.float64)

    aa_ref = q_to_arm_aa(q_ref, fk.elbow_hinge_axis)
    shoulder_world = fk.bone_world_rotations(aa_ref, spine3_aa)[1]
    shoulder_pos = fk.fk(aa_ref, spine3_pos, spine3_aa)[2]
    upper_offset, forearm_offset = fk._bone_offsets[2], fk._bone_offsets[3]
    upper_len = float(np.linalg.norm(upper_offset))
    forearm_len = float(np.linalg.norm(forearm_offset))
    tpose_upper = upper_offset / upper_len
    tpose_forearm = forearm_offset / forearm_len
    hinge = fk.elbow_hinge_axis

    flat_elbow = flat_ee - flat_rot.apply(grasp_offset)
    # Elbow: nearest point on the two-sphere intersection circle
    # (|elbow − shoulder| = upper_len and |elbow − ee| = grasp_dist) to the
    # implied elbow; clamped along the shoulder→ee axis when the spheres do
    # not meet (gripper pulled beyond reach or inside it).
    to_ee = flat_ee - shoulder_pos
    dist_ee = np.maximum(np.linalg.norm(to_ee, axis=-1), 1e-9)
    axis_hat = to_ee / dist_ee[..., None]
    along = np.clip(
        (upper_len**2 - grasp_dist**2 + dist_ee**2) / (2.0 * dist_ee),
        -upper_len,
        upper_len,
    )
    radius = np.sqrt(np.maximum(upper_len**2 - along**2, 0.0))
    center = shoulder_pos + along[..., None] * axis_hat
    offset = flat_elbow - center
    perp = offset - np.sum(offset * axis_hat, axis=-1, keepdims=True) * axis_hat
    perp_len = np.linalg.norm(perp, axis=-1, keepdims=True)
    fallback = np.cross(axis_hat, _WORLD_UP)
    weak = np.linalg.norm(fallback, axis=-1, keepdims=True) < 1e-8
    fallback = np.where(weak, np.cross(axis_hat, np.array([0.0, 1.0, 0.0])), fallback)
    fallback = fallback / np.linalg.norm(fallback, axis=-1, keepdims=True)
    perp_dir = np.where(perp_len > 1e-9, perp / np.maximum(perp_len, 1e-9), fallback)
    elbow_proj = center + radius[..., None] * perp_dir

    # Minimal rotation keeping the gripper on the commanded ee position.
    grip_implied = flat_ee - flat_elbow
    grip_proj = flat_ee - elbow_proj
    if grasp_dist > 1e-9:
        anchor = Rotation.from_rotvec(
            _shortest_arc_rotvecs(
                grip_implied
                / np.maximum(np.linalg.norm(grip_implied, axis=-1, keepdims=True), 1e-9),
                grip_proj
                / np.maximum(np.linalg.norm(grip_proj, axis=-1, keepdims=True), 1e-9),
            )
        )
        flat_rot = anchor * flat_rot

    wrist_proj = elbow_proj + flat_rot.apply(forearm_offset)
    u = (elbow_proj - shoulder_pos) / upper_len
    f = (wrist_proj - elbow_proj) / forearm_len

    r_up = Rotation.from_rotvec(_shortest_arc_rotvecs(tpose_upper, u))
    normal = np.cross(u, f)
    normal_len = np.linalg.norm(normal, axis=-1)
    normal = normal / np.maximum(normal_len, 1e-12)[..., None]
    hinge_now = r_up.apply(hinge)
    theta = np.arctan2(
        np.sum(np.cross(hinge_now, normal) * u, axis=-1),
        np.sum(hinge_now * normal, axis=-1),
    )
    theta = np.where(normal_len < 1e-8, 0.0, theta)
    elbow_world = Rotation.from_rotvec(theta[..., None] * u) * r_up
    rest_forearm = elbow_world.apply(tpose_forearm)
    wrist_world = Rotation.from_rotvec(_shortest_arc_rotvecs(rest_forearm, f)) * (
        elbow_world
    )

    elbow_slot = (shoulder_world.inv() * elbow_world).as_rotvec()
    wrist_slot = (elbow_world.inv() * wrist_world).as_rotvec()
    arm_aa = np.stack(
        (np.broadcast_to(q_ref[Q_CLAVICLE], elbow_slot.shape), elbow_slot, wrist_slot),
        axis=-2,
    )

    # After the anchor rotation the implied and reconstructed rotations agree
    # on the forearm direction, so their relative rotation is the roll a
    # parallel-jaw grasp cannot transmit; the anchor swing itself shows up in
    # the elbow displacement term.
    untransmitted = np.linalg.norm((flat_rot.inv() * wrist_world).as_rotvec(), axis=-1)
    residual = np.linalg.norm(flat_elbow - elbow_proj, axis=-1) + untransmitted
    return (
        arm_aa.reshape(*batch, 3, 3),
        wrist_proj.reshape(*batch, 3),
        residual.reshape(batch),
    )


def q_reaching_wrist(
    fk: SmplLeftArmFK,
    wrist_target: np.ndarray,
    q_seed: np.ndarray,
    spine3_pos: np.ndarray | None = None,
    spine3_aa: np.ndarray | None = None,
    posture_weight: float = 0.05,
) -> np.ndarray:
    """``(7,)`` arm configuration whose wrist sits at ``wrist_target``.

    Inverse of the Cartesian goal the MPC tracks, for showing a goal *pose*
    where the goal itself is only a point: a wrist position leaves the arm's
    posture underdetermined (7 DOF for 3 constraints), so the fit is pulled
    toward ``q_seed`` by ``posture_weight`` and returns the nearest reaching
    configuration rather than an arbitrary branch. An out-of-reach target
    yields the closest the arm can get.

    Args:
        wrist_target:   ``(3,)`` world wrist position (*not* spine3-relative).
        q_seed:         ``(7,)`` configuration to stay near — normally the
                        arm's current one.
        posture_weight: Pull toward ``q_seed``, in radians per metre of wrist
                        error.
    """
    from scipy.optimize import least_squares  # pylint: disable=import-outside-toplevel

    wrist_target = np.asarray(wrist_target, dtype=np.float64)
    q_seed = np.asarray(q_seed, dtype=np.float64)

    def residual(q: np.ndarray) -> np.ndarray:
        wrist = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), spine3_pos, spine3_aa)[4]
        return np.concatenate([wrist - wrist_target, posture_weight * (q - q_seed)])

    return np.asarray(least_squares(residual, q_seed).x, dtype=np.float64)
