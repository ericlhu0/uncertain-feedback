"""The person the robot moves: their body and what their arm has done."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from uncertain_feedback.planners.mpc.arm_features import (
    arm_feature_series,
    arm_q_from_features,
)
from uncertain_feedback.planners.mpc.kinematics import (
    Q_CLAVICLE,
    Q_DIM,
    WRIST_CHAIN_IDX,
    SmplLeftArmFK,
    q_reaching_wrist,
    q_to_arm_aa,
)

_FIELDS = ("_fk", "_spine3_pos", "_spine3_aa", "_posture", "_hml_pose", "_history")


def _frozen(array: Any, dtype: Any = np.float64) -> np.ndarray:
    """A copy backed by immutable bytes, so no view of it can be made writeable."""
    array = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _assign(
    human: Human,
    fk: SmplLeftArmFK,
    spine3_pos: np.ndarray,
    spine3_aa: np.ndarray,
    posture: np.ndarray,
    hml_pose: np.ndarray | None,
    history: np.ndarray,
) -> None:
    values = (
        fk,
        _frozen(spine3_pos),
        _frozen(spine3_aa),
        _frozen(posture),
        None if hml_pose is None else _frozen(hml_pose, hml_pose.dtype),
        _frozen(np.reshape(history, (-1, Q_DIM))),
    )
    for name, value in zip(_FIELDS, values):
        object.__setattr__(human, name, value)


def _rebuild(*fields: Any) -> Human:
    human = object.__new__(Human)
    _assign(human, *fields)
    return human


class Human:
    """The person the robot moves: their body and what their arm has done.

    Immutable: :meth:`step`, :meth:`rewind` and :meth:`reset_human_with_q`
    return new Humans. Arrays are stored as immutable bytes and every property
    returns a fresh copy, so a read is a snapshot. ``history`` holds the
    executed arm states and ``q`` is the current one. The geometry methods take
    an explicit ``q`` with any leading shape and return arrays.

    ``posture`` and ``spine3_pos`` are in the frame the pose file decodes to:
    the SMPL-neutral skeleton around the T-pose pelvis. Compare positions only
    with ones this Human computed, not with raw MDM output.

    Args:
        pose: Saved HML263 ``.pt`` pose file; ``None`` for the T-pose body.
        arm:  ``(3, 3)`` start-arm axis-angles replacing the pose file's arm.
    """

    __slots__ = _FIELDS
    _fk: SmplLeftArmFK
    _spine3_pos: np.ndarray
    _spine3_aa: np.ndarray
    _posture: np.ndarray
    _hml_pose: np.ndarray | None
    _history: np.ndarray

    def __init__(
        self, pose: Path | None = None, arm: npt.ArrayLike | None = None
    ) -> None:
        hml_pose: np.ndarray | None = None
        if pose is None:
            fk = SmplLeftArmFK()
            arm_aa, posture, spine3_aa = (
                np.zeros((3, 3)),
                fk.tpose_all_joints,
                np.zeros(3),
            )
        else:
            # pylint: disable-next=import-outside-toplevel
            from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
                decode_hml_pose,
                load_hml_pose,
            )

            hml_pose = load_hml_pose(pose)
            arm_aa, posture, spine3_aa, collar_aa = decode_hml_pose(hml_pose)
            fk = SmplLeftArmFK(collar_aa=collar_aa)
        if arm is not None:
            arm_aa = np.asarray(arm, dtype=np.float64)
            if arm_aa.shape != (3, 3):
                raise ValueError(f"arm must have shape (3, 3), got {arm_aa.shape}")
        q0 = fk.arm_aa_to_q(arm_aa, spine3_aa)
        _assign(self, fk, posture[9], spine3_aa, posture, hml_pose, q0)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Human is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Human is immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, tuple(getattr(self, name) for name in _FIELDS))

    def _with_history(self, history: np.ndarray) -> Human:
        return _rebuild(
            self._fk,
            self._spine3_pos,
            self._spine3_aa,
            self._posture,
            self._hml_pose,
            history,
        )

    @property
    def fk(self) -> SmplLeftArmFK:
        """This person's fitted arm kinematics (read by the visualizers)."""
        return self._fk

    @property
    def spine3_pos(self) -> np.ndarray:
        """``(3,)`` spine3 anchor; Cartesian goals are measured from it."""
        return self._spine3_pos.copy()

    @property
    def spine3_aa(self) -> np.ndarray:
        """``(3,)`` spine3 world axis-angle."""
        return self._spine3_aa.copy()

    @property
    def posture(self) -> np.ndarray:
        """``(22, 3)`` decoded body joints (the arm entries are the pose's own arm)."""
        return self._posture.copy()

    @property
    def hml_pose(self) -> np.ndarray | None:
        """``(263,)`` normalized HML263 pose the body was decoded from, if any."""
        return None if self._hml_pose is None else self._hml_pose.copy()

    @property
    def history(self) -> np.ndarray:
        """``(T, 7)`` executed arm states, oldest first."""
        return self._history.copy()

    @property
    def q(self) -> np.ndarray:
        """``(7,)`` current arm state."""
        return self._history[-1].copy()

    def step(self, frames: np.ndarray) -> Human:
        """This person after also executing ``frames`` (``(7,)`` or ``(N, 7)``)."""
        frames = np.reshape(np.asarray(frames, dtype=np.float64), (-1, Q_DIM))
        return self._with_history(np.concatenate([self._history, frames]))

    def rewind(self, frame: int) -> Human:
        """This person as of ``frame``: history cut back to ``[0, frame]``."""
        if not 0 <= frame < len(self._history):
            raise IndexError(f"frame {frame} outside history of {len(self._history)}")
        return self._with_history(self._history[: frame + 1])

    def reset_human_with_q(self, q: np.ndarray) -> Human:
        """The same body with a fresh history that starts at ``q``."""
        return self._with_history(np.asarray(q, dtype=np.float64))

    def measured(
        self,
        fk: SmplLeftArmFK,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
        posture: np.ndarray,
        q: np.ndarray,
    ) -> Human:
        """This person as an env measured them: its kinematics, anchor, body and arm.

        The pose file's ``hml_pose`` carries over; the history starts at ``q``.
        """
        return _rebuild(fk, spine3_pos, spine3_aa, posture, self._hml_pose, q)

    def fk_positions_from_q(self, q: np.ndarray) -> np.ndarray:
        """``(..., 7)`` arm states to ``(..., 5, 3)`` world arm-chain positions."""
        arm_aa = self.arm_aa_from_q(q)
        flat = arm_aa.reshape((-1, 3, 3))
        positions = self._fk.fk_batch(flat, self.spine3_pos, self.spine3_aa)
        return positions.reshape((*arm_aa.shape[:-2], 5, 3))

    def ik_q_from_positions(self, positions: np.ndarray) -> np.ndarray:
        """``(..., 5, 3)`` arm-chain positions to ``(..., 7)`` arm states.

        Only bone directions are used, so positions from a skeleton with other
        proportions (MDM's) still map onto this person's arm.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[-2:] != (5, 3):
            raise ValueError(f"positions must end in (5, 3), got {positions.shape}")
        arm_aa = self._fk.arm_aa_from_positions_batch(positions, self.spine3_aa)
        return self.q_from_arm_aa(arm_aa)

    def wrist_from_q(self, q: np.ndarray) -> np.ndarray:
        """``(..., 7)`` arm states to ``(..., 3)`` spine3-relative wrist positions."""
        return self.fk_positions_from_q(q)[..., WRIST_CHAIN_IDX, :] - self.spine3_pos

    def q_from_wrist(self, target: np.ndarray) -> np.ndarray:
        """``(7,)`` arm state nearest ``q`` whose spine3-relative wrist is ``target``."""
        return q_reaching_wrist(
            self._fk,
            np.asarray(target, dtype=np.float64) + self.spine3_pos,
            self.q,
            self.spine3_pos,
            self.spine3_aa,
        )

    def features_from_q(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Anatomical joint features (radians) of ``(..., 7)`` arm states."""
        return arm_feature_series(q, self)

    def q_from_features(self, features: np.ndarray) -> np.ndarray:
        """``(T, 5)`` features to ``(T, 7)`` arm states on the current clavicle."""
        return arm_q_from_features(features, self.q[Q_CLAVICLE], self)

    def arm_aa_from_q(self, q: np.ndarray) -> np.ndarray:
        """``(..., 7)`` arm states to ``(..., 3, 3)`` axis-angles."""
        return q_to_arm_aa(q, self._fk.elbow_hinge_axis)

    def q_from_arm_aa(self, arm_aa: np.ndarray) -> np.ndarray:
        """``(..., 3, 3)`` axis-angles to ``(..., 7)`` arm states."""
        return self._fk.arm_aa_to_q_batch(arm_aa, self.spine3_aa)
