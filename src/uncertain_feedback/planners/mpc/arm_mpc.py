"""Sampling-based MPC for the SMPL left arm in joint angle space.

State and target are expressed as axis-angle vectors, matching SMPL's
native ``body_pose`` format (``smplx.SMPL``)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Joint index constants
# ---------------------------------------------------------------------------

LEFT_ARM_JOINT_NAMES = ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]

# Indices into the 22-joint HumanML3D skeleton
LEFT_ARM_HML_INDICES = [13, 16, 18, 20]

# Indices into SMPL's body_pose (23 joints, 0-indexed; joint 0 = pelvis = global_orient)
LEFT_ARM_SMPL_INDICES = [12, 15, 17, 19]

_N_JOINTS = 4  # number of controlled joints


# ---------------------------------------------------------------------------
# Helper: batched SO(3) composition
# ---------------------------------------------------------------------------


def _compose_rotvec(q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compose axis-angle rotations element-wise: R_new = R_delta ∘ R_q.

    Args:
        q:     ``(..., 3)`` current axis-angle vectors.
        delta: ``(..., 3)`` delta axis-angle vectors.

    Returns:
        ``(..., 3)`` composed axis-angle vectors.
    """
    flat_q = q.reshape(-1, 3)
    flat_d = delta.reshape(-1, 3)
    composed = (Rotation.from_rotvec(flat_d) * Rotation.from_rotvec(flat_q)).as_rotvec()
    return composed.reshape(q.shape)


# ---------------------------------------------------------------------------
# MPC controller
# ---------------------------------------------------------------------------


class SmplLeftArmMPC:
    """Sampling-based MPC for the SMPL left arm.

    Draws ``n_samples`` random action sequences, rolls each out, and returns
    the one with the lowest terminal cost.

    Args:
        horizon:         Number of look-ahead steps.
        n_samples:       Number of candidate action sequences sampled per
                         ``solve`` call.
        max_angle_delta: Standard deviation of the sampling distribution
                         (radians).
    """

    def __init__(
        self,
        horizon: int = 10,
        n_samples: int = 512,
        max_angle_delta: float = 0.001,
    ) -> None:
        self.horizon = horizon
        self.n_samples = n_samples
        self.max_angle_delta = max_angle_delta

        # Warm-start: previous best plan shifted forward by one step
        self._prev_best: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollout(self, current_q: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Roll out N trajectories from ``current_q`` using action sequences
        ``U``.

        Args:
            current_q: ``(4, 3)`` current joint angles.
            U:         ``(N, H, 4, 3)`` sampled action sequences.

        Returns:
            ``(N, H+1, 4, 3)`` state trajectories (includes initial state).
        """
        N, H = U.shape[0], U.shape[1]
        Q = np.empty((N, H + 1, _N_JOINTS, 3), dtype=np.float64)
        Q[:, 0] = current_q[np.newaxis]

        for t in range(H):
            Q[:, t + 1] = _compose_rotvec(Q[:, t], U[:, t])

        return Q

    def _cost(self, Q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        """Compute terminal cost for each of the N sampled trajectories.

        Args:
            Q:        ``(N, H+1, 4, 3)`` state trajectories.
            target_q: ``(4, 3)``         target joint configuration.

        Returns:
            ``(N,)`` cost per trajectory.
        """
        return ((Q[:, -1] - target_q[np.newaxis]) ** 2).sum(axis=(-2, -1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample action sequences and return the best one.

        Args:
            current_q: ``(4, 3)`` current axis-angle joint angles for
                       [left_collar, left_shoulder, left_elbow, left_wrist].
            target_q:  ``(4, 3)`` desired axis-angle joint configuration.

        Returns:
            Tuple of:

            - ``first_action`` ``(4, 3)``: best delta to apply at the current step.
            - ``plan`` ``(H, 4, 3)``: full best action sequence.
        """
        current_q = np.asarray(current_q, dtype=np.float64)
        target_q = np.asarray(target_q, dtype=np.float64)

        # Warm-start: shift previous best plan by one step; fill last with zeros
        if self._prev_best is not None:
            mean = np.concatenate(
                [self._prev_best[1:], np.zeros((1, _N_JOINTS, 3))], axis=0
            )
        else:
            mean = np.zeros((self.horizon, _N_JOINTS, 3), dtype=np.float64)

        U = np.random.normal(
            loc=mean,
            scale=self.max_angle_delta,
            size=(self.n_samples, self.horizon, _N_JOINTS, 3),
        )

        Q = self._rollout(current_q, U)
        costs = self._cost(Q, target_q)

        best_idx = np.argmin(costs)
        best_plan = U[best_idx]

        self._prev_best = best_plan
        return best_plan[0], best_plan

    def step(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
    ) -> np.ndarray:
        """Perform one MPC step.

        Samples action sequences, applies the best first action to ``current_q``
        via SO(3) composition, and returns the updated joint angles.

        Args:
            current_q: ``(4, 3)`` current axis-angle joint angles.
            target_q:  ``(4, 3)`` target axis-angle joint configuration.

        Returns:
            ``(4, 3)`` updated axis-angle joint angles.
        """
        first_action, _ = self.solve(current_q, target_q)
        return _compose_rotvec(np.asarray(current_q, dtype=np.float64), first_action)
