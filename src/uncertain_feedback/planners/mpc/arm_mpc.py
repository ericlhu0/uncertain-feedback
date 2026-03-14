# pylint: disable=duplicate-code
"""Sampling-based MPC for the SMPL left arm in joint angle space.

State and target are expressed as axis-angle vectors, matching SMPL's
native ``body_pose`` format (``smplx.SMPL``)
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.visualizer import ArmVisualizer

# ---------------------------------------------------------------------------
# Joint index constants
# ---------------------------------------------------------------------------

_N_JOINTS = 4  # number of controlled joints


@dataclass
class _MpcConfig:
    horizon: int
    n_samples: int
    max_angle_delta: float


@dataclass
class _VisConfig:
    fk: SmplLeftArmFK
    spine_pos: np.ndarray | None
    spine_aa: np.ndarray | None
    body_pos: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helper: batched SO(3) composition
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MPC controller
# ---------------------------------------------------------------------------


class SmplLeftArmMPC:
    """Sampling-based MPC for the SMPL left arm.

    Draws ``n_samples`` random action sequences, rolls each out, and returns
    the one with the lowest terminal cost.

    The controller maintains a queue of goals.  It always targets the first
    goal in the queue.  Once the controller reaches that goal (within
    ``goal_threshold``), the goal is popped and the next one becomes active.
    Goals can be added at any time via :meth:`append_goal` or
    :meth:`prepend_goal`.

    Args:
        horizon:         Number of look-ahead steps.
        n_samples:       Number of candidate action sequences sampled per
                         ``solve`` call.
        max_angle_delta: Standard deviation of the sampling distribution
                         (radians).
        goals:           Initial list of ``(4, 3)`` target joint configurations.
        goal_threshold:  L2 distance (in rot-vec space) below which the
                         current goal is considered reached (default: 0.01).
        visualize:       If ``True``, open a live matplotlib window and update
                         it each time :meth:`step` is called.  Requires
                         ``fk`` to also be provided.
        fk:              :class:`SmplLeftArmFK` instance (required when
                         ``visualize=True``).
        spine3_pos:      ``(3,)`` world position of spine3 (optional).
        spine3_aa:       ``(3,)`` world axis-angle of spine3 (optional).
    """

    def __init__(
        self,
        horizon: int = 10,
        n_samples: int = 512,
        max_angle_delta: float = 0.001,
        goals: list[np.ndarray] | None = None,
        goal_threshold: float = 0.01,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
    ) -> None:
        self._config = _MpcConfig(horizon, n_samples, max_angle_delta)
        self.visualize = visualize

        self._goals: deque[np.ndarray] = deque(
            [np.asarray(g, dtype=np.float64) for g in goals] if goals else []
        )
        self._goal_threshold = goal_threshold

        if visualize:
            if fk is None:
                raise ValueError("visualize=True requires `fk` to be provided.")
            self._vis_config: _VisConfig | None = _VisConfig(fk, spine3_pos, spine3_aa)
        else:
            self._vis_config = None

        # Warm-start: previous best plan shifted forward by one step
        self._prev_best: np.ndarray | None = None

        # Live visualizer (lazily initialised on first step)
        self._vis: ArmVisualizer | None = None

    # ------------------------------------------------------------------
    # Goal queue management
    # ------------------------------------------------------------------

    @property
    def current_goal(self) -> np.ndarray | None:
        """The active goal (front of the queue), or ``None`` if the queue is
        empty."""
        return self._goals[0] if self._goals else None

    def append_goal(self, goal: np.ndarray) -> None:
        """Add a goal to the back of the queue."""
        self._goals.append(np.asarray(goal, dtype=np.float64))

    def prepend_goal(self, goal: np.ndarray) -> None:
        """Insert a goal at the front of the queue (becomes the immediate next
        target)."""
        self._goals.appendleft(np.asarray(goal, dtype=np.float64))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollout(self, current_q: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out N trajectories from ``current_q`` using action sequences
        ``actions``.

        Args:
            current_q: ``(4, 3)`` current joint angles.
            actions:   ``(N, H, 4, 3)`` sampled action sequences.

        Returns:
            ``(N, H+1, 4, 3)`` state trajectories (includes initial state).
        """
        n_seqs, h_len = actions.shape[0], actions.shape[1]
        q_trajs = np.empty((n_seqs, h_len + 1, _N_JOINTS, 3), dtype=np.float64)
        q_trajs[:, 0] = current_q[np.newaxis]

        for t in range(h_len):
            q_trajs[:, t + 1] = _compose_rotvec(q_trajs[:, t], actions[:, t])

        return q_trajs

    def _cost(self, q_trajs: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        """Compute terminal cost for each of the N sampled trajectories.

        Args:
            q_trajs:  ``(N, H+1, 4, 3)`` state trajectories.
            target_q: ``(4, 3)``         target joint configuration.

        Returns:
            ``(N,)`` cost per trajectory.
        """
        return ((q_trajs[:, -1] - target_q[np.newaxis]) ** 2).sum(axis=(-2, -1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        current_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample action sequences and return the best one.

        Args:
            current_q: ``(4, 3)`` current axis-angle joint angles for
                       [left_collar, left_shoulder, left_elbow, left_wrist].

        Returns:
            Tuple of:

            - ``first_action`` ``(4, 3)``: best delta to apply at the current step.
            - ``plan`` ``(H, 4, 3)``: full best action sequence.

        Raises:
            RuntimeError: If the goal queue is empty.
        """
        if self.current_goal is None:
            raise RuntimeError(
                "Goal queue is empty. Add a goal before calling solve()."
            )
        current_q = np.asarray(current_q, dtype=np.float64)
        target_q = self.current_goal

        # Warm-start: shift previous best plan by one step; fill last with zeros
        if self._prev_best is not None:
            mean = np.concatenate(
                [self._prev_best[1:], np.zeros((1, _N_JOINTS, 3))], axis=0
            )
        else:
            mean = np.zeros((self._config.horizon, _N_JOINTS, 3), dtype=np.float64)

        actions = np.random.normal(
            loc=mean,
            scale=self._config.max_angle_delta,
            size=(self._config.n_samples, self._config.horizon, _N_JOINTS, 3),
        )

        q_trajs = self._rollout(current_q, actions)
        costs = self._cost(q_trajs, target_q)

        best_idx = np.argmin(costs)
        best_plan = actions[best_idx]

        self._prev_best = best_plan
        return best_plan[0], best_plan

    def reset_warmstart(self) -> None:
        """Reset the warm-start plan (call before re-running from a new initial
        pose)."""
        self._prev_best = None

    def step(
        self,
        current_q: np.ndarray,
    ) -> np.ndarray:
        """Perform one MPC step.

        Samples action sequences, applies the best first action to ``current_q``
        via SO(3) composition, and returns the updated joint angles.  If the
        current goal is reached (L2 distance < ``goal_threshold``) and more
        goals remain, the front goal is popped and the warm-start is reset.
        If ``visualize=True`` was set at construction, the live window is
        updated automatically.

        Args:
            current_q: ``(4, 3)`` current axis-angle joint angles.

        Returns:
            ``(4, 3)`` updated axis-angle joint angles.
        """
        first_action, _ = self.solve(current_q)
        next_q = _compose_rotvec(np.asarray(current_q, dtype=np.float64), first_action)

        # Advance goal queue when the current goal is reached
        goal = self.current_goal
        if goal is not None:
            dist = float(np.linalg.norm(next_q - goal))
            if dist < self._goal_threshold and len(self._goals) > 1:
                self._goals.popleft()
                self.reset_warmstart()

        if self._vis_config is not None:
            if self._vis is None:
                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    list(self._goals),
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                )
            dist = float(np.linalg.norm(next_q - self.current_goal))
            self._vis.update_step(next_q, dist=dist)

        return next_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SMPL left arm MPC with live visualization"
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of MPC steps")
    parser.add_argument("--samples", type=int, default=256, help="CEM sample count")
    parser.add_argument("--horizon", type=int, default=10, help="MPC horizon")
    parser.add_argument(
        "--no-vis", action="store_true", help="Disable live visualization"
    )
    args = parser.parse_args()

    demo_fk = SmplLeftArmFK()

    demo_initial_q = np.zeros((4, 3))
    demo_goals = [
        np.array(
            [
                [0.3, 0.3, 0.3],  # left_collar
                [0.0, -1.45, 0.0],  # left_shoulder
                [0.0, 0.0, 0.4],  # left_elbow
                [0.0, 0.0, 0.0],  # left_wrist
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],  # left_collar
                [0.0, -0.8, 0.0],  # left_shoulder
                [0.0, 0.0, 0.8],  # left_elbow
                [0.0, 0.0, 0.0],  # left_wrist
            ]
        ),
    ]

    demo_mpc = SmplLeftArmMPC(
        horizon=args.horizon,
        n_samples=args.samples,
        goals=demo_goals,
        visualize=not args.no_vis,
        fk=demo_fk,
    )

    demo_q = demo_initial_q.copy()
    for _ in range(args.steps):
        demo_q = demo_mpc.step(demo_q)

    plt.ioff()
    plt.show()
