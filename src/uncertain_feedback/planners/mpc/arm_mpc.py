# pylint: disable=duplicate-code
"""Sampling-based MPC for the SMPL left arm in 7-DOF joint space."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.kinematic import KinematicEnv
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    ElbowHeightCost,
)
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    _compose_q,
    q_to_arm_aa,
)

if TYPE_CHECKING:
    from uncertain_feedback.utils.plot import ArmVisualizer


def _as_state_q(value: np.ndarray, name: str) -> np.ndarray:
    """Return a clavicle/shoulder/elbow planner state."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (Q_DIM,):
        raise ValueError(
            f"{name} must have shape ({Q_DIM},) for "
            "[clavicle rotvec, shoulder rotvec, elbow angle], "
            f"got {arr.shape}"
        )
    return arr


@dataclass
class _VisConfig:
    fk: SmplLeftArmFK
    spine_pos: np.ndarray | None
    spine_aa: np.ndarray | None
    body_pos: np.ndarray | None = None
    capture: bool = False
    compact: bool = False


# ---------------------------------------------------------------------------
# MPC controller
# ---------------------------------------------------------------------------


class SmplLeftArmMPC:
    """Sampling-based MPC for the SMPL left arm.

    Draws ``n_mpc_samples`` random action sequences, rolls each out, and returns
    the one with the lowest terminal cost.

    The controller maintains a queue of goals.  It always targets the first
    goal in the queue.  Once the controller reaches that goal (within
    ``goal_threshold``), the goal is popped and the next one becomes active.
    Goals can be added at any time via :meth:`append_goal` or
    :meth:`prepend_goal`.

    Args:
        horizon:         Number of look-ahead steps.
        n_mpc_samples:   Number of candidate action sequences sampled per
                         ``solve`` call.
        max_angle_delta: Standard deviation of the sampling distribution
                         (radians).
        goals:           Initial list of ``(7,)`` target joint configurations.
        goal_threshold:  L2 distance in the 7-DOF state below which the
                         current goal is considered reached (default: 0.01).
        visualize:       If ``True``, open a live matplotlib window and update
                         it each time :meth:`step` is called.  Requires
                         ``fk`` to also be provided.
        fk:              :class:`SmplLeftArmFK` instance (required when
                         ``visualize=True``).
        spine3_pos:      ``(3,)`` world position of spine3 (optional).
        spine3_aa:       ``(3,)`` world axis-angle of spine3 (optional).
        body_pos:        ``(22, 3)`` background skeleton joint positions
                         (optional).
        seed:            Seed for planner-local action sampling (optional).
        env:             Execution environment realizing each commanded step
                         (default: :class:`KinematicEnv`, exact pass-through).
    """

    def __init__(
        self,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.001,
        goals: list[np.ndarray] | None = None,
        goal_threshold: float = 0.01,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
        seed: int | None = None,
        env: ExecutionEnv | None = None,
    ) -> None:
        self._horizon = horizon
        self._n_mpc_samples = n_mpc_samples
        self._max_angle_delta = max_angle_delta
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.visualize = visualize
        self._extra_costs = extra_costs or CompositeTrajectoryCost()
        self._env: ExecutionEnv = env if env is not None else KinematicEnv()

        self._goals: deque[np.ndarray] = deque(
            [_as_state_q(g, "goal") for g in goals] if goals else []
        )
        self._goal_threshold = goal_threshold

        self._fk: SmplLeftArmFK = fk if fk is not None else SmplLeftArmFK()

        if visualize:
            if fk is None:
                raise ValueError("visualize=True requires `fk` to be provided.")
            self._vis_config: _VisConfig | None = _VisConfig(
                fk, spine3_pos, spine3_aa, body_pos=body_pos
            )
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

    def goal_reached(self, q: np.ndarray) -> bool:
        """Whether ``q`` has reached the final goal so the rollout can end.

        Only the last remaining goal counts: while earlier goals are still
        queued the rollout has not finished. Returns ``True`` when ``q`` is
        within ``goal_threshold`` (L2 in the 7-DOF state) of that final goal.
        :func:`~uncertain_feedback.planners.run.run_planning_loop` polls this
        (together with :attr:`mdm_ready_to_terminate`) to stop early.
        """
        goal = self.current_goal
        if goal is None or len(self._goals) > 1:
            return False
        dist = float(np.linalg.norm(np.asarray(q, dtype=np.float64) - goal))
        return dist < self._goal_threshold

    @property
    def mdm_ready_to_terminate(self) -> bool:
        """Whether MDM playback state permits ending the rollout.

        Always ``True`` for planners without an MDM correction phase; the MDM
        subclass overrides this to keep the rollout alive until a queued
        correction has finished playing back.
        """
        return True

    def get_visualizer(self) -> ArmVisualizer | None:
        """Return the active live visualizer, or ``None`` if not yet created."""
        return self._vis

    def close_visualizer(self) -> ArmVisualizer | None:
        """Close the live window and detach the visualizer.

        Returns the detached visualizer so its recorded frames can be reused
        (e.g. to prepend pre-MDM frames after generation).
        """
        vis = self._vis
        if vis is not None:
            vis.close()
        self._vis = None
        return vis

    def set_extra_costs(self, costs: CompositeTrajectoryCost) -> None:
        """Replace the active extra cost terms (e.g. after a preference update)."""
        self._extra_costs = costs
        if self._vis is not None:
            self._vis.update_elbow_height_range(self._elbow_height_world_range())

    def _elbow_height_world_range(self) -> tuple[float, float] | None:
        """Return configured elbow-height bounds as world-space Y coordinates."""
        for term in self._extra_costs._terms:  # pylint: disable=protected-access
            if isinstance(term, ElbowHeightCost):
                spine_y = float(term.context.spine3_pos[1])
                return (
                    spine_y + float(term.min_height),
                    spine_y + float(term.max_height),
                )
        return None

    def set_visualization_mode(self, capture: bool, compact: bool) -> None:
        """Set capture and compact flags on the vis config (call before first step)."""
        if self._vis_config is not None:
            self._vis_config.capture = capture
            self._vis_config.compact = compact

    def append_goal(self, goal: np.ndarray) -> None:
        """Add a goal to the back of the queue."""
        self._goals.append(_as_state_q(goal, "goal"))

    def prepend_goal(self, goal: np.ndarray) -> None:
        """Insert a goal at the front of the queue (becomes the immediate next
        target)."""
        self._goals.appendleft(_as_state_q(goal, "goal"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_actions(self, mean: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
        rng = self._rng if self._rng is not None else np.random
        return rng.normal(
            loc=mean,
            scale=self._max_angle_delta,
            size=size,
        )

    def _rollout(self, current_q: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out N trajectories from ``current_q`` using action sequences
        ``actions``.

        Args:
            current_q: ``(7,)`` current joint angles.
            actions:   ``(N, H, 7)`` sampled action sequences.

        Returns:
            ``(N, H+1, 7)`` state trajectories (includes initial state).
        """
        n_seqs, h_len = actions.shape[0], actions.shape[1]
        q_trajs = np.empty((n_seqs, h_len + 1, Q_DIM), dtype=np.float64)
        q_trajs[:, 0] = current_q[np.newaxis]

        for t in range(h_len):
            q_trajs[:, t + 1] = _compose_q(q_trajs[:, t], actions[:, t])

        return q_trajs

    def _cost(self, q_trajs: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        """Compute terminal cost for each of the N sampled trajectories.

        Args:
            q_trajs:  ``(N, H+1, 7)`` state trajectories.
            target_q: ``(7,)`` target joint configuration.

        Returns:
            ``(N,)`` cost per trajectory.
        """
        joint_cost = ((q_trajs[:, -1] - target_q[np.newaxis]) ** 2).sum(axis=-1)
        aa_trajs = q_to_arm_aa(q_trajs, self._fk.elbow_hinge_axis)
        return joint_cost + self._extra_costs(aa_trajs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        current_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample action sequences and return the best one.

        Args:
            current_q: ``(7,)`` clavicle/shoulder/elbow state.

        Returns:
            Tuple of:

            - ``first_action`` ``(7,)``: best delta to apply at the current step.
            - ``plan`` ``(H, 7)``: full best action sequence.

        Raises:
            RuntimeError: If the goal queue is empty.
        """
        if self.current_goal is None:
            raise RuntimeError(
                "Goal queue is empty. Add a goal before calling solve()."
            )
        current_q = _as_state_q(current_q, "current_q")
        target_q = self.current_goal

        # Warm-start: shift previous best plan by one step; fill last with zeros
        if self._prev_best is not None:
            mean = np.concatenate([self._prev_best[1:], np.zeros((1, Q_DIM))], axis=0)
        else:
            mean = np.zeros((self._horizon, Q_DIM), dtype=np.float64)

        actions = self._sample_actions(
            mean, (self._n_mpc_samples, self._horizon, Q_DIM)
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
        via SO(3) composition, realizes the result through the execution env,
        and returns the achieved joint angles.  If the current goal is reached
        (L2 distance < ``goal_threshold``) and more goals remain, the front
        goal is popped and the warm-start is reset.
        If ``visualize=True`` was set at construction, the live window is
        updated automatically.

        Args:
            current_q: ``(7,)`` clavicle/shoulder/elbow state.

        Returns:
            ``(7,)`` achieved planner state.
        """
        first_action, _ = self.solve(current_q)
        next_q = self._env.execute(
            _compose_q(_as_state_q(current_q, "current_q"), first_action)
        )

        # Advance goal queue when the current goal is reached
        goal = self.current_goal
        if goal is not None:
            dist = float(np.linalg.norm(next_q - goal))
            if dist < self._goal_threshold and len(self._goals) > 1:
                self._goals.popleft()
                self.reset_warmstart()

        if self._vis_config is not None:
            if self._vis is None:
                from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
                    ArmVisualizer,
                )

                self._vis = ArmVisualizer(self._vis_config.fk)
                self._vis.open_live(
                    q_to_arm_aa(self._goals[-1], self._fk.elbow_hinge_axis),
                    self._vis_config.spine_pos,
                    self._vis_config.spine_aa,
                    body_pos=self._vis_config.body_pos,
                    compact=self._vis_config.compact,
                    elbow_height_range=self._elbow_height_world_range(),
                )
                if self._vis_config.capture:
                    self._vis.start_capture()
            dist = float(np.linalg.norm(next_q - self.current_goal))
            self._vis.update_step(
                q_to_arm_aa(next_q, self._fk.elbow_hinge_axis), dist=dist
            )

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

    demo_initial_q = np.zeros(Q_DIM)
    demo_goals = [
        np.array([0.0, -1.45, 0.0, 0.0, 0.0, 0.4, 0.0]),
        np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.8, 0.0]),
    ]

    demo_mpc = SmplLeftArmMPC(
        horizon=args.horizon,
        n_mpc_samples=args.samples,
        goals=demo_goals,
        visualize=not args.no_vis,
        fk=demo_fk,
    )

    demo_q = demo_initial_q.copy()
    for _ in range(args.steps):
        demo_q = demo_mpc.step(demo_q)

    plt.ioff()
    plt.show()
