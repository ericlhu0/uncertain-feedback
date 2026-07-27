"""Robot-joint-action MPC: sample robot deltas, cost the human arm.

The standard planners sample deltas in the *human* arm state and leave the
robot to chase each commanded configuration through grasp FK + IK — which lets
them command motions the grasp cannot physically transmit (forearm roll) and
lets per-joint IK clipping bend the executed direction. Here the sample space
is the robot's own joint deltas: every rollout is robot-feasible by
construction, and each rolled-out ee pose is mapped through the rigid measured
grasp back to a human arm configuration
(:func:`~uncertain_feedback.planners.mpc.kinematics.project_forearm_frames`),
so every cost term still scores the human arm. The projection residual — the
part of the implied motion the arm cannot follow — is itself penalized, so the
planner prefers actions the grasp can actually transmit. Execution sends the
best joint target straight to the robot (:meth:`ExecutionEnv.execute_robot`),
bypassing the IK path entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.arm_mpc_cartesian_no_mdm import (
    ArmMPCCartesianNoMDM,
)
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    project_forearm_frames,
)

if TYPE_CHECKING:
    from uncertain_feedback.envs.grasp import MeasuredGrasp


class _RobotActionsMixin:
    """Robot-joint-delta sampling for a :class:`SmplLeftArmMPC` subclass.

    Combines with a planner that supplies the attributes below; the env must
    implement the robot-action interface of :class:`ExecutionEnv`
    (``robot_fk``/``current_robot_q``/``robot_joint_limits``/``current_grasp``/
    ``execute_robot``).
    """

    # Supplied by the planner this is mixed into.
    _env: ExecutionEnv
    _fk: SmplLeftArmFK
    _rng: np.random.Generator | None
    _horizon: int
    _n_mpc_samples: int
    _prev_best: np.ndarray | None
    _extra_costs: CompositeTrajectoryCost
    _spine3_pos: np.ndarray
    _spine3_aa: np.ndarray
    current_cartesian_goal: np.ndarray | None

    def _init_robot_actions(
        self,
        max_robot_joint_delta: float,
        robot_joint_delta_std: float | None,
        robot_infeasibility_weight: float,
        max_grasp_residual: float,
        grasp_residual_frames: int,
    ) -> None:
        self._max_robot_joint_delta = float(max_robot_joint_delta)
        # Std well below the inf-norm cap keeps mean+noise inside it for most
        # samples; at std == cap nearly every sample saturates and the uniform
        # rescale drowns the warm-started mean in noise.
        self._robot_joint_delta_std = (
            float(robot_joint_delta_std)
            if robot_joint_delta_std is not None
            else self._max_robot_joint_delta / 3.0
        )
        self._robot_infeasibility_weight = float(robot_infeasibility_weight)
        self._max_grasp_residual = float(max_grasp_residual)
        self._grasp_residual_frames = int(grasp_residual_frames)

    def _robot_rollouts(self, robot_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample warm-started joint-delta sequences and integrate them.

        Each sampled action is capped at ``max_robot_joint_delta`` (inf-norm,
        uniformly scaled) — the same cap ``execute_robot`` enforces — so the
        rollouts never plan a per-step motion execution will refuse; without
        this the warm start random-walks to ever larger actions whose predicted
        progress execution then scales away. Returns ``(N, H+1, 7)`` joint
        trajectories (limit-clipped per step) and the ``(N, H, 7)`` actions
        that produced them.
        """
        lower, upper = self._env.robot_joint_limits()
        if self._prev_best is not None:
            mean = np.concatenate([self._prev_best[1:], np.zeros((1, Q_DIM))], axis=0)
        else:
            mean = np.zeros((self._horizon, Q_DIM), dtype=np.float64)
        rng = self._rng if self._rng is not None else np.random
        actions = rng.normal(
            loc=mean,
            scale=self._robot_joint_delta_std,
            size=(self._n_mpc_samples, self._horizon, Q_DIM),
        )
        largest = np.abs(actions).max(axis=-1, keepdims=True)
        actions *= np.minimum(1.0, self._max_robot_joint_delta / largest)
        trajs = np.empty(
            (self._n_mpc_samples, self._horizon + 1, Q_DIM), dtype=np.float64
        )
        trajs[:, 0] = robot_q[np.newaxis]
        for t in range(self._horizon):
            trajs[:, t + 1] = np.clip(trajs[:, t] + actions[:, t], lower, upper)
        return trajs, actions

    def _project_rollouts(
        self, robot_trajs: np.ndarray, current_q: np.ndarray, grasp: MeasuredGrasp
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Human-arm rollouts implied by robot joint rollouts.

        Robot FK → ee poses → inverse of the rigid measured grasp → implied
        forearm frames → grasp-anchored projection onto the arm manifold.
        Returns ``(..., 3, 3)`` arm axis-angles, ``(..., 3)`` wrist positions,
        and the ``(...,)`` infeasibility residual.
        """
        # Deferred: envs.sim_mannequin imports envs.grasp, which imports this
        # package back — a module-level import here closes that cycle.
        from uncertain_feedback.envs.sim_mannequin import (  # pylint: disable=import-outside-toplevel
            _SMPL_TO_PB,
        )

        ee_pos, ee_rot = self._env.robot_fk().ee_pose(robot_trajs)
        forearm_rot_pb = ee_rot @ grasp.rotation.inv().as_matrix()
        return project_forearm_frames(
            self._fk,
            ee_pos @ _SMPL_TO_PB,
            _SMPL_TO_PB.T @ forearm_rot_pb,
            grasp.position,
            current_q,
            self._spine3_pos,
            self._spine3_aa,
        )

    def _robot_solve(self, current_q: np.ndarray, base_cost) -> np.ndarray:
        """Best next robot joint target under ``base_cost`` + infeasibility.

        Rollouts whose *leading* frames (the ones about to be executed; the
        tail is re-solved every step) exceed ``max_grasp_residual`` are
        discarded outright — they would move the grasp in a way the arm cannot
        follow. Among the survivors, ``base_cost(aa_trajs, wrist_pos) -> (N,)``
        scores the projected human rollouts plus the squared whole-horizon
        residual (squared so its units match the squared-distance goal costs).
        If no sample keeps the grasp, the least-violating one is taken.
        """
        current_q = np.asarray(current_q, dtype=np.float64)
        # The grasp must exist before the robot state is read: establishing it
        # (first sim solve) moves the robot to the grasp configuration.
        grasp = self._env.current_grasp(current_q)
        robot_q = self._env.current_robot_q()
        robot_trajs, actions = self._robot_rollouts(robot_q)
        aa_trajs, wrist_pos, residual = self._project_rollouts(
            robot_trajs, current_q, grasp
        )
        leading = residual[:, 1 : 1 + self._grasp_residual_frames].max(axis=1)
        feasible = leading <= self._max_grasp_residual
        if np.any(feasible):
            costs = base_cost(aa_trajs, wrist_pos) + (
                self._robot_infeasibility_weight * (residual[:, 1:] ** 2).mean(axis=1)
            )
            costs = np.where(feasible, costs, np.inf)
        else:
            costs = leading
        best_idx = int(np.argmin(costs))
        self._prev_best = actions[best_idx]
        return robot_trajs[best_idx, 1]

    def _robot_cartesian_cost(
        self, aa_trajs: np.ndarray, wrist_pos: np.ndarray
    ) -> np.ndarray:
        """Terminal wrist distance to the front Cartesian goal + extra costs."""
        target = self.current_cartesian_goal
        wrist_rel = wrist_pos[:, -1] - self._spine3_pos
        return ((wrist_rel - target) ** 2).sum(axis=-1) + self._extra_costs(aa_trajs)


class ArmMPCCartesianNoMDMRobot(_RobotActionsMixin, ArmMPCCartesianNoMDM):
    """Pure Cartesian wrist-goal MPC acting in robot joint space.

    Args:
        max_robot_joint_delta: Per-step inf-norm cap on sampled robot joint
            deltas (radians) — the same cap execution enforces.
        robot_joint_delta_std: Std of the joint-delta sampling noise around
            the warm-started mean (radians). None means a third of the cap.
        robot_infeasibility_weight: Weight on the grasp-transmission residual
            (per metre of projection error / radian of untransmitted roll).
        max_grasp_residual: Per-frame residual above which a rollout's leading
            frames count as breaking the grasp, discarding the rollout.
        grasp_residual_frames: How many leading frames that gate covers.
    """

    def __init__(
        self,
        cartesian_goals: list[np.ndarray],
        initial_q: np.ndarray,
        cartesian_threshold: float = 0.05,
        horizon: int = 10,
        n_mpc_samples: int = 512,
        max_angle_delta: float = 0.0025,
        max_robot_joint_delta: float = 0.005,
        robot_joint_delta_std: float | None = None,
        robot_infeasibility_weight: float = 1.0,
        max_grasp_residual: float = 0.02,
        grasp_residual_frames: int = 3,
        goal_threshold: float = 0.1,
        visualize: bool = False,
        fk: SmplLeftArmFK | None = None,
        spine3_pos: np.ndarray | None = None,
        spine3_aa: np.ndarray | None = None,
        body_pos: np.ndarray | None = None,
        extra_costs: CompositeTrajectoryCost | None = None,
        seed: int | None = None,
        env: ExecutionEnv | None = None,
    ) -> None:
        if env is None:
            raise ValueError("ArmMPCCartesianNoMDMRobot requires a robot env.")
        super().__init__(
            cartesian_goals=cartesian_goals,
            initial_q=initial_q,
            cartesian_threshold=cartesian_threshold,
            horizon=horizon,
            n_mpc_samples=n_mpc_samples,
            max_angle_delta=max_angle_delta,
            goal_threshold=goal_threshold,
            visualize=visualize,
            fk=fk,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            body_pos=body_pos,
            extra_costs=extra_costs,
            seed=seed,
            env=env,
        )
        self._init_robot_actions(
            max_robot_joint_delta,
            robot_joint_delta_std,
            robot_infeasibility_weight,
            max_grasp_residual,
            grasp_residual_frames,
        )

    def step(self, current_q: np.ndarray) -> np.ndarray:
        """Perform one robot-action MPC step."""
        current_q = np.asarray(current_q, dtype=np.float64)
        if not self._cartesian_goals:
            self._env.current_grasp(current_q)
            return self._env.execute_robot(self._env.current_robot_q())

        target = self._robot_solve(current_q, self._robot_cartesian_cost)
        next_q = self._env.execute_robot(target)
        goal, dist = self._cartesian_progress(next_q)
        self._update_cartesian_vis(next_q, dist, goal)
        return next_q
