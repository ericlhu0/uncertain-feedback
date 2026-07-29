"""Robot-joint-delta action space: sample robot deltas, cost the human arm.

The human action space leaves the robot to chase each commanded configuration
through grasp FK + IK — which lets it command motions the grasp cannot
physically transmit (forearm roll) and lets per-joint IK clipping bend the
executed direction. Here the sample space is the robot's own joint deltas:
every rollout is robot-feasible by construction, and each rolled-out ee pose
is mapped through the rigid measured grasp back to a human arm configuration
(:func:`~uncertain_feedback.planners.mpc.kinematics.project_forearm_frames`),
so every cost term still scores the human arm. The projection residual — the
part of the implied motion the arm cannot follow — is itself penalized, so the
planner prefers actions the grasp can actually transmit. Execution sends the
best joint target straight to the robot (:meth:`ExecutionEnv.execute_robot`),
bypassing the IK path entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.action_spaces.base import (
    ActionSpace,
    RolloutBatch,
    StageCost,
)
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    Q_ELBOW,
    Q_SHOULDER,
    SmplLeftArmFK,
    project_forearm_frames,
)


@dataclass(frozen=True)
class RobotActionsConfig:
    """Robot-joint-delta sampling parameters (the ``robot_actions:`` section).

    Args:
        max_joint_delta: Per-step inf-norm cap on sampled robot joint deltas
            (radians) — the same cap execution enforces.
        joint_delta_std: Std of the joint-delta sampling noise around the
            warm-started mean (radians). None means a third of the cap.
        infeasibility_weight: Weight on the grasp-transmission residual (per
            metre of projection error / radian of untransmitted roll).
        max_grasp_residual: Per-frame residual above which a rollout's leading
            frames count as breaking the grasp, discarding the rollout.
        grasp_residual_frames: How many leading frames that gate covers.
    """

    max_joint_delta: float = 0.005
    joint_delta_std: float | None = None
    infeasibility_weight: float = 1.0
    max_grasp_residual: float = 0.02
    grasp_residual_frames: int = 3


class RobotJointActions(ActionSpace):
    """Samples deltas in the robot's 7 joints; costs the projected human arm."""

    def __init__(
        self,
        cfg: RobotActionsConfig,
        fk: SmplLeftArmFK,
        rng: np.random.Generator | None,
        n_samples: int,
        horizon: int,
        spine3_pos: np.ndarray,
        spine3_aa: np.ndarray,
    ) -> None:
        self._fk = fk
        self._rng = rng
        self._n_samples = n_samples
        self._horizon = horizon
        self._spine3_pos = spine3_pos
        self._spine3_aa = spine3_aa
        self._max_joint_delta = float(cfg.max_joint_delta)
        # Std well below the inf-norm cap keeps mean+noise inside it for most
        # samples; at std == cap nearly every sample saturates and the uniform
        # rescale drowns the warm-started mean in noise.
        self._joint_delta_std = (
            float(cfg.joint_delta_std)
            if cfg.joint_delta_std is not None
            else self._max_joint_delta / 3.0
        )
        self._infeasibility_weight = float(cfg.infeasibility_weight)
        self._max_grasp_residual = float(cfg.max_grasp_residual)
        self._grasp_residual_frames = int(cfg.grasp_residual_frames)

    def rollouts(
        self, env: ExecutionEnv, current_q: np.ndarray, mean: np.ndarray
    ) -> RolloutBatch:
        # The grasp must exist before the robot state is read: establishing it
        # (first sim solve) moves the robot to the grasp configuration.
        grasp = env.current_grasp(current_q)
        robot_q = env.current_robot_q()

        # Each sampled action is capped at ``max_joint_delta`` (inf-norm,
        # uniformly scaled) — the same cap ``execute_robot`` enforces — so the
        # rollouts never plan a per-step motion execution will refuse; without
        # this the warm start random-walks to ever larger actions whose
        # predicted progress execution then scales away.
        lower, upper = env.robot_joint_limits()
        rng = self._rng if self._rng is not None else np.random
        actions = rng.normal(
            loc=mean,
            scale=self._joint_delta_std,
            size=(self._n_samples, self._horizon, Q_DIM),
        )
        largest = np.abs(actions).max(axis=-1, keepdims=True)
        actions *= np.minimum(1.0, self._max_joint_delta / largest)
        trajs = np.empty((self._n_samples, self._horizon + 1, Q_DIM), dtype=np.float64)
        trajs[:, 0] = robot_q[np.newaxis]
        for t in range(self._horizon):
            trajs[:, t + 1] = np.clip(trajs[:, t] + actions[:, t], lower, upper)

        # Robot FK → ee poses → inverse of the rigid measured grasp → implied
        # forearm frames → grasp-anchored projection onto the arm manifold.
        # Deferred: envs.sim_mannequin imports envs.grasp, which imports this
        # package back — a module-level import here closes that cycle.
        from uncertain_feedback.envs.sim_mannequin import (  # pylint: disable=import-outside-toplevel
            _SMPL_TO_PB,
        )

        ee_pos, ee_rot = env.robot_fk().ee_pose(trajs)
        forearm_rot_pb = ee_rot @ grasp.rotation.inv().as_matrix()
        aa_trajs, wrist_pos, residual = project_forearm_frames(
            self._fk,
            ee_pos @ _SMPL_TO_PB,
            _SMPL_TO_PB.T @ forearm_rot_pb,
            grasp.position,
            np.asarray(current_q, dtype=np.float64),
            self._spine3_pos,
            self._spine3_aa,
        )
        return RolloutBatch(
            actions=actions,
            aa_trajs=aa_trajs,
            wrist_pos=wrist_pos,
            robot_trajs=trajs,
            grasp_residual=residual,
        )

    def shape_costs(self, batch: RolloutBatch, stage_cost: StageCost) -> np.ndarray:
        """Stage cost + infeasibility penalty, hard-gated on the leading frames.

        Rollouts whose *leading* frames (the ones about to be executed; the
        tail is re-solved every step) exceed ``max_grasp_residual`` are
        discarded outright — they would move the grasp in a way the arm cannot
        follow. The whole-horizon residual is squared so its units match the
        squared-distance goal costs. If no sample keeps the grasp, the
        least-violating one is taken.
        """
        residual = batch.grasp_residual
        assert residual is not None
        leading = residual[:, 1 : 1 + self._grasp_residual_frames].max(axis=1)
        feasible = leading <= self._max_grasp_residual
        if np.any(feasible):
            costs = stage_cost(batch) + (
                self._infeasibility_weight * (residual[:, 1:] ** 2).mean(axis=1)
            )
            costs = np.where(feasible, costs, np.inf)
        else:
            costs = leading
        return costs

    def command(self, batch: RolloutBatch, best_idx: int) -> np.ndarray:
        assert batch.robot_trajs is not None
        return batch.robot_trajs[best_idx, 1]

    def execute(
        self, env: ExecutionEnv, current_q: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        _ = current_q
        return env.execute_robot(command)

    def hold(self, env: ExecutionEnv, current_q: np.ndarray) -> np.ndarray:
        env.current_grasp(np.asarray(current_q, dtype=np.float64))
        return env.execute_robot(env.current_robot_q())

    def tracking_cost(self, q_target: np.ndarray) -> StageCost:
        """Terminal distance to a playback frame, on the actuatable DOFs.

        The clavicle block is excluded: the robot cannot actuate the shoulder
        girdle, so any clavicle error in the playback frame is unreachable
        either way. The shoulder error is geodesic, not rotvec L2 — measured
        shoulder rotations sit near the ±pi rotvec boundary (the anatomical
        decode of a bent arm carries a ~pi twist), where the rotvec sign flips
        between steps and a plain L2 would see a phantom ~2pi error. Extra
        costs are not applied — playback follows a trajectory the user already
        validated, matching the direct-playback semantics of the human action
        space.
        """
        q_target = np.asarray(q_target, dtype=np.float64)
        hinge = self._fk.elbow_hinge_axis
        target_rot_inv = Rotation.from_rotvec(q_target[Q_SHOULDER]).inv()

        def cost(batch: RolloutBatch) -> np.ndarray:
            aa_trajs = batch.aa_trajs
            # Every horizon frame is costed, not just the terminal one: the
            # target is a (near-)stationary frame, and a terminal-only cost
            # leaves the executed first step free to wander around it.
            shoulder = aa_trajs[:, 1:, 1]
            relative = (
                (target_rot_inv * Rotation.from_rotvec(shoulder.reshape(-1, 3)))
                .as_rotvec()
                .reshape(shoulder.shape)
            )
            shoulder_err = (relative**2).sum(axis=-1).mean(axis=-1)
            elbow_err = ((aa_trajs[:, 1:, 2] @ hinge - q_target[Q_ELBOW]) ** 2).mean(
                axis=-1
            )
            return shoulder_err + elbow_err

        return cost
