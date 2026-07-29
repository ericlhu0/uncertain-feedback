"""Kinematic stand-in env for previewing a plan offline.

Snapshots the real env's robot chain, measured grasp, joint state, limits, and
step cap, then lets a planner roll out against them without commanding
anything. Both action spaces are served, because a preview is only worth
watching if it runs the planner that is about to run live:

* ``execute_robot`` (robot-action planners) stores the joint target and reports
  the human configuration the grasp projection implies — the same forward model
  the planner samples with.
* ``execute`` (human-action planners) drives the robot the way
  :meth:`RealEnv._drive` does: the grasp on the commanded arm gives a gripper
  pose, the source env's own IK solves it from the previewed robot's previous
  configuration, and the step is rate-capped and clipped to the joint box. The
  arm itself tracks the command exactly, which is the preview's standing
  assumption.

Either way the previewed robot and arm stay consistent by construction instead
of the robot chasing a human-space plan through a *second* IK, and the executed
joint targets are kept in :attr:`robot_trajectory` so the preview can animate
the actual planned robot motion.

Exact IK is delegated to the source env rather than reimplemented: a planner
gated on robot reachability is only meaningful against the solver and padded
joint box that will run the plan.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.grasp import MeasuredGrasp, forearm_frame_fk
from uncertain_feedback.envs.robot_fk import RobotChainFK
from uncertain_feedback.envs.sim_mannequin import _SMPL_TO_PB
from uncertain_feedback.planners.mpc.kinematics import (
    SmplLeftArmFK,
    project_forearm_frames,
)


class RobotPlanPreviewEnv(ExecutionEnv):
    """Offline double of a robot env, frozen at one measured state."""

    def __init__(
        self,
        fk: SmplLeftArmFK,
        chain: RobotChainFK,
        grasp: MeasuredGrasp,
        robot_q: np.ndarray,
        joint_limits: tuple[np.ndarray, np.ndarray],
        q_ref: np.ndarray,
        spine3_pos: np.ndarray | None,
        spine3_aa: np.ndarray | None,
        ik_env: ExecutionEnv,
        max_joint_delta: float,
    ) -> None:
        super().__init__()
        self._fk_arm = fk
        self._chain = chain
        self._grasp = grasp
        self._robot_q = np.asarray(robot_q, dtype=np.float64).copy()
        self._lower, self._upper = joint_limits
        self._q_ref = np.asarray(q_ref, dtype=np.float64)
        self._preview_spine3_pos = spine3_pos
        self._preview_spine3_aa = spine3_aa
        self._ik_env = ik_env
        self._max_joint_delta = float(max_joint_delta)
        self.robot_trajectory: list[np.ndarray] = [self._robot_q.copy()]

    def robot_fk(self) -> RobotChainFK:
        return self._chain

    def current_robot_q(self) -> np.ndarray:
        return self._robot_q.copy()

    def robot_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

    def robot_max_joint_delta(self) -> float:
        return self._max_joint_delta

    def solve_robot_ik_exact(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> np.ndarray | None:
        return self._ik_env.solve_robot_ik_exact(target_pos, target_quat, q_seed)

    def solve_robot_ik_exact_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._ik_env.solve_robot_ik_exact_batch(target_pos, target_quat, q_seed)

    def track_robot_ik_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._ik_env.track_robot_ik_batch(target_pos, target_quat, q_seed)

    def current_grasp(self, q: np.ndarray) -> MeasuredGrasp:
        return self._grasp

    def execute_robot(self, target: np.ndarray) -> np.ndarray:
        self._robot_q = np.clip(
            np.asarray(target, dtype=np.float64), self._lower, self._upper
        )
        self.robot_trajectory.append(self._robot_q.copy())
        return self._measured_q()

    def _measured_q(self) -> np.ndarray:
        ee_pos, ee_rot = self._chain.ee_pose(self._robot_q)
        forearm_rot = ee_rot @ self._grasp.rotation.inv().as_matrix()
        arm_aa, _wrist, _residual = project_forearm_frames(
            self._fk_arm,
            ee_pos @ _SMPL_TO_PB,
            _SMPL_TO_PB.T @ forearm_rot,
            self._grasp.position,
            self._q_ref,
            self._preview_spine3_pos,
            self._preview_spine3_aa,
        )
        hinge = self._fk_arm.elbow_hinge_axis
        return np.concatenate((arm_aa[0], arm_aa[1], [float(arm_aa[2] @ hinge)]))

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        """Carry the robot along a human-action plan; the arm tracks exactly.

        The robot advances only to an *exact* solution. A planner gated on
        reachability never commands a frame without one, so missing the pose
        gently the way the live env does (position kept, attitude spent) would
        only hide a gate that has failed — holding instead leaves it visible as
        the gripper coming off the forearm.
        """
        q_cmd = np.asarray(q_cmd, dtype=np.float64)
        target_pos, target_rot = self._gripper_pose(q_cmd)
        solution = self._ik_env.solve_robot_ik_exact(
            target_pos, target_rot.as_quat(), self._robot_q
        )
        if solution is not None:
            delta = np.clip(
                solution - self._robot_q, -self._max_joint_delta, self._max_joint_delta
            )
            self._robot_q = np.clip(self._robot_q + delta, self._lower, self._upper)
        self.robot_trajectory.append(self._robot_q.copy())
        return q_cmd

    def _gripper_pose(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """The pose the rigid grasp puts on the gripper for arm configuration ``q``."""
        forearm_pos, forearm_rot = forearm_frame_fk(
            self._fk_arm, q, self._preview_spine3_pos, self._preview_spine3_aa
        )
        return self._grasp.gripper_pose(
            _SMPL_TO_PB @ forearm_pos, Rotation.from_matrix(_SMPL_TO_PB) * forearm_rot
        )

    def visualize(self, path=None) -> np.ndarray:
        raise NotImplementedError("preview stand-in has nothing to render")

    def save_video(self, path, fps: int = 20) -> None:
        raise NotImplementedError("preview stand-in has nothing to render")
