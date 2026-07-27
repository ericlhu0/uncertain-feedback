"""Kinematic stand-in env for previewing a robot-action plan offline.

Snapshots the real env's robot chain, measured grasp, joint state, and limits,
then lets a robot-action planner roll out against them without commanding
anything: ``execute_robot`` just stores the joint target and reports the human
configuration the grasp projection implies — the same forward model the
planner samples with, so the previewed robot and arm stay consistent by
construction. The executed joint targets are kept in
:attr:`robot_trajectory` so the preview can animate the actual planned robot
motion instead of chasing the arm through IK.
"""

from __future__ import annotations

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.grasp import MeasuredGrasp
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
        self.robot_trajectory: list[np.ndarray] = [self._robot_q.copy()]

    def robot_fk(self) -> RobotChainFK:
        return self._chain

    def current_robot_q(self) -> np.ndarray:
        return self._robot_q.copy()

    def robot_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

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
        raise NotImplementedError("preview stand-in only executes robot actions")

    def visualize(self, path=None) -> np.ndarray:
        raise NotImplementedError("preview stand-in has nothing to render")

    def save_video(self, path, fps: int = 20) -> None:
        raise NotImplementedError("preview stand-in has nothing to render")
