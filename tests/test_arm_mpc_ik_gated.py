"""Tests for the IK-gated human-action Cartesian MPC.

Uses a kinematic env double: a Panda ee chain with no physics, a grasp
measured once at the start, a human arm that executes commands exactly, and a
robot that IK-tracks the grasp after each step (so the planner's next solve
warm-starts from a consistent robot state).
"""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.envs.grasp import MeasuredGrasp, forearm_frame_fk
from uncertain_feedback.envs.robot_fk import RobotChainFK
from uncertain_feedback.envs.sim_mannequin import _PANDA_URDF, _SMPL_TO_PB
from uncertain_feedback.planners.mpc.arm_mpc_ik_gated import (
    ArmMPCCartesianNoMDMIKGated,
)
from uncertain_feedback.planners.mpc.kinematics import Q_DIM, SmplLeftArmFK, q_to_arm_aa

_PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)


def _panda_chain() -> tuple[RobotChainFK, np.ndarray, np.ndarray]:
    cid = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(_PANDA_URDF), useFixedBase=True, physicsClientId=cid)
        ee_index = next(
            j
            for j in range(p.getNumJoints(body, physicsClientId=cid))
            if p.getJointInfo(body, j, physicsClientId=cid)[12] == b"ee_link"
        )
        movable = [
            j
            for j in range(p.getNumJoints(body, physicsClientId=cid))
            if p.getJointInfo(body, j, physicsClientId=cid)[2] != p.JOINT_FIXED
        ]
        lower = np.array(
            [p.getJointInfo(body, j, physicsClientId=cid)[8] for j in movable]
        )
        upper = np.array(
            [p.getJointInfo(body, j, physicsClientId=cid)[9] for j in movable]
        )
        unbounded = lower > upper
        lower = np.where(unbounded, -np.inf, lower)
        upper = np.where(unbounded, np.inf, upper)
        return RobotChainFK.from_pybullet(body, ee_index, cid), lower, upper
    finally:
        p.disconnect(cid)


class _KinematicGraspEnv(ExecutionEnv):
    """Human-action env double: arm moves kinematically, robot follows the grasp."""

    def __init__(self, fk: SmplLeftArmFK, q0: np.ndarray) -> None:
        super().__init__()
        self._fk_arm = fk
        self._chain, self._lower, self._upper = _panda_chain()
        self._robot_q = _PANDA_HOME.copy()
        self.batch_calls = 0
        self.track_calls = 0
        forearm_pos, forearm_rot = forearm_frame_fk(fk, q0, None, None)
        ee_pos, ee_rot = self._chain.ee_pose(self._robot_q)
        self._grasp = MeasuredGrasp.measure(
            _SMPL_TO_PB @ forearm_pos,
            Rotation.from_matrix(_SMPL_TO_PB) * forearm_rot,
            ee_pos,
            Rotation.from_matrix(ee_rot),
        )

    def robot_fk(self) -> RobotChainFK:
        return self._chain

    def current_robot_q(self) -> np.ndarray:
        return self._robot_q.copy()

    def robot_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

    def solve_robot_ik_exact(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> np.ndarray | None:
        robot_q = np.asarray(q_seed, dtype=np.float64).copy()
        target_rot = Rotation.from_quat(target_quat).as_matrix()
        for _ in range(20):
            pos, rot, jac = self._chain.ee_pose_jacobian(robot_q)
            err = np.concatenate(
                [target_pos - pos, Rotation.from_matrix(target_rot @ rot.T).as_rotvec()]
            )
            delta = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(7), jac.T @ err)
            robot_q = np.clip(robot_q + delta, self._lower, self._upper)
        return robot_q

    def solve_robot_ik_exact_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.batch_calls += 1
        return super().solve_robot_ik_exact_batch(target_pos, target_quat, q_seed)

    def track_robot_ik_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.track_calls += 1
        return super().track_robot_ik_batch(target_pos, target_quat, q_seed)

    def current_grasp(self, q: np.ndarray) -> MeasuredGrasp:
        return self._grasp

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        q = np.asarray(q_cmd, dtype=np.float64)
        forearm_pos, forearm_rot = forearm_frame_fk(self._fk_arm, q, None, None)
        target_pos, target_rot = self._grasp.gripper_pose(
            _SMPL_TO_PB @ forearm_pos, Rotation.from_matrix(_SMPL_TO_PB) * forearm_rot
        )
        target_mat = target_rot.as_matrix()
        for _ in range(20):
            pos, rot, jac = self._chain.ee_pose_jacobian(self._robot_q)
            err = np.concatenate(
                [target_pos - pos, Rotation.from_matrix(target_mat @ rot.T).as_rotvec()]
            )
            delta = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(7), jac.T @ err)
            self._robot_q = np.clip(self._robot_q + delta, self._lower, self._upper)
        return q

    def visualize(self, path: Path | None = None) -> np.ndarray:
        raise NotImplementedError

    def save_video(self, path, fps: int = 20) -> None:
        raise NotImplementedError


def _make_env_and_start(fk: SmplLeftArmFK) -> tuple[_KinematicGraspEnv, np.ndarray]:
    q0 = np.zeros(Q_DIM)
    q0[6] = -0.8
    return _KinematicGraspEnv(fk, q0), q0


def test_ik_gated_planner_reaches_cartesian_goal() -> None:
    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    spine3 = fk.tpose_spine3_pos
    goal = (wrist0 - spine3) + np.array([0.0, 0.08, -0.05])

    mpc = ArmMPCCartesianNoMDMIKGated(
        cartesian_goals=[goal],
        initial_q=q0,
        horizon=5,
        n_mpc_samples=128,
        max_angle_delta=0.02,
        fk=fk,
        seed=0,
        env=env,
    )
    dist0 = float(np.linalg.norm((wrist0 - spine3) - goal))
    q = q0
    for _ in range(60):
        q = mpc.step(q)
        if mpc.goal_reached(q):
            break
    wrist = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), None, None)[4]
    dist = float(np.linalg.norm((wrist - spine3) - goal))
    assert dist < 0.05 < dist0


def test_ik_gated_solve_keeps_first_steps_reachable() -> None:
    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    goal = (wrist0 - fk.tpose_spine3_pos) + np.array([0.0, 0.08, -0.05])
    mpc = ArmMPCCartesianNoMDMIKGated(
        cartesian_goals=[goal],
        initial_q=q0,
        horizon=5,
        n_mpc_samples=64,
        max_angle_delta=0.02,
        fk=fk,
        seed=0,
        env=env,
    )
    _first, plan = mpc.solve(q0)
    best_traj = mpc._rollout(np.asarray(q0, dtype=np.float64), plan[np.newaxis])
    residual = float(mpc._grasp_ik_residuals(best_traj)[0])
    assert residual <= mpc._max_grasp_ik_residual + 1e-9
    # The gate goes through the continuation path, never execution's enumerating
    # solve; this double has no such split, so track forwards to the batch.
    assert env.track_calls == 2 * mpc._grasp_ik_frames
    assert env.batch_calls == env.track_calls


def test_ik_gated_pinned_robot_rejects_all_motion() -> None:
    """With the robot pinned, the explicit hold is the only feasible sample."""
    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    env._lower = env._robot_q.copy()
    env._upper = env._robot_q.copy()
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    goal = (wrist0 - fk.tpose_spine3_pos) + np.array([0.0, 0.08, -0.05])
    mpc = ArmMPCCartesianNoMDMIKGated(
        cartesian_goals=[goal],
        initial_q=q0,
        horizon=5,
        n_mpc_samples=64,
        max_angle_delta=0.02,
        fk=fk,
        seed=0,
        env=env,
    )
    actions = mpc._sample_actions(
        np.zeros((mpc._horizon, Q_DIM)), (64, mpc._horizon, Q_DIM)
    )
    q_trajs = mpc._rollout(np.asarray(q0, dtype=np.float64), actions)
    residuals = mpc._grasp_ik_residuals(q_trajs)
    assert residuals[0] <= mpc._max_grasp_ik_residual
    assert np.all(residuals[1:] > mpc._max_grasp_ik_residual)
    costs = mpc._cartesian_cost(q_trajs)
    assert np.isfinite(costs[0])
    assert np.all(np.isinf(costs[1:]))
    q = mpc.step(q0)
    np.testing.assert_allclose(q, q0, atol=1e-12)


def _preview_env_for(fk: SmplLeftArmFK, env: _KinematicGraspEnv, q0: np.ndarray):
    from uncertain_feedback.envs.robot_preview import RobotPlanPreviewEnv

    return RobotPlanPreviewEnv(
        fk=fk,
        chain=env.robot_fk(),
        grasp=env.current_grasp(q0),
        robot_q=env.current_robot_q(),
        joint_limits=env.robot_joint_limits(),
        q_ref=q0,
        spine3_pos=None,
        spine3_aa=None,
        ik_env=env,
        max_joint_delta=0.05,
    )


def test_preview_stand_in_gates_a_human_action_rollout() -> None:
    """The offline preview stand-in enforces the gate the live run will.

    It used to refuse ``execute`` outright, so an IK-gated run was previewed
    with the ungated planner — the drawn plan walked into poses the run itself
    discards, and the grasp diverged at the end of the preview. With the robot
    pinned, nothing but the hold is reachable: the gated rollout must stand
    still where the ungated one drives the gripper off the forearm.
    """
    from uncertain_feedback.planners.mpc.arm_mpc_cartesian_no_mdm import (
        ArmMPCCartesianNoMDM,
    )

    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    env._lower = env._robot_q.copy()
    env._upper = env._robot_q.copy()
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    goal = (wrist0 - fk.tpose_spine3_pos) + np.array([0.0, 0.08, -0.05])
    kwargs = dict(
        cartesian_goals=[goal],
        initial_q=q0,
        horizon=5,
        n_mpc_samples=64,
        max_angle_delta=0.02,
        fk=fk,
        seed=0,
    )

    gated_env = _preview_env_for(fk, env, q0)
    gated = ArmMPCCartesianNoMDMIKGated(**kwargs, env=gated_env)
    q = q0
    for _ in range(10):
        q = gated.step(q)
    np.testing.assert_allclose(q, q0, atol=1e-12)
    assert len(gated_env.robot_trajectory) == 11

    ungated_env = _preview_env_for(fk, env, q0)
    ungated = ArmMPCCartesianNoMDM(**kwargs, env=ungated_env)
    q_ungated = q0
    for _ in range(10):
        q_ungated = ungated.step(q_ungated)

    def grasp_gap(preview_env, q_frame: np.ndarray) -> float:
        target_pos, _ = preview_env._gripper_pose(q_frame)
        ee_pos, _ = preview_env.robot_fk().ee_pose(preview_env.current_robot_q())
        return float(np.linalg.norm(target_pos - ee_pos))

    assert grasp_gap(gated_env, q) < 1e-9 < 0.01 < grasp_gap(ungated_env, q_ungated)


def test_preview_rollout_table_covers_every_cartesian_planner() -> None:
    """Every previewable planner picks its rollout; none falls through ungated."""
    from uncertain_feedback.planners import run as run_module

    assert set(run_module._PREVIEW_ROLLOUTS) == set(run_module._CARTESIAN_PLANNERS)
    assert (
        run_module._PREVIEW_ROLLOUTS["arm_mpc_cartesian_no_mdm_ik_gated"]
        is run_module._rollout_gated_reference_trajectory
    )


def test_continuation_refuses_the_branch_change_enumeration_would_take() -> None:
    """The gate's solver must not answer with a pose only a branch change reaches.

    Such a solution is exact, so the residual check cannot see anything wrong
    with it, but it sits far enough away that the arm spends tens of steps at
    ``robot_max_joint_delta`` getting there with the grasp wrong throughout.
    Execution's enumerating solve offers exactly that; continuation refuses it.
    """
    from ssik.prebuilt import gen3_ik

    from uncertain_feedback.envs.real import (
        _IK_TRACK_MAX_DIST,
        _gen3_seeded_track_batch,
    )

    seed = np.array([0.0, 0.26, 0.0, -2.27, 0.0, 0.96, 1.57])
    far = seed.copy()
    far[0] += 1.5
    target = gen3_ik.fk(far)
    lower, upper = np.full(7, -3.05), np.full(7, 3.05)

    _solutions, feasible = _gen3_seeded_track_batch(
        target[np.newaxis],
        seed[np.newaxis],
        lower,
        upper,
        np.zeros(7, dtype=bool),
    )
    assert not feasible[0]

    # The pose is reachable — enumeration finds it — but only off-branch, which
    # is why gating on the enumerating solve passed rollouts the arm could not
    # follow.
    in_box = [
        np.asarray(s.q)
        for s in gen3_ik.solve(
            target, q_seed=seed, max_solutions=None, respect_limits=False
        )
        if np.all(np.asarray(s.q) >= lower) and np.all(np.asarray(s.q) <= upper)
    ]
    assert in_box
    assert min(np.max(np.abs(q - seed)) for q in in_box) > _IK_TRACK_MAX_DIST


def test_ik_gated_yaml_config_loads() -> None:
    from uncertain_feedback.planners.mpc.config import load_mpc_config

    cfg = load_mpc_config(
        Path("src/uncertain_feedback/planners/mpc/configs")
        / "arm_mpc_cartesian_no_mdm_ik_gated_real.yaml"
    )
    assert cfg.planner == "arm_mpc_cartesian_no_mdm_ik_gated"
    assert cfg.max_grasp_ik_residual > 0.0
    assert cfg.grasp_residual_frames >= 1
    assert cfg.env in ("real", "sim_mannequin")
