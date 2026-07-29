"""Tests for the robot-joint-action MPC planners.

Uses a kinematic robot-env double: a Panda ee chain with no physics, a grasp
measured once at the start, and a human arm that follows the ee exactly
through the same grasp-inverse + manifold projection the planner uses.
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
from uncertain_feedback.planners.mpc import (
    ArmMPC,
    CartesianConfig,
    FeedbackConfig,
    RobotActionsConfig,
)
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import (
    Q_DIM,
    SmplLeftArmFK,
    project_forearm_frames,
    q_to_arm_aa,
)

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


class _KinematicRobotEnv(ExecutionEnv):
    """Robot-action env double: ee moves kinematically, arm follows the grasp."""

    def __init__(self, fk: SmplLeftArmFK, q0: np.ndarray) -> None:
        super().__init__()
        self._fk_arm = fk
        self._chain, self._lower, self._upper = _panda_chain()
        self._robot_q = _PANDA_HOME.copy()
        # Measure the grasp against the canonical-branch representative of q0,
        # as the measuring envs do (their q comes from the positional decode);
        # a raw non-canonical q would bake a permanent pi roll into the grasp.
        q0 = fk.arm_aa_to_q(
            fk.arm_aa_from_positions(
                fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None), None
            ),
            None,
        )
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

    def current_grasp(self, q: np.ndarray) -> MeasuredGrasp:
        return self._grasp

    def execute_robot(self, target: np.ndarray) -> np.ndarray:
        self._robot_q = np.clip(target, self._lower, self._upper)
        return self._measured_q()

    def _measured_q(self) -> np.ndarray:
        ee_pos, ee_rot = self._chain.ee_pose(self._robot_q)
        forearm_rot = ee_rot @ self._grasp.rotation.inv().as_matrix()
        aa, _wrist, _res = project_forearm_frames(
            self._fk_arm,
            ee_pos @ _SMPL_TO_PB,
            _SMPL_TO_PB.T @ forearm_rot,
            self._grasp.position,
            np.zeros(Q_DIM),
        )
        hinge = self._fk_arm.elbow_hinge_axis
        return np.concatenate((aa[0], aa[1], [float(aa[2] @ hinge)]))

    def execute(self, q_cmd: np.ndarray) -> np.ndarray:
        raise AssertionError("robot-action planners must not call execute()")

    def visualize(self, path: Path | None = None) -> np.ndarray:
        raise NotImplementedError

    def save_video(self, path, fps: int = 20) -> None:
        raise NotImplementedError


class _RecordingCost:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        self.shapes.append(q_trajs.shape)
        return np.zeros(q_trajs.shape[0])


def _make_env_and_start(fk: SmplLeftArmFK) -> tuple[_KinematicRobotEnv, np.ndarray]:
    q_seed = np.zeros(Q_DIM)
    q_seed[6] = -0.8
    env = _KinematicRobotEnv(fk, q_seed)
    return env, env._measured_q()


def test_no_mdm_robot_planner_reaches_cartesian_goal() -> None:
    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    spine3 = fk.tpose_spine3_pos
    goal = (wrist0 - spine3) + np.array([0.0, 0.08, -0.05])

    recorder = _RecordingCost()
    mpc = ArmMPC(
        horizon=5,
        n_mpc_samples=128,
        fk=fk,
        extra_costs=CompositeTrajectoryCost([recorder]),
        seed=0,
        env=env,
        initial_q=q0,
        cartesian=CartesianConfig(goals=[goal]),
        # The grasp gate scales with the sampling std dev; match it here (the
        # default is tuned for the configs' 0.005).
        robot_actions=RobotActionsConfig(max_joint_delta=0.02, max_grasp_residual=0.05),
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
    assert all(shape == (128, 6, 3, 3) for shape in recorder.shapes)


def test_mdm_robot_planner_tracks_playback_then_reaches_goal() -> None:
    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    spine3 = fk.tpose_spine3_pos
    goal = (wrist0 - spine3) + np.array([0.0, 0.06, -0.04])

    mpc = ArmMPC(
        horizon=5,
        n_mpc_samples=256,
        fk=fk,
        seed=0,
        env=env,
        initial_q=q0,
        cartesian=CartesianConfig(goals=[goal]),
        feedback=FeedbackConfig(max_playback_delta=0.1),
        robot_actions=RobotActionsConfig(max_joint_delta=0.01, max_grasp_residual=0.03),
    )

    frames = np.tile(q0, (8, 1))
    frames[:, 6] = q0[6] + np.linspace(0.0, -0.3, 8)
    mpc.push_trajectory(frames)
    assert mpc._feedback is not None and mpc._feedback.in_playback()

    q = q0
    for _ in range(80):
        q = mpc.step(q)
        if mpc.mdm_ready_to_terminate:
            break
    assert mpc.mdm_ready_to_terminate
    assert abs(float(q[6]) - float(frames[-1, 6])) < 0.15

    for _ in range(80):
        q = mpc.step(q)
        if mpc.goal_reached(q):
            break
    wrist = fk.fk(q_to_arm_aa(q, fk.elbow_hinge_axis), None, None)[4]
    assert float(np.linalg.norm((wrist - spine3) - goal)) < 0.05


def test_robot_planner_yaml_configs_load() -> None:
    from uncertain_feedback.planners.mpc.config import load_mpc_config

    configs = Path("src/uncertain_feedback/planners/mpc/configs")
    for name in [
        "robot_real.yaml",
        "mdm_robot_real.yaml",
        "robot_mannequin_kinova.yaml",
    ]:
        cfg = load_mpc_config(configs / name)
        assert cfg.robot_actions is not None
        # The rig-tuned numbers churn; check the sampling geometry instead —
        # the std must sit well below the inf-norm cap or the uniform rescale
        # drowns the warm-started mean in noise.
        assert cfg.robot_actions.max_joint_delta > 0.0
        assert cfg.robot_actions.joint_delta_std is not None
        assert (
            0.0
            < cfg.robot_actions.joint_delta_std
            <= cfg.robot_actions.max_joint_delta / 2.0
        )
        assert cfg.robot_actions.infeasibility_weight > 0.0
        assert cfg.robot_actions.max_grasp_residual > 0.0
        assert cfg.robot_actions.grasp_residual_frames >= 1
        assert cfg.env in ("real", "sim_mannequin")


def test_robot_solve_discards_grasp_breaking_rollouts() -> None:
    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    goal = (wrist0 - fk.tpose_spine3_pos) + np.array([0.0, 0.08, -0.05])
    mpc = ArmMPC(
        horizon=5,
        n_mpc_samples=128,
        fk=fk,
        seed=0,
        env=env,
        initial_q=q0,
        cartesian=CartesianConfig(goals=[goal]),
        robot_actions=RobotActionsConfig(
            max_joint_delta=0.02, max_grasp_residual=0.02, grasp_residual_frames=3
        ),
    )
    grasp = env.current_grasp(q0)
    robot_q = env.current_robot_q()
    assert mpc._goal_space is not None
    stage = mpc._goal_space.stage_cost(mpc._extra_costs)
    batch, best = mpc._solve_sampling(q0, stage, mpc._actions)
    target = mpc._actions.command(batch, best)
    # The chosen first step keeps the grasp within the gate.
    step = np.stack([robot_q, target])
    ee_pos, ee_rot = env.robot_fk().ee_pose(step[np.newaxis])
    forearm_rot_pb = ee_rot @ grasp.rotation.inv().as_matrix()
    _aa, _w, residual = project_forearm_frames(
        fk,
        ee_pos @ _SMPL_TO_PB,
        _SMPL_TO_PB.T @ forearm_rot_pb,
        grasp.position,
        np.asarray(q0, dtype=np.float64),
        fk.tpose_spine3_pos,
        np.zeros(3),
    )
    assert float(residual[0, 1]) <= 0.02 + 1e-9

    # An impossible gate falls back to the least-violating sample, not a crash.
    mpc_strict = ArmMPC(
        horizon=5,
        n_mpc_samples=64,
        fk=fk,
        seed=0,
        env=env,
        initial_q=q0,
        cartesian=CartesianConfig(goals=[goal]),
        robot_actions=RobotActionsConfig(max_joint_delta=0.02, max_grasp_residual=0.0),
    )
    q = mpc_strict.step(q0)
    assert q.shape == (Q_DIM,)


def test_robot_plan_preview_env_rollout_is_consistent() -> None:
    from uncertain_feedback.envs.robot_preview import RobotPlanPreviewEnv

    fk = SmplLeftArmFK()
    env, q0 = _make_env_and_start(fk)
    preview_env = RobotPlanPreviewEnv(
        fk=fk,
        chain=env.robot_fk(),
        grasp=env.current_grasp(q0),
        robot_q=env.current_robot_q(),
        joint_limits=env.robot_joint_limits(),
        q_ref=q0,
        spine3_pos=None,
        spine3_aa=None,
        ik_env=env,
        max_joint_delta=0.02,
    )
    wrist0 = fk.fk(q_to_arm_aa(q0, fk.elbow_hinge_axis), None, None)[4]
    goal = (wrist0 - fk.tpose_spine3_pos) + np.array([0.0, 0.05, -0.03])
    mpc = ArmMPC(
        horizon=5,
        n_mpc_samples=64,
        fk=fk,
        seed=0,
        env=preview_env,
        initial_q=q0,
        cartesian=CartesianConfig(goals=[goal]),
        robot_actions=RobotActionsConfig(max_joint_delta=0.02, max_grasp_residual=0.05),
    )
    q = q0
    human = [q0]
    for _ in range(15):
        q = mpc.step(q)
        human.append(q)
    robot = np.asarray(preview_env.robot_trajectory)
    assert robot.shape == (len(human), 7)
    # Re-projecting each recorded robot configuration reproduces the human
    # trajectory frame — the previewed robot and arm agree by construction.
    for joints, q_frame in zip(robot[1:], human[1:]):
        preview_env._robot_q = np.asarray(joints)
        np.testing.assert_allclose(preview_env._measured_q(), q_frame, atol=1e-9)


def test_envs_first_import_order_has_no_cycle() -> None:
    """envs.grasp <-> planners.mpc must import cleanly from either side.

    The robot planners live in the planners.mpc package __init__ and use grasp
    machinery; importing an env module first (as run.py does via robot_preview)
    once hit a circular import. Run in a fresh interpreter so this test is not
    masked by modules pytest already loaded.
    """
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import uncertain_feedback.envs.robot_preview; "
            "import uncertain_feedback.planners.mpc",
        ],
        check=True,
    )
