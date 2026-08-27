"""Shared planning rig: config, kinematics, optionally the motion generator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    MpcCostContext,
    build_extra_costs,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.simulated_users import SimulatedUser


@dataclass(frozen=True)
class PlanningRig:
    """The loaded planner config plus everything episodes plan against."""

    cfg: MpcRunConfig
    fk: SmplLeftArmFK
    context: MpcCostContext
    q0: np.ndarray
    spine3_pos: np.ndarray
    spine3_aa: np.ndarray
    body_pos: np.ndarray | None
    gen: MotionGenerator | None
    initial_hml_pose: np.ndarray | None


def build_rig(config_path: Path, *, seed: int, load_generator: bool) -> PlanningRig:
    """Load the planner config and derive the episode-planning context.

    ``load_generator=False`` skips the heavyweight motion-generator load for
    approaches that ground feedback without it; the start pose then comes from
    the config's ``arm:`` axis-angles instead of the MDM pose file. With the
    generator loaded, ``arm:`` still overrides the pose file's arm (the
    precedence :mod:`uncertain_feedback.planners.run` uses), so a run can pick a
    start arm configuration while keeping the pose's torso and body geometry.
    """
    cfg = replace(load_mpc_config(config_path), seed=seed)
    if load_generator:
        if cfg.pose is None:
            raise ValueError("MDM-grounded evaluation requires a pose: file.")
        gen: MotionGenerator | None = make_motion_generator(
            cfg.motion_generator, None, seed=seed
        )
        assert gen is not None
        loaded_pose = gen.load_pose(cfg.pose)
        initial_hml_pose: np.ndarray | None = loaded_pose
        arm_aa, body, spine3_aa_raw, collar_aa = gen.decode_pose(loaded_pose)
        body_pos: np.ndarray | None = np.asarray(body, dtype=np.float64)
        spine3_pos = np.asarray(body[9], dtype=np.float64)
        spine3_aa = np.asarray(spine3_aa_raw, dtype=np.float64)
        fk = SmplLeftArmFK()
        fk.collar_aa = np.asarray(collar_aa, dtype=np.float64)
        if cfg.arm is not None:
            arm_aa = cfg.arm
        q0 = fk.arm_aa_to_q(np.asarray(arm_aa, dtype=np.float64), spine3_aa)
    else:
        if cfg.arm is None:
            raise ValueError("Evaluation without the generator requires arm: angles.")
        gen = None
        initial_hml_pose = None
        body_pos = None
        fk = SmplLeftArmFK()
        spine3_pos = np.asarray(fk.tpose_spine3_pos, dtype=np.float64)
        spine3_aa = np.zeros(3, dtype=np.float64)
        q0 = fk.arm_aa_to_q(np.asarray(cfg.arm, dtype=np.float64), spine3_aa)
    context = MpcCostContext(
        fk=fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        time_of_day=cfg.simulated_user.time_of_day,
    )
    return PlanningRig(
        cfg=cfg,
        fk=fk,
        context=context,
        q0=q0,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        body_pos=body_pos,
        gen=gen,
        initial_hml_pose=initial_hml_pose,
    )


def base_extra_costs(rig: PlanningRig, user: SimulatedUser) -> CompositeTrajectoryCost:
    """Hand-authored comfort costs plus the persona's joint-box limits."""
    return CompositeTrajectoryCost(
        [*build_extra_costs(rig.cfg.costs, rig.context).terms(), user.limit_cost()]
    )


def cfg_with_goal(cfg: MpcRunConfig, goal: np.ndarray) -> MpcRunConfig:
    """The run config with its Cartesian goal queue replaced by ``goal``."""
    assert cfg.cartesian is not None
    return replace(cfg, cartesian=replace(cfg.cartesian, goals=[goal]))
