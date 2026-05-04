from __future__ import annotations

import numpy as np
import pytest

from uncertain_feedback.planners.mpc.arm_mpc import SmplLeftArmMPC
from uncertain_feedback.planners.mpc.arm_mpc_cartesian import LeftArmMPCCartesian
from uncertain_feedback.planners.mpc.config import load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    ElbowHeightCost,
    MpcCostContext,
    build_extra_costs,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.planners.mpc.left_arm_cartesian_mpc_no_mdm import (
    LeftArmCartesianMPCNoMDM,
)


def _write_config(tmp_path, body: str):
    path = tmp_path / "mpc.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _base_yaml(extra: str = "") -> str:
    return f"""
planner: arm_mpc
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
{extra}
"""


def _cost_context(fk: SmplLeftArmFK) -> MpcCostContext:
    return MpcCostContext(
        fk=fk,
        spine3_pos=fk.tpose_spine3_pos,
        spine3_aa=np.zeros(3),
        fixed_collar_aa=np.zeros(3),
    )


class _FixedCost:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        assert q_trajs.shape[0] == self._values.shape[0]
        return self._values


def test_load_mpc_config_with_elbow_height(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  elbow_height:
    min: 0.1
    max: 0.45
    weight: 100
"""),
    )

    cfg = load_mpc_config(path)

    assert cfg.planner == "arm_mpc"
    assert cfg.steps == 2
    assert cfg.costs == {"elbow_height": {"min": 0.1, "max": 0.45, "weight": 100}}


def test_load_mpc_config_rejects_unknown_cost(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  shoulder_spin:
    min: 0
    max: 1
"""),
    )

    with pytest.raises(ValueError, match="Unknown MPC cost"):
        load_mpc_config(path)


def test_build_extra_costs_rejects_invalid_range(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        _base_yaml("""
costs:
  elbow_height:
    min: 0.5
    max: 0.1
"""),
    )

    cfg = load_mpc_config(path)
    fk = SmplLeftArmFK()

    with pytest.raises(ValueError, match="min must be less than max"):
        build_extra_costs(cfg.costs, _cost_context(fk))


def test_load_mpc_config_rejects_bad_cartesian_goal(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
planner: arm_mpc_cartesian
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
cartesian:
  goals:
    - [0.1, 0.2]
""",
    )

    with pytest.raises(ValueError, match="cartesian.goals"):
        load_mpc_config(path)


def test_load_mpc_config_accepts_no_mdm_cartesian_planner(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
planner: leftarmcartesianmpcnomdm
steps: 2
horizon: 3
n_mpc_samples: 4
max_angle_delta: 0.0025
cartesian:
  goals:
    - [0.1, 0.2, 0.3]
""",
    )

    cfg = load_mpc_config(path)

    assert cfg.planner == "leftarmcartesianmpcnomdm"
    assert cfg.cartesian.goals == [[0.1, 0.2, 0.3]]


def test_elbow_height_cost_zero_inside_range() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    context = _cost_context(fk)
    elbow_height = fk.fk_controlled(np.zeros((3, 3)))[3, 1] - context.spine3_pos[1]

    cost = ElbowHeightCost(
        min_height=elbow_height - 0.01,
        max_height=elbow_height + 0.01,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    np.testing.assert_allclose(cost(q_trajs), [0.0])


def test_elbow_height_cost_penalizes_outside_range() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((1, 2, 3, 3), dtype=np.float64)
    context = _cost_context(fk)
    elbow_height = fk.fk_controlled(np.zeros((3, 3)))[3, 1] - context.spine3_pos[1]

    cost = ElbowHeightCost(
        min_height=elbow_height + 0.1,
        max_height=elbow_height + 0.2,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    assert cost(q_trajs)[0] > 0.9


def test_elbow_height_cost_scores_entire_rollout_not_only_terminal() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    inside = np.zeros((3, 3), dtype=np.float64)
    high = np.zeros((3, 3), dtype=np.float64)
    high[0, 2] = 1.0
    elbow_height = fk.fk_controlled(inside)[3, 1] - context.spine3_pos[1]
    q_trajs = np.array(
        [
            [inside, high, inside],
            [inside, inside, inside],
        ],
        dtype=np.float64,
    )
    cost = ElbowHeightCost(
        min_height=elbow_height - 0.01,
        max_height=elbow_height + 0.01,
        weight=100.0,
        progress_weight=100.0,
        context=context,
    )

    costs = cost(q_trajs)

    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1], 0.0)


def test_elbow_height_progress_penalty_only_penalizes_getting_worse_outside() -> None:
    fk = SmplLeftArmFK()
    context = _cost_context(fk)
    low = np.zeros((3, 3), dtype=np.float64)
    low[0, 2] = -1.0
    lower = np.zeros((3, 3), dtype=np.float64)
    lower[0, 2] = -1.5
    less_low = np.zeros((3, 3), dtype=np.float64)
    less_low[0, 2] = -0.5
    q_trajs = np.array(
        [
            [low, lower],
            [low, less_low],
        ],
        dtype=np.float64,
    )
    cost = ElbowHeightCost(
        min_height=0.0,
        max_height=0.1,
        weight=0.0,
        progress_weight=100.0,
        context=context,
    )

    costs = cost(q_trajs)

    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1], 0.0)


def test_joint_space_mpc_adds_extra_costs() -> None:
    q_trajs = np.zeros((2, 2, 3, 3), dtype=np.float64)
    target_q = np.zeros((3, 3), dtype=np.float64)
    extra_costs = CompositeTrajectoryCost([_FixedCost([2.0, 3.0])])
    mpc = SmplLeftArmMPC(goals=[target_q], extra_costs=extra_costs)

    np.testing.assert_allclose(mpc._cost(q_trajs, target_q), [2.0, 3.0])


def test_cartesian_mpc_adds_extra_costs() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((2, 2, 3, 3), dtype=np.float64)
    wrist_rel = fk.fk_controlled(np.zeros((3, 3)))[-1] - fk.tpose_spine3_pos
    extra_costs = CompositeTrajectoryCost([_FixedCost([4.0, 5.0])])
    mpc = LeftArmMPCCartesian(
        cartesian_goals=[wrist_rel],
        initial_arm_aa=np.zeros((3, 3)),
        fk=fk,
        extra_costs=extra_costs,
    )

    np.testing.assert_allclose(mpc._cartesian_cost(q_trajs), [4.0, 5.0])


def test_no_mdm_cartesian_mpc_adds_extra_costs() -> None:
    fk = SmplLeftArmFK()
    q_trajs = np.zeros((2, 2, 3, 3), dtype=np.float64)
    wrist_rel = fk.fk_controlled(np.zeros((3, 3)))[-1] - fk.tpose_spine3_pos
    extra_costs = CompositeTrajectoryCost([_FixedCost([6.0, 7.0])])
    mpc = LeftArmCartesianMPCNoMDM(
        cartesian_goals=[wrist_rel],
        initial_arm_aa=np.zeros((3, 3)),
        fk=fk,
        extra_costs=extra_costs,
    )

    np.testing.assert_allclose(mpc._cartesian_cost(q_trajs), [6.0, 7.0])
