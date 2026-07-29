"""Human-arm-delta action space: Gaussian deltas composed on SO(3)."""

from __future__ import annotations

import numpy as np

from uncertain_feedback.envs.base import ExecutionEnv
from uncertain_feedback.planners.mpc.action_spaces.base import (
    ActionSpace,
    RolloutBatch,
    StageCost,
)
from uncertain_feedback.planners.mpc.kinematics import (
    Q_CLAVICLE,
    Q_DIM,
    SmplLeftArmFK,
    _compose_q,
    q_to_arm_aa,
)


class HumanArmActions(ActionSpace):
    """Samples deltas in the human 7-DOF arm state.

    Args:
        zero_first_sample: Force sample 0 to a zero-motion hold so a
            feasibility-constrained solve always has one feasible candidate.
    """

    def __init__(
        self,
        fk: SmplLeftArmFK,
        rng: np.random.Generator | None,
        n_samples: int,
        horizon: int,
        max_angle_delta: float,
        zero_first_sample: bool = False,
    ) -> None:
        self._fk = fk
        self._rng = rng
        self._n_samples = n_samples
        self._horizon = horizon
        self._max_angle_delta = max_angle_delta
        self._zero_first_sample = zero_first_sample

    def rollouts(
        self, env: ExecutionEnv, current_q: np.ndarray, mean: np.ndarray
    ) -> RolloutBatch:
        _ = env
        rng = self._rng if self._rng is not None else np.random
        actions = rng.normal(
            loc=mean,
            scale=self._max_angle_delta,
            size=(self._n_samples, self._horizon, Q_DIM),
        )
        # A robot holding the forearm cannot actuate the shoulder girdle, so
        # plans may only use the shoulder and elbow DOFs; the clavicle stays at
        # its (measured) current value.
        actions[..., Q_CLAVICLE] = 0.0
        if self._zero_first_sample:
            actions[0] = 0.0

        q_trajs = np.empty(
            (self._n_samples, self._horizon + 1, Q_DIM), dtype=np.float64
        )
        q_trajs[:, 0] = current_q[np.newaxis]
        for t in range(self._horizon):
            q_trajs[:, t + 1] = _compose_q(q_trajs[:, t], actions[:, t])

        aa_trajs = q_to_arm_aa(q_trajs, self._fk.elbow_hinge_axis)
        return RolloutBatch(actions=actions, aa_trajs=aa_trajs, q_trajs=q_trajs)

    def shape_costs(self, batch: RolloutBatch, stage_cost: StageCost) -> np.ndarray:
        return stage_cost(batch)

    def command(self, batch: RolloutBatch, best_idx: int) -> np.ndarray:
        return batch.actions[best_idx, 0]

    def execute(
        self, env: ExecutionEnv, current_q: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        return env.execute(_compose_q(current_q, command))

    def hold(self, env: ExecutionEnv, current_q: np.ndarray) -> np.ndarray:
        return env.hold(np.asarray(current_q, dtype=np.float64))
