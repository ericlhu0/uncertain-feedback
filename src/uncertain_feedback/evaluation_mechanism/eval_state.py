"""Serializable evaluation state for self-service cost feedback.

The ``agent`` (codex) backend authors the cost in its own subprocess, so it can't
share the live MPC objects the in-process backends use to roll out and score a
candidate. :class:`EvalState` bundles exactly the picklable inputs needed to
rebuild the rollout closure and the generated-cost context in a fresh process. The
generator pickles it next to the task;
``evaluation_mechanism/render_cost_comparison.py`` loads it to render the
rollout-vs-correction overlay codex inspects each turn.

The :class:`~uncertain_feedback.planners.mpc.human.Human` (with its
``SmplLeftArmFK``) and the comfort cost terms pickle, so the state round-trips. The full run config is
reduced to :class:`EvalMpcConfig` before serialization so persona metadata and
other non-operational settings never enter the agent workspace.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedPythonCost,
    build_generated_cost_context,
)
from uncertain_feedback.planners.mpc.goal_spaces import goal_point
from uncertain_feedback.planners.mpc.human import Human

if TYPE_CHECKING:
    from uncertain_feedback.planners.mpc.config import MpcRunConfig


@dataclass(frozen=True)
class EvalMpcConfig:  # pylint: disable=too-many-instance-attributes
    """Operational MPC fields needed to reproduce a candidate-cost rollout."""

    steps: int
    horizon: int
    n_mpc_samples: int
    max_angle_delta: float
    seed: int | None
    cartesian_goals: tuple[tuple[float, float, float], ...]
    cartesian_threshold: float
    # Constrained rollouts need the env's robot IK, which cannot ride a
    # pickle, so candidate costs are not scoreable for constrained configs.
    has_constraints: bool = False

    @classmethod
    def from_config(cls, cfg: MpcRunConfig | EvalMpcConfig) -> EvalMpcConfig:
        """Copy only rollout-relevant fields from a full MPC config."""
        if isinstance(cfg, EvalMpcConfig):
            return cfg
        return cls(
            steps=cfg.steps,
            horizon=cfg.horizon,
            n_mpc_samples=cfg.n_mpc_samples,
            max_angle_delta=cfg.max_angle_delta,
            seed=cfg.seed,
            cartesian_goals=tuple(
                (float(point[0]), float(point[1]), float(point[2]))
                for goal in (cfg.cartesian.goals if cfg.cartesian is not None else ())
                if (point := goal_point(goal)) is not None
            ),
            cartesian_threshold=(
                cfg.cartesian.threshold if cfg.cartesian is not None else 0.01
            ),
            has_constraints=bool(cfg.constraints),
        )


@dataclass
class EvalState:
    """Everything needed to roll out and score a candidate cost off-process.

    ``human`` is the person at the correction: its history is the executed
    motion and its ``q`` the configuration the candidate rollouts start from.
    """

    cfg: MpcRunConfig | EvalMpcConfig
    human: Human
    correction_traj: np.ndarray
    window: int
    base_extra_costs: CompositeTrajectoryCost
    reference_traj: np.ndarray | None = None
    full_correction_traj: np.ndarray | None = None
    cartesian_goal: np.ndarray | None = None
    cartesian_threshold: float | None = None
    rejected_trajs: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        self.cfg = EvalMpcConfig.from_config(self.cfg)

    def save(self, path: Path) -> None:
        """Pickle this state to ``path``."""
        self.cfg = EvalMpcConfig.from_config(self.cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "EvalState":
        """Load and sanitize a current or legacy :class:`EvalState` pickle."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        state.cfg = EvalMpcConfig.from_config(state.cfg)
        return state

    def make_generated_context(self) -> GeneratedCostContext:
        """Rebuild the runtime context passed to generated cost code."""
        return build_generated_cost_context(
            self.human,
            self.correction_traj,
            window=self.window,
            reference_traj=self.reference_traj,
            full_correction_traj=self.full_correction_traj,
            cartesian_goal=self.cartesian_goal,
            cartesian_threshold=self.cartesian_threshold,
            rejected_trajs=self.rejected_trajs,
        )

    def make_rollout_fn(
        self,
    ) -> Callable[[GeneratedPythonCost], np.ndarray | None]:
        """Rebuild the goal-seeking rollout closure used to score a candidate."""
        from uncertain_feedback.planners.mpc.goal_spaces import (  # pylint: disable=import-outside-toplevel
            CartesianConfig,
        )
        from uncertain_feedback.planners.mpc.mpc import (  # pylint: disable=import-outside-toplevel
            ArmMPC,
        )

        def rollout(cost: GeneratedPythonCost) -> np.ndarray | None:
            cfg = EvalMpcConfig.from_config(self.cfg)
            # Robot-action configs score candidates on the same human-space
            # kinematic rollout — the cost being evaluated is a human-arm cost.
            if cfg.has_constraints:
                return None
            if not cfg.cartesian_goals:
                return None
            extra_costs = CompositeTrajectoryCost(
                [*self.base_extra_costs.terms(), cost]
            )
            planner = ArmMPC(
                self.human.reset_human_with_q(self.human.q),
                horizon=cfg.horizon,
                n_mpc_samples=cfg.n_mpc_samples,
                max_angle_delta=cfg.max_angle_delta,
                visualize=False,
                extra_costs=extra_costs,
                seed=cfg.seed,
                cartesian=CartesianConfig(
                    goals=[list(goal) for goal in cfg.cartesian_goals],
                    threshold=cfg.cartesian_threshold,
                ),
            )
            for _ in range(max(1, cfg.steps)):
                try:
                    moved = planner.step()
                except RuntimeError:
                    break
                if planner.goal_reached(moved.q):
                    break
            return planner.human.arm_aa_from_q(planner.human.history)

        return rollout
