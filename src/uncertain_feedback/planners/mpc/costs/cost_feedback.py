"""Serializable evaluation state for self-service cost feedback.

The ``agent`` (codex) backend authors the cost in its own subprocess, so it can't
share the live MPC objects the in-process backends use to roll out and score a
candidate. :class:`EvalState` bundles exactly the picklable inputs needed to
rebuild the rollout closure and the generated-cost context in a fresh process. The
generator pickles it next to the task; ``experiments/render_cost_comparison.py``
loads it to render the rollout-vs-correction overlay codex inspects each turn.

``SmplLeftArmFK`` (carried via :class:`MpcCostContext`) and the comfort cost terms
are plain numpy / dataclasses, so the pickle round-trips.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.planners.mpc.costs.base import (
    CompositeTrajectoryCost,
    MpcCostContext,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedPythonCost,
    build_generated_cost_context,
)


@dataclass
class EvalState:
    """Everything needed to roll out and score a candidate cost off-process."""

    cfg: Any  # MpcRunConfig (imported lazily to avoid a config import cycle)
    current_q: np.ndarray
    correction_traj: np.ndarray
    q_history: list[np.ndarray]
    window: int
    cost_context: MpcCostContext
    base_extra_costs: CompositeTrajectoryCost
    body_pos: np.ndarray | None
    spine3_pos: np.ndarray | None
    spine3_aa: np.ndarray | None
    reference_traj: np.ndarray | None = None
    full_correction_traj: np.ndarray | None = None
    cartesian_goal: np.ndarray | None = None
    cartesian_threshold: float | None = None
    rejected_trajs: tuple[np.ndarray, ...] = ()

    def save(self, path: Path) -> None:
        """Pickle this state to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "EvalState":
        """Load a pickled :class:`EvalState` from ``path``."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def make_generated_context(self) -> GeneratedCostContext:
        """Rebuild the runtime context passed to generated cost code."""
        return build_generated_cost_context(
            self.cost_context,
            self.current_q,
            self.correction_traj,
            list(self.q_history),
            window=self.window,
            body_pos=self.body_pos,
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
        # Local import: run.py imports the costs package, so importing it at module
        # top would create a cycle.
        from uncertain_feedback.planners.run import (  # pylint: disable=import-outside-toplevel
            _make_cost_eval_rollout,
        )

        return _make_cost_eval_rollout(
            self.cfg,
            self.current_q,
            self.cost_context,
            self.base_extra_costs,
            self.body_pos,
            self.spine3_pos,
            self.spine3_aa,
        )
