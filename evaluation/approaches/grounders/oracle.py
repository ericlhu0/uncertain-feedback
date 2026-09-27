"""Oracle grounder: the persona's own ideal correction, no language involved.

Upper bound on grounding: the round's correction is a replan from the feedback
pose toward the goal under the base costs plus the persona's hidden cost, the
same costs that define the oracle path. Isolates cost learning from grounding
error.
"""

from __future__ import annotations

import numpy as np

from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.config import cfg_with_goal
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import rollout_to_goal
from uncertain_feedback.simulated_users import HiddenCostTerm
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory


class OracleGrounder(Grounder):
    """The utterance is ignored; the hidden-cost replan from the trigger is the candidate."""

    def __init__(self) -> None:
        super().__init__()
        self._goal: np.ndarray | None = None

    def begin_goal(self, goal: np.ndarray, oracle_path: np.ndarray) -> None:
        del oracle_path
        self._goal = np.asarray(goal, dtype=np.float64)

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text
        assert self._goal is not None, "begin_goal() must run before use"
        oracle_costs = CompositeTrajectoryCost(
            [
                *base_extra_costs(self.cfg.costs, human, self.user).terms(),
                HiddenCostTerm(user=self.user, human=human),
            ]
        )
        window = rollout_to_goal(
            cfg_with_goal(self.cfg, self._goal),
            human,
            self._goal,
            oracle_costs,
            steps=len(nominal_plan) - 1,
            stop_at_goal=False,
            log_prefix="[oracle]",
        )
        oracle_aa = human.arm_aa_from_q(window.history)
        candidates: dict[int, np.ndarray] = {0: oracle_aa}
        _, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=0,
            magnitude=magnitude,
            correction_traj=scale_trajectory(oracle_aa, magnitude),
        )
