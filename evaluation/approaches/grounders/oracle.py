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
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.planners.mpc.rollout import rollout_to_goal
from uncertain_feedback.planners.rig import base_extra_costs, cfg_with_goal
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
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text
        assert self._goal is not None, "begin_goal() must run before use"
        rig = self.rig
        oracle_costs = CompositeTrajectoryCost(
            [
                *base_extra_costs(rig, self.user).terms(),
                HiddenCostTerm(user=self.user, context=rig.context),
            ]
        )
        window = rollout_to_goal(
            cfg_with_goal(rig.cfg, self._goal),
            np.asarray(q_feedback, dtype=np.float64),
            self._goal,
            rig.context,
            oracle_costs,
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            steps=len(nominal_plan) - 1,
            stop_at_goal=False,
            log_prefix="[oracle]",
        )
        oracle_aa = q_to_arm_aa(window, rig.fk.elbow_hinge_axis)
        candidates: dict[int, np.ndarray] = {0: oracle_aa}
        _, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=0,
            magnitude=magnitude,
            correction_traj=scale_trajectory(oracle_aa, magnitude),
        )
