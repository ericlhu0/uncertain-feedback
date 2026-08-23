"""No-op grounder: the nominal plan is the only candidate.

Language is left entirely to cost generation — the round's correction is the
nominal continuation itself, so any behavioural change comes from replanning
with the newly generated costs.
"""

from __future__ import annotations

import numpy as np

from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.structs import GroundingResult
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory


class NominalGrounder(Grounder):
    """The utterance grounds to nothing; the nominal plan is passed through."""

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
        goal: np.ndarray,
    ) -> GroundingResult:
        del text, q_feedback, goal
        nominal_aa = q_to_arm_aa(nominal_plan, self.rig.fk.elbow_hinge_axis)
        candidates: dict[int, np.ndarray] = {0: nominal_aa}
        _, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=0,
            magnitude=magnitude,
            correction_traj=scale_trajectory(nominal_aa, magnitude),
        )
