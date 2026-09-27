"""No-op grounder: the nominal plan is the only candidate.

Language is left entirely to cost generation — the round's correction is the
nominal continuation itself, so any behavioural change comes from replanning
with the newly generated costs.
"""

from __future__ import annotations

import numpy as np

from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory


class NominalGrounder(Grounder):
    """The utterance grounds to nothing; the nominal plan is passed through."""

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text
        nominal_aa = human.arm_aa_from_q(nominal_plan)
        candidates: dict[int, np.ndarray] = {0: nominal_aa}
        _, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=0,
            magnitude=magnitude,
            correction_traj=scale_trajectory(nominal_aa, magnitude),
        )
