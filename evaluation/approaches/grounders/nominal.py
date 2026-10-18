"""No-op grounder: the nominal plan is the only candidate, and nothing runs.

Language is left entirely to cost generation. The persona still judges the
nominal plan (so the round records a verdict), but the arm does not replay the
motion it was just stopped from: the correction is the stop pose alone, so the
next motion is the replan with the newly generated costs.
"""

from __future__ import annotations

import numpy as np

from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.human import Human


def stop_pose(human: Human) -> np.ndarray:
    """A correction that moves nothing: the one-frame ``(1, 3, 3)`` current pose."""
    return human.arm_aa_from_q(human.q[None])


class NominalGrounder(Grounder):
    """The utterance grounds to nothing; the arm holds at the stop pose."""

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text
        candidates: dict[int, np.ndarray] = {0: human.arm_aa_from_q(nominal_plan)}
        _, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=0,
            magnitude=magnitude,
            correction_traj=stop_pose(human),
        )
