"""Predefined-parameterized-edit grounding baseline (no text-to-motion model)."""

from __future__ import annotations

import numpy as np

from evaluation.approaches.base import Approach, ClusterSelector
from evaluation.structs import GroundingResult
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_SHOULDER_ROW = 0
_ELBOW_ROW = 1


class ParameterizedEditApproach(Approach):
    """Candidates are fixed parametric edits of the nominal continuation.

    Stands in for prior systems that modify trajectories through predefined
    edits over predefined parameters: the utterance itself is ignored, and the
    candidate set is the nominal plan plus a ramped axis-angle offset per
    (joint row, axis, sign). Corrections outside this family are unreachable
    by construction.
    """

    requires_generator = False

    def __init__(
        self,
        name: str = "edit_baseline",
        learning: str = "immediate",
        edit_delta: float = 0.4,
    ) -> None:
        super().__init__(name=name, learning=learning)
        self._edit_delta = edit_delta

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text, q_feedback
        nominal_aa = q_to_arm_aa(nominal_plan, self.rig.fk.elbow_hinge_axis)
        ramp = np.linspace(0.0, 1.0, nominal_aa.shape[0])
        candidates: dict[int, np.ndarray] = {0: nominal_aa}
        label = 1
        for row in (_SHOULDER_ROW, _ELBOW_ROW):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    edited = nominal_aa.copy()
                    edited[:, row, axis] += sign * self._edit_delta * ramp
                    candidates[label] = edited
                    label += 1
        chosen_label, magnitude = cluster_selector(candidates)
        correction = scale_trajectory(candidates[chosen_label], magnitude)
        return GroundingResult(
            candidates=candidates,
            chosen_label=chosen_label,
            magnitude=magnitude,
            correction_traj=correction,
        )
