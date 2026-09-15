"""Score one case's candidate corrections with every grounding metric."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.metrics.grounding.expressivity import (
    candidate_diversity,
    candidate_position_diversity,
)
from evaluation.metrics.grounding.progress import correction_progress
from evaluation.metrics.grounding.violation import correction_violation
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users import SimulatedUser


def candidate_row(
    user: SimulatedUser,
    oracle_correction: np.ndarray,
    candidate: np.ndarray,
    context: MpcCostContext,
) -> dict[str, Any]:
    """Violation and progress of one trajectory against the case's oracle."""
    progress = correction_progress(oracle_correction, candidate, context)
    return {
        "violation": correction_violation(user, candidate, context),
        "arc_progress": progress.arc_progress,
        "alignment": progress.alignment,
        "oracle_path_length": progress.oracle_path_length,
        "oracle_displacement": progress.oracle_displacement,
        **{f"progress_{name}": v for name, v in progress.per_feature.items()},
    }


def case_row(candidates: dict[int, np.ndarray], context: MpcCostContext) -> dict[str, Any]:
    """Spread of the case's candidate menu."""
    diversity = candidate_diversity(candidates, context)
    position = candidate_position_diversity(candidates, context)
    return {
        "n_candidates": len(candidates),
        "diversity": diversity.diversity,
        "rms_displacement": diversity.rms_displacement,
        "position_diversity": position.diversity,
        "position_rms_displacement": position.rms_displacement,
    }
