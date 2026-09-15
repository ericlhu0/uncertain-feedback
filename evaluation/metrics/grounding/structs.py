from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GroundingResult:
    """Candidate motions for one utterance and the selected correction."""

    candidates: dict[int, np.ndarray]
    chosen_label: int
    magnitude: float
    correction_traj: np.ndarray
    # Raw generator draws behind the candidates and their cluster labels, when
    # the grounder samples; saved per round, then dropped from the record.
    samples: np.ndarray | None = None
    sample_labels: np.ndarray | None = None
