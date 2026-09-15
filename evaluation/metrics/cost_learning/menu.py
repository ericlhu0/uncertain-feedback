"""Menu violation: do the corrections proposed after learning respect the hidden bound?"""

from __future__ import annotations

from typing import Any

from evaluation.benchmarks.structs import MenuProbe
from evaluation.metrics.grounding.violation import correction_violation


def menu_rows(probe: MenuProbe) -> list[dict[str, Any]]:
    """One record per proposed candidate: its mean per-frame hidden-bound violation."""
    task = probe.task
    return [
        {
            "persona": task.persona,
            "verbalizer": task.verbalizer,
            "seed": task.seed,
            "approach": probe.approach,
            "goal_index": probe.goal_index,
            "utterance_text": probe.utterance.text,
            "learned_terms": probe.learned_terms,
            "label": label,
            "chosen": label == probe.chosen_label,
            "violation": correction_violation(probe.user, candidate, probe.context),
        }
        for label, candidate in probe.candidates.items()
    ]
