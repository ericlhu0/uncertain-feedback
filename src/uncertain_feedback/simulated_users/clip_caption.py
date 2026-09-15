"""Clip-caption verbalizer: speaks the way the clip pipeline captions clips.

Renders the same single window image ``autolabel``'s Draft caption describes —
start pose, desired end pose, and the wrist and elbow traces between them
(:meth:`ArmVisualizer.render_correction_summary`) — and asks the stock
:data:`DRAFT_PROMPT` for one phrase, so evaluation utterances are drawn from
the same distribution as the correction-clip captions. Responses are cached on
disk per (episode, round) like the visual verbalizer's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uncertain_feedback.data_collection.dataset_auto_correction.captioning import (
    DRAFT_PROMPT,
    caption_model,
    draft_lines,
)
from uncertain_feedback.planners.mpc.arm_features import arm_aa_from_state
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users.attribution import (
    CorrectionIntent,
    has_feedback_content,
)
from uncertain_feedback.simulated_users.verbalizers import VERBALIZERS, Utterance
from uncertain_feedback.utils.plot import ArmVisualizer
from uncertain_feedback.utils.smpl_mesh import SmplMeshCache


class ClipCaptionVerbalizer:
    """Draft-caption verbalizer with a per-(episode, round) disk cache."""

    def __init__(self, model_name: str, cache_dir: Path, body_pos: np.ndarray) -> None:
        self._model = caption_model(model_name)
        self._cache_dir = cache_dir
        self._mesh = SmplMeshCache(np.asarray(body_pos, dtype=np.float64))

    def verbalize(
        self,
        intent: CorrectionIntent,
        q_trigger: np.ndarray,
        oracle_path: np.ndarray,
        context: MpcCostContext,
        episode_key: str,
        round_index: int,
        window: int = 20,
    ) -> Utterance | None:
        """Return one cached-or-generated clip-style caption of the desired window."""
        if not has_feedback_content(intent):
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"{episode_key}_round{round_index}.txt"
        if cache_path.exists():
            return Utterance(
                cache_path.read_text(encoding="utf-8").strip(), "clip_caption"
            )

        start = intent.join_index
        end = min(start + window, oracle_path.shape[0] - 1)
        window_aa = np.stack(
            [
                arm_aa_from_state(q_trigger, context),
                *(
                    arm_aa_from_state(state, context)
                    for state in oracle_path[start : end + 1]
                ),
            ]
        )
        image_path = self._cache_dir / f"{episode_key}_round{round_index}_window.png"
        ArmVisualizer(context.fk).render_correction_summary(
            image_path,
            arm_traj=window_aa,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            mesh=self._mesh,
        )
        text = self._model.get_full_output(DRAFT_PROMPT, image_input=[str(image_path)])
        lines = draft_lines(text, 1)
        if not lines:
            return None
        cache_path.write_text(lines[0], encoding="utf-8")
        return Utterance(lines[0], "clip_caption")


VERBALIZERS["clip_caption"] = ClipCaptionVerbalizer
