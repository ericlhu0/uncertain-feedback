"""Visual free-form verbalizer: a VLM speaks from rendered pose images.

Unlike the specificity-axis verbalizers, the visual verbalizer shows a
vision--language model the arm at the trigger pose and at the end of the
oracle correction window and asks for one free-form correction sentence in
the care recipient's voice. It never sees hidden bounds, joint names, or
numbers, and its responses are cached on disk per (episode, round) so reruns
are reproducible without API calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uncertain_feedback.llm.openai_model import OpenAIModel
from uncertain_feedback.planners.mpc.arm_features import arm_aa_from_state
from uncertain_feedback.planners.mpc.costs.base import MpcCostContext
from uncertain_feedback.simulated_users.attribution import (
    CorrectionIntent,
    has_feedback_content,
)
from uncertain_feedback.simulated_users.verbalizers import VERBALIZERS, Utterance
from uncertain_feedback.utils.plot import ArmVisualizer

PROMPT = (
    "You are role-playing a care recipient whose arm a caregiver robot is "
    "moving. The first image shows your arm right now. The second image shows "
    "how you want your arm to be (blue), with where it is now in orange. Say "
    "one short sentence to the caregiver, in your own words, correcting the "
    "motion."
)


class VisualVerbalizer:
    """VLM-backed verbalizer with a per-(episode, round) disk cache."""

    def __init__(self, model: OpenAIModel, cache_dir: Path) -> None:
        self._model = model
        self._cache_dir = cache_dir

    def _render_pose(
        self,
        path: Path,
        pose_aa: np.ndarray,
        current_aa: np.ndarray,
        context: MpcCostContext,
    ) -> None:
        # Duplicate the single frame so the pose lands on the dark, legible end
        # of the overlay's light-to-dark frame gradient.
        ArmVisualizer(context.fk).render_cluster_contrast_overlay(
            path,
            mdm_trajs={0: np.stack([pose_aa, pose_aa])},
            highlight_label=0,
            current_q=current_aa,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            include_others=False,
            include_reference=False,
        )

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
        """Return one cached-or-generated free-form correction sentence."""
        if not has_feedback_content(intent):
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"{episode_key}_round{round_index}.txt"
        if cache_path.exists():
            return Utterance(cache_path.read_text(encoding="utf-8").strip(), "visual")

        target_index = min(intent.join_index + window, oracle_path.shape[0] - 1)
        trigger_aa = arm_aa_from_state(q_trigger, context)
        target_aa = arm_aa_from_state(oracle_path[target_index], context)
        now_path = self._cache_dir / f"{episode_key}_round{round_index}_now.png"
        desired_path = self._cache_dir / f"{episode_key}_round{round_index}_desired.png"
        self._render_pose(now_path, trigger_aa, trigger_aa, context)
        self._render_pose(desired_path, target_aa, trigger_aa, context)

        text = self._model.get_full_output(
            PROMPT, image_input=[str(now_path), str(desired_path)]
        ).strip()
        cache_path.write_text(text, encoding="utf-8")
        return Utterance(text, "visual")


VERBALIZERS["visual"] = VisualVerbalizer
