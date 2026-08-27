"""Bind a task's verbalizer (feedback abstraction level) to episode state."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.structs import InteractionTask
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import MpcCostContext
from uncertain_feedback.simulated_users import (
    CorrectionIntent,
    Utterance,
    verbalize_everyday,
    verbalize_joint_resolved,
    verbalize_motion_directive,
    verbalize_vague,
)

# The verbalizer callable bound at episode setup: (intent, q_trigger, event_index).
BoundVerbalizer = Callable[[CorrectionIntent, np.ndarray, int], Utterance | None]


def bind_verbalizer(
    task: InteractionTask,
    cfg: MpcRunConfig,
    context: MpcCostContext,
    oracle_path: np.ndarray,
    episode_key: str,
    cache_dir: Path,
) -> BoundVerbalizer:
    """Bind the task's verbalizer to its episode state (rng, VLM, oracle)."""
    if task.verbalizer == "scripted":
        if task.feedback_text is None:
            raise ValueError("verbalizer: scripted needs the task's feedback_text.")
        spoken = Utterance(text=task.feedback_text, form="scripted")
        return lambda intent, q_trigger, event_index: spoken
    if task.verbalizer == "vague":
        return lambda intent, q_trigger, event_index: verbalize_vague(intent)
    if task.verbalizer == "joint_resolved":
        return lambda intent, q_trigger, event_index: verbalize_joint_resolved(intent)
    if task.verbalizer == "motion_directive":
        return lambda intent, q_trigger, event_index: verbalize_motion_directive(intent)
    if task.verbalizer == "everyday":
        rng = np.random.default_rng(task.seed)
        return lambda intent, q_trigger, event_index: verbalize_everyday(intent, rng)
    if task.verbalizer == "visual":
        # Imported lazily so non-visual episodes never touch the OpenAI client.
        from uncertain_feedback.llm.openai_model import (  # pylint: disable=import-outside-toplevel
            OpenAIModel,
        )
        from uncertain_feedback.simulated_users.visual import (  # pylint: disable=import-outside-toplevel
            VisualVerbalizer,
        )

        if cfg.llm_cost.model is None:
            raise ValueError("verbalizer: visual needs llm_cost.model.")
        visual = VisualVerbalizer(
            OpenAIModel(
                model=cfg.llm_cost.model,
                system_prompt="You answer with exactly one short spoken sentence.",
            ),
            cache_dir,
        )
        return lambda intent, q_trigger, event_index: visual.verbalize(
            intent,
            q_trigger,
            oracle_path,
            context,
            episode_key,
            event_index,
            window=cfg.simulated_user.nominal_steps,
        )
    raise ValueError(f"Unknown verbalizer {task.verbalizer!r}.")
