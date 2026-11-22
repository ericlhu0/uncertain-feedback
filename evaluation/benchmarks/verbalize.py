"""Bind a task's verbalizer (feedback abstraction level) to episode state."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.benchmarks.structs import InteractionTask
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import (
    CorrectionIntent,
    Utterance,
    verbalize_everyday,
    verbalize_joint_resolved,
    verbalize_motion_directive,
    verbalize_vague,
)

# The verbalizer callable bound at episode setup: (intent, q_trigger,
# event_index, route), ``route`` the oracle path ``intent`` was attributed on.
BoundVerbalizer = Callable[
    [CorrectionIntent, np.ndarray, int, np.ndarray], Utterance | None
]


def bind_verbalizer(
    task: InteractionTask,
    cfg: MpcRunConfig,
    human: Human,
    episode_key: str,
    cache_dir: Path,
) -> BoundVerbalizer:
    """Bind the task's verbalizer to its episode state (rng, VLM)."""
    if task.verbalizer == "scripted":
        if task.feedback_text is None:
            raise ValueError("verbalizer: scripted needs the task's feedback_text.")
        spoken = Utterance(text=task.feedback_text, form="scripted")
        return lambda intent, q_trigger, event_index, route: spoken
    if task.verbalizer == "vague":
        return lambda intent, q_trigger, event_index, route: verbalize_vague(intent)
    if task.verbalizer == "joint_resolved":
        return lambda intent, q_trigger, event_index, route: verbalize_joint_resolved(
            intent
        )
    if task.verbalizer == "motion_directive":
        return lambda intent, q_trigger, event_index, route: verbalize_motion_directive(
            intent
        )
    if task.verbalizer == "everyday":
        rng = np.random.default_rng(task.seed)
        return lambda intent, q_trigger, event_index, route: verbalize_everyday(
            intent, rng
        )
    if task.verbalizer == "visual":
        # Imported lazily so non-visual episodes never touch the OpenAI client.
        from uncertain_feedback.llm import (  # pylint: disable=import-outside-toplevel
            make_model,
        )
        from uncertain_feedback.simulated_users.visual import (  # pylint: disable=import-outside-toplevel
            VisualVerbalizer,
        )

        if cfg.llm_cost.model is None:
            raise ValueError("verbalizer: visual needs llm_cost.model.")
        visual = VisualVerbalizer(
            make_model(
                model=cfg.llm_cost.model,
                system_prompt="You answer with exactly one short spoken sentence.",
            ),
            cache_dir,
        )
        return lambda intent, q_trigger, event_index, route: visual.verbalize(
            intent,
            q_trigger,
            route,
            human,
            episode_key,
            event_index,
            window=cfg.simulated_user.nominal_steps,
        )
    if task.verbalizer == "clip_caption":
        # Imported lazily so non-caption episodes never touch the OpenAI client.
        from uncertain_feedback.simulated_users.clip_caption import (  # pylint: disable=import-outside-toplevel
            ClipCaptionVerbalizer,
        )

        if cfg.llm_cost.model is None:
            raise ValueError("verbalizer: clip_caption needs llm_cost.model.")
        clip_caption = ClipCaptionVerbalizer(
            cfg.llm_cost.model, cache_dir, human.posture
        )
        return lambda intent, q_trigger, event_index, route: clip_caption.verbalize(
            intent,
            q_trigger,
            route,
            human,
            episode_key,
            event_index,
            window=cfg.simulated_user.nominal_steps,
        )
    raise ValueError(f"Unknown verbalizer {task.verbalizer!r}.")
