"""Benchmark ABC and the persona x verbalizer x goal-sequence benchmark."""

from __future__ import annotations

import abc
from typing import Sequence

from evaluation.structs import InteractionTask
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.simulated_users import PERSONAS


class Benchmark(abc.ABC):
    """A distribution over interaction tasks, scored by hidden-bound oracles."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def generate_tasks(self, seed: int, cfg: MpcRunConfig) -> list[InteractionTask]:
        """Enumerate the tasks this benchmark evaluates."""


class InteractionBenchmark(Benchmark):
    """Personas x verbalizer abstraction levels x goal sequences.

    Goals resolve per persona: an explicit ``goals`` list wins; otherwise
    ``use_persona_goals`` takes the planner config's per-persona cartesian +
    transfer goals; otherwise the config's ``cartesian.goals``.
    """

    def __init__(
        self,
        name: str,
        personas: Sequence[str] | None = None,
        verbalizers: Sequence[str] = ("everyday",),
        goals: Sequence[Sequence[float]] | None = None,
        use_persona_goals: bool = False,
        max_goals: int = 1,
        max_rounds: int = 3,
        feedback_text: str | None = None,
    ) -> None:
        super().__init__(name)
        self.feedback_text = feedback_text
        self.personas = list(personas) if personas is not None else None
        self.verbalizers = list(verbalizers)
        self.goals = (
            [[float(value) for value in goal] for goal in goals]
            if goals is not None
            else None
        )
        self.use_persona_goals = use_persona_goals
        self.max_goals = max_goals
        self.max_rounds = max_rounds

    def _goal_sequence(
        self, persona: str, cfg: MpcRunConfig
    ) -> tuple[tuple[float, float, float], ...]:
        goals: list[Sequence[float]]
        if self.goals is not None:
            goals = list(self.goals)
        elif self.use_persona_goals and persona in cfg.persona_goals:
            persona_goals = cfg.persona_goals[persona]
            goals = [*persona_goals.cartesian, *persona_goals.transfer]
        else:
            assert cfg.cartesian is not None
            goals = [[float(v) for v in goal] for goal in cfg.cartesian.goals]
        goals = goals[: self.max_goals]
        return tuple((float(goal[0]), float(goal[1]), float(goal[2])) for goal in goals)

    def generate_tasks(self, seed: int, cfg: MpcRunConfig) -> list[InteractionTask]:
        names = self.personas
        if names is None:
            names = [name for name, user in PERSONAS.items() if user.bounds]
        tasks: list[InteractionTask] = []
        for persona in names:
            if not PERSONAS[persona].bounds:
                raise ValueError(f"Persona {persona!r} has no hidden bounds.")
            goal_seq = self._goal_sequence(persona, cfg)
            if not goal_seq:
                raise ValueError(f"No goals resolved for persona {persona!r}.")
            for verbalizer in self.verbalizers:
                tasks.append(
                    InteractionTask(
                        persona=persona,
                        verbalizer=verbalizer,
                        goals=goal_seq,
                        max_rounds=self.max_rounds,
                        seed=seed,
                        feedback_text=self.feedback_text,
                    )
                )
        return tasks
