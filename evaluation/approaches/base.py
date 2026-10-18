"""Approach: a named composition of grounder x cost generation x steering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.approaches.cost_gen import CostGen, NoCostGen
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.approaches.grounders.mdm import MdmGrounder
from evaluation.approaches.grounders.nominal import NominalGrounder
from evaluation.approaches.steering import LearnedSteering, NoSteering, Steering
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.goal_spaces import GoalRegion
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.simulated_users import SimulatedUser

# Where an episode leaves what the approach learned (see Approach.save).
LEARNED_STATE_FILE = "learned_state.json"


class Approach:
    """A system variant under evaluation: grounder x cost_gen x steering.

    ``steering`` selects how MDM sampling is steered: toward the persona's
    hidden bounds (``cg``, a ceiling) or the bounds of the learned costs in
    effect (``learned``, any cost learner). It is only meaningful for the mdm grounder, which reads its
    mode from here rather than from the planner yaml's
    ``feedback.uq.steering.mode``.
    """

    def __init__(
        self,
        name: str,
        grounder: Grounder,
        cost_gen: CostGen,
        steering: Steering | None = None,
    ) -> None:
        steering = steering if steering is not None else NoSteering()
        if steering.mode != "off" and not isinstance(grounder, MdmGrounder):
            raise ValueError("steering requires the mdm grounder.")
        if (
            isinstance(grounder, NominalGrounder)
            and not isinstance(cost_gen, NoCostGen)
            and cost_gen.source != "nominal"
        ):
            raise ValueError(
                "the nominal grounder selects no correction; "
                "cost_gen.source must be 'nominal'."
            )
        if isinstance(steering, LearnedSteering):
            steering.learned_bounds = cost_gen.learned_bounds
        self.name = name
        self.grounder = grounder
        self.cost_gen = cost_gen
        self.steering = steering
        if isinstance(grounder, MdmGrounder):
            grounder.steering = steering
        self._base = CompositeTrajectoryCost()
        self._resumed_from: Path | None = None

    @property
    def requires_generator(self) -> bool:
        """Whether this approach needs the motion generator loaded."""
        return self.grounder.requires_generator

    def reset(
        self,
        cfg: MpcRunConfig,
        human: Human,
        gen: MotionGenerator | None,
        user: SimulatedUser,
        seed: int,
        episode_dir: Path,
    ) -> None:
        """Bind the episode and drop all learned state."""
        self._base = base_extra_costs(cfg.costs, human, user)
        self.grounder.reset(cfg, gen, user, seed, episode_dir)
        self.cost_gen.reset(cfg, self._base, episode_dir)
        self._resumed_from = None

    def save(self, path: Path) -> None:
        """Write everything learned so far to ``path``, for :meth:`load`.

        That is the cost learner's rounds and costs and the grounder's
        conversation. The rounds' artifacts stay where they were written and
        are referenced relative to ``path``.
        """
        state = {
            "approach": self.name,
            "resumed_from": (
                None if self._resumed_from is None else str(self._resumed_from)
            ),
            "cost_gen": self.cost_gen.snapshot(path.parent),
            "grounder": self.grounder.memory(),
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Continue from what :meth:`save` wrote; call after :meth:`reset`."""
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["approach"] != self.name:
            raise ValueError(
                f"{path} holds what {state['approach']!r} learned, not {self.name!r}"
            )
        self.cost_gen.restore(state["cost_gen"], path.parent)
        self.grounder.restore(state["grounder"])
        self._resumed_from = path.resolve()

    def planning_costs(self) -> CompositeTrajectoryCost:
        """Base comfort costs plus whatever has been learned so far."""
        return CompositeTrajectoryCost(
            [*self._base.terms(), *self.cost_gen.learned_terms()]
        )

    def begin_goal(self, goal: GoalRegion, oracle_path: np.ndarray) -> None:
        """Announce the goal and its oracle path before the goal's rounds."""
        self.grounder.begin_goal(goal, oracle_path)

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        """Turn one utterance into candidate motions and a selected correction."""
        return self.grounder.ground(text, human, nominal_plan, cluster_selector)

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        """Distill the resolved correction into persistent planner costs."""
        return self.cost_gen.learn(ctx)

    def consolidate(self) -> None:
        """The slow update between attempts: distill everything learned so far."""
        self.cost_gen.consolidate()
