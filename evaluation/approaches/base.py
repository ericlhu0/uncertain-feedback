"""Approach: a named composition of grounder x cost generation x steering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.approaches.cost_gen import CostGen, NoCostGen
from evaluation.approaches.grounders.base import ClusterSelector, Grounder
from evaluation.approaches.grounders.mdm import MdmGrounder
from evaluation.approaches.grounders.nominal import NominalGrounder
from evaluation.approaches.steering import NoSteering, Steering
from evaluation.structs import (
    GroundingResult,
    InteractionTask,
    LearnOutcome,
    RoundContext,
)
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.rig import PlanningRig, base_extra_costs
from uncertain_feedback.simulated_users import SimulatedUser


class Approach:
    """A system variant under evaluation: grounder x cost_gen x steering.

    ``steering`` selects how MDM sampling is steered toward the persona's
    bounds. It is only meaningful for the mdm grounder, which reads its mode
    from here rather than from the planner yaml's
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
        self.name = name
        self.grounder = grounder
        self.cost_gen = cost_gen
        self.steering = steering
        if isinstance(grounder, MdmGrounder):
            grounder.steering = steering
        self._base = CompositeTrajectoryCost()

    @property
    def requires_generator(self) -> bool:
        return self.grounder.requires_generator

    def reset(
        self,
        rig: PlanningRig,
        user: SimulatedUser,
        task: InteractionTask,
        episode_dir: Path,
    ) -> None:
        """Bind the episode and drop all learned state."""
        self._base = base_extra_costs(rig, user)
        self.grounder.reset(rig, user, task, episode_dir)
        self.cost_gen.reset(rig, self._base, episode_dir)

    def planning_costs(self) -> CompositeTrajectoryCost:
        """Base comfort costs plus whatever has been learned so far."""
        return CompositeTrajectoryCost(
            [*self._base.terms(), *self.cost_gen.learned_terms()]
        )

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
        goal: np.ndarray,
    ) -> GroundingResult:
        """Turn one utterance into candidate motions and a selected correction."""
        return self.grounder.ground(
            text, q_feedback, nominal_plan, cluster_selector, goal
        )

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        """Distill the resolved correction into persistent planner costs."""
        return self.cost_gen.learn(ctx)
