"""CostGen ABC: the LLM cost-generation axis, shared by every approach."""

from __future__ import annotations

import abc
from pathlib import Path

from evaluation.structs import LearnOutcome, RoundContext
from uncertain_feedback.cost_generation import (
    CostGenerationResult,
    CostRound,
    generate_cost_for_cluster,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
)
from uncertain_feedback.planners.rig import PlanningRig, cfg_with_goal

COST_GEN_SOURCES = ("chosen", "nominal")


class CostGen(abc.ABC):
    """Distills corrections into persistent planner costs, round after round.

    Subclasses decide what persists into planning; ``source`` anchors
    generation on the ``chosen`` correction motion or, for the language-only
    path, on the ``nominal`` plan.
    """

    def __init__(self, source: str = "chosen") -> None:
        if source not in COST_GEN_SOURCES:
            raise ValueError(f"source must be one of {COST_GEN_SOURCES}.")
        self.source = source
        self._rig: PlanningRig | None = None
        self._base = CompositeTrajectoryCost()
        self._episode_dir = Path(".")
        self._generated: list[GeneratedPythonCost] = []
        self._cost_rounds: list[CostRound] = []

    def reset(
        self, rig: PlanningRig, base: CompositeTrajectoryCost, episode_dir: Path
    ) -> None:
        """Bind the episode and drop all learned state."""
        self._rig = rig
        self._base = base
        self._episode_dir = episode_dir
        self._generated = []
        self._cost_rounds = []

    @abc.abstractmethod
    def learned_terms(self) -> list[GeneratedPythonCost]:
        """Whatever has been learned so far, as extra planning cost terms."""

    @abc.abstractmethod
    def learn(self, ctx: RoundContext) -> LearnOutcome:
        """Distill the resolved correction into persistent planner costs."""

    def _generate(
        self, ctx: RoundContext
    ) -> tuple[GeneratedPythonCost | None, CostGenerationResult]:
        """One round of cost generation; records the round on success."""
        rig = self._rig
        assert rig is not None, "reset() must run before use"
        language_only = self.source == "nominal"
        if language_only and ctx.nominal_plan is None:
            raise ValueError("source='nominal' requires RoundContext.nominal_plan.")
        generation = generate_cost_for_cluster(
            mpc=None,
            cfg=cfg_with_goal(rig.cfg, ctx.goal),
            instruction=ctx.utterance_text,
            cluster_traj=(
                ctx.nominal_plan if language_only else ctx.grounding.correction_traj
            ),
            current_q=ctx.q_feedback,
            q_history=ctx.q_history,
            context=rig.context,
            base_extra_costs=self._base,
            cost_dir=ctx.round_dir / "cost_generation",
            body_pos=rig.body_pos,
            spine3_pos=rig.spine3_pos,
            spine3_aa=rig.spine3_aa,
            candidate_trajs=None if language_only else ctx.grounding.candidates,
            highlight_label=None if language_only else ctx.grounding.chosen_label,
            undesirable_labels=frozenset() if language_only else ctx.rejected_labels,
            install=False,
            log_prefix="[evaluation]",
        )
        generated = generation.generated_cost
        if generated is None:
            return None, generation
        self._generated.append(generated)
        state_path = ctx.round_dir / "state.pkl"
        generation.eval_state.save(state_path)
        self._cost_rounds.append(
            CostRound(
                index=len(self._cost_rounds),
                goal=(float(ctx.goal[0]), float(ctx.goal[1]), float(ctx.goal[2])),
                feedback_text=ctx.utterance_text,
                trigger_step=len(ctx.q_history) - 1,
                round_dir=ctx.round_dir.resolve(),
                state_path=state_path.resolve(),
                cost_code=generated.code,
                params=generated.params,
                summaries=generation.summaries,
                image_paths=tuple(
                    path.resolve() for path in generation.images.values()
                ),
                description=generation.description,
                explanation=generation.explanation,
                interpretation=generation.interpretation,
                grounding=generation.grounding,
            )
        )
        return generated, generation
