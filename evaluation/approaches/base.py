"""Approach ABC: grounds feedback into motion and learns persistent costs."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.rig import EvalRig, base_extra_costs, cfg_with_goal
from evaluation.structs import (
    GroundingResult,
    InteractionTask,
    LearnOutcome,
    RoundContext,
)
from uncertain_feedback.cost_generation import (
    CombineCostGenerator,
    CostGenerationResult,
    CostRound,
    generate_cost_for_cluster,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
)
from uncertain_feedback.simulated_users import SimulatedUser

ClusterSelector = Callable[[dict[int, np.ndarray]], tuple[int, float]]

LEARNING_MODES = ("none", "immediate", "lifelong")


class Approach(abc.ABC):
    """A system variant under evaluation.

    The preference-learning update is shared here (immediate cost generation
    plus lifelong combination); subclasses supply the grounding mechanism.
    ``learning`` selects what persists into planning: ``none`` (execute
    corrections but learn nothing), ``immediate`` (stack every per-round
    cost), or ``lifelong`` (one unified replacement cost via combination).
    """

    requires_generator: bool = True

    def __init__(self, name: str, learning: str = "lifelong") -> None:
        if learning not in LEARNING_MODES:
            raise ValueError(f"learning must be one of {LEARNING_MODES}.")
        self.name = name
        self.learning = learning
        # "chosen" anchors cost generation on the selected correction motion;
        # "nominal" is the language-only ablation (utterance + nominal plan).
        self.learn_from = "chosen"
        self._rig: EvalRig | None = None
        self._user: SimulatedUser | None = None
        self._episode_dir = Path(".")
        self._base = CompositeTrajectoryCost()
        self._generated: list[GeneratedPythonCost] = []
        self._unified: GeneratedPythonCost | None = None
        self._cost_rounds: list[CostRound] = []

    @property
    def rig(self) -> EvalRig:
        """The bound rig; valid after :meth:`reset`."""
        assert self._rig is not None, "reset() must run before use"
        return self._rig

    @property
    def user(self) -> SimulatedUser:
        """The bound persona; valid after :meth:`reset`."""
        assert self._user is not None, "reset() must run before use"
        return self._user

    def reset(
        self,
        rig: EvalRig,
        user: SimulatedUser,
        task: InteractionTask,
        episode_dir: Path,
    ) -> None:
        """Bind the episode and drop all learned state."""
        self._rig = rig
        self._user = user
        self._episode_dir = episode_dir
        self._base = base_extra_costs(rig, user)
        self._generated = []
        self._unified = None
        self._cost_rounds = []
        self._reset_grounding(task)

    def _reset_grounding(self, task: InteractionTask) -> None:
        """Hook for grounding-specific per-episode state."""

    def planning_costs(self) -> CompositeTrajectoryCost:
        """Base comfort costs plus whatever has been learned so far."""
        terms = list(self._base.terms())
        if self.learning == "immediate":
            terms.extend(self._generated)
        elif self.learning == "lifelong" and self._unified is not None:
            terms.append(self._unified)
        return CompositeTrajectoryCost(terms)

    @abc.abstractmethod
    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        """Turn one utterance into candidate motions and a selected correction."""

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        """Distill the resolved correction into persistent planner costs."""
        if self.learning == "none":
            return LearnOutcome(cost_accepted=False, unified_installed=False)
        rig = self.rig
        language_only = self.learn_from == "nominal"
        if language_only and ctx.nominal_plan is None:
            raise ValueError("learn_from='nominal' requires RoundContext.nominal_plan.")
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
            return LearnOutcome(cost_accepted=False, unified_installed=False)
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
        unified_installed = False
        if self.learning == "lifelong":
            if len(self._cost_rounds) == 1:
                self._unified = generated
                unified_installed = True
            else:
                combined = self._combine(ctx, generation)
                if combined is not None:
                    self._unified = combined
                    unified_installed = True
        return LearnOutcome(
            cost_accepted=True,
            unified_installed=unified_installed,
            description=generated.description,
        )

    def _combine(
        self, ctx: RoundContext, generation: CostGenerationResult
    ) -> GeneratedPythonCost | None:
        cfg = self.rig.cfg
        combinator = CombineCostGenerator(
            context=generation.generated_context,
            instruction=ctx.utterance_text,
            summaries=generation.summaries,
            run_dir=self._episode_dir / f"combine_{ctx.event_index:02d}",
            images=generation.images,
            use_images=cfg.llm_cost.use_images,
            model=cfg.llm_cost.model,
            strict=cfg.llm_cost.strict,
            mpc=None,
            rollout_fn=generation.eval_state.make_rollout_fn(),
            eval_state=generation.eval_state,
            save_candidate_videos=False,
            codex_cmd=cfg.llm_cost.codex_cmd,
            rounds=self._cost_rounds,
        )
        return combinator.generate(install=False)
