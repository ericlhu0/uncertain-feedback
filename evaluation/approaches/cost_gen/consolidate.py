"""Consolidated cost generation: one unified replacement cost via combination."""

from __future__ import annotations

from pathlib import Path

from evaluation.approaches.cost_gen.base import CostGen
from evaluation.rig import EvalRig
from evaluation.structs import LearnOutcome, RoundContext
from uncertain_feedback.cost_generation import (
    CombineCostGenerator,
    CostGenerationResult,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
)


class ConsolidateCostGen(CostGen):
    """Per-round costs are replaced by one combined cost (formerly "lifelong")."""

    def __init__(self, source: str = "chosen") -> None:
        super().__init__(source)
        self._unified: GeneratedPythonCost | None = None

    def reset(
        self, rig: EvalRig, base: CompositeTrajectoryCost, episode_dir: Path
    ) -> None:
        super().reset(rig, base, episode_dir)
        self._unified = None

    def learned_terms(self) -> list[GeneratedPythonCost]:
        return [self._unified] if self._unified is not None else []

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        generated, generation = self._generate(ctx)
        if generated is None:
            return LearnOutcome(cost_accepted=False, unified_installed=False)
        unified_installed = False
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
        assert self._rig is not None
        cfg = self._rig.cfg
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
