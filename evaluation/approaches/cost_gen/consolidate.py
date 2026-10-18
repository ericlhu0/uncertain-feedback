"""Consolidated cost generation: one unified replacement cost via combination."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from evaluation.approaches.cost_gen.base import CostGen
from evaluation.approaches.cost_gen.bounds import grounding_terms
from evaluation.approaches.cost_gen.structs import LearnOutcome, RoundContext
from uncertain_feedback.cost_generation import (
    CombineCostGenerator,
    CostGenerationResult,
)
from uncertain_feedback.evaluation_mechanism import EvalState
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
)
from uncertain_feedback.simulated_users.base import Bound


class ConsolidateCostGen(CostGen):
    """Immediate per-round costs, consolidated into one cost between attempts.

    Each accepted round's cost plans at once, stacked on the cost of the last
    consolidation (the paper's immediate update). :meth:`consolidate`, which
    the episode calls when the attempt ends, replaces them all with one combined
    cost (the slow lifelong update; formerly "lifelong"). Its preference is also
    kept as bound terms: stage two's grounding when a single round was learned,
    the combination's ``terms`` otherwise. :meth:`learned_bounds` is what
    learned steering compiles from them and from the stacked rounds' groundings.
    """

    def __init__(self, source: str = "chosen") -> None:
        super().__init__(source)
        self._unified: GeneratedPythonCost | None = None
        self._unified_terms: list[dict[str, Any]] = []
        # Rounds learned since the last consolidation, stacked on the unified cost.
        self._pending: list[int] = []
        self._bounds: tuple[Bound, ...] | None = None

    def reset(
        self, cfg: MpcRunConfig, base: CompositeTrajectoryCost, episode_dir: Path
    ) -> None:
        super().reset(cfg, base, episode_dir)
        self._install(None, [])
        self._pending = []

    def _install(
        self, cost: GeneratedPythonCost | None, terms: list[dict[str, Any]]
    ) -> None:
        self._unified = cost
        self._unified_terms = list(terms)
        self._bounds = None

    def _stacked(self) -> list[int]:
        return [i for i in self._pending if not self._cost_rounds[i].retired]

    def learned_bounds(self) -> tuple[Bound, ...]:
        """The bounds of every cost in effect that every recorded round admits."""
        if self._bounds is None:
            terms = list(self._unified_terms) if self._unified is not None else []
            for i in self._stacked():
                terms += grounding_terms(self._cost_rounds[i].grounding)
            self._bounds = self._admissible(terms)
        return self._bounds

    def snapshot(self, root: Path) -> dict[str, Any]:
        snapshot = super().snapshot(root)
        unified = self._unified
        snapshot["unified"] = (
            None
            if unified is None
            else {
                "code": unified.code,
                "params": unified.params,
                "description": unified.description,
                # It runs in the context of the round that installed it.
                "round": next(
                    (
                        i
                        for i, cost in enumerate(self._generated)
                        if cost.context is unified.context
                    ),
                    len(self._generated) - 1,
                ),
                "terms": self._unified_terms,
            }
        )
        snapshot["pending"] = list(self._pending)
        return snapshot

    def restore(self, snapshot: dict[str, Any], root: Path) -> None:
        super().restore(snapshot, root)
        # Snapshots from before consolidation moved between attempts carry
        # neither pending rounds nor bound terms.
        self._pending = list(snapshot.get("pending", []))
        unified = snapshot["unified"]
        if unified is None:
            self._install(None, [])
            return
        self._install(
            GeneratedPythonCost(
                unified["code"],
                dict(unified["params"]),
                self._generated[unified["round"]].context,
                unified["description"],
            ),
            unified.get("terms", []),
        )

    def learned_terms(self) -> list[GeneratedPythonCost]:
        unified = [self._unified] if self._unified is not None else []
        return unified + [self._generated[i] for i in self._stacked()]

    def retire(self, terms: Sequence[GeneratedPythonCost], note: str) -> None:
        """Stacked costs retire like immediate ones. Retiring the unified cost
        retires every round folded into it; the next combination re-derives a
        cost from all rounds, notes included."""
        super().retire(terms, note)
        if self._unified is not None and any(self._unified is term for term in terms):
            self._cost_rounds = [
                (
                    round_
                    if round_.retired or i in self._pending
                    else replace(round_, retired=note)
                )
                for i, round_ in enumerate(self._cost_rounds)
            ]
            self._install(None, [])
        self._bounds = None

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        return self.record(ctx, self.generate(ctx))

    def record(
        self, ctx: RoundContext, generation: CostGenerationResult
    ) -> LearnOutcome:
        outcome = super().record(ctx, generation)
        if outcome.cost_accepted:
            self._pending.append(len(self._cost_rounds) - 1)
            self._bounds = None
        return outcome

    def consolidate(self) -> None:
        """Replace the stacked rounds and the unified cost with one combined cost.

        A single learned round needs no combination: its own cost and grounding
        become the unified ones. When a combination fails the stacked costs stay
        in effect, so the next attempt still plans with everything learned.
        """
        if not self._pending:
            return
        if len(self._cost_rounds) == 1:
            round_ = self._cost_rounds[0]
            self._install(self._generated[0], grounding_terms(round_.grounding))
            self._pending = []
            return
        combined = self._combine()
        if combined is not None:
            self._install(*combined)
            self._pending = []

    def _combine(self) -> tuple[GeneratedPythonCost, list[dict[str, Any]]] | None:
        """The combined cost and its bound terms, or ``None`` when it failed.

        It runs in the context of the latest round, as the per-round
        combination did with the round that triggered it.
        """
        cfg = self._cfg
        assert cfg is not None
        latest = self._cost_rounds[-1]
        state = EvalState.load(latest.state_path)
        combinator = CombineCostGenerator(
            context=state.make_generated_context(),
            instruction=latest.feedback_text,
            summaries=latest.summaries,
            run_dir=self._episode_dir / "combine",
            use_images=cfg.llm_cost.use_images,
            model=cfg.llm_cost.model,
            strict=cfg.llm_cost.strict,
            mpc=None,
            rollout_fn=state.make_rollout_fn(),
            eval_state=state,
            save_candidate_videos=False,
            codex_cmd=cfg.llm_cost.codex_cmd,
            rounds=self._cost_rounds,
        )
        cost = combinator.generate(install=False)
        return None if cost is None else (cost, combinator.terms)
