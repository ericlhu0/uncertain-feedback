"""Consolidated cost generation: one unified replacement cost via combination."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from evaluation.approaches.cost_gen.base import CostGen
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


class ConsolidateCostGen(CostGen):
    """Immediate per-round costs, consolidated into one cost between attempts.

    Each accepted round's cost plans at once, stacked on the cost of the last
    consolidation (the paper's immediate update). :meth:`consolidate`, which
    the episode calls when the attempt ends, replaces them all with one combined
    cost (the slow lifelong update; formerly "lifelong").
    """

    def __init__(self, source: str = "chosen") -> None:
        super().__init__(source)
        self._unified: GeneratedPythonCost | None = None
        # Rounds learned since the last consolidation, stacked on the unified cost.
        self._pending: list[int] = []

    def reset(
        self, cfg: MpcRunConfig, base: CompositeTrajectoryCost, episode_dir: Path
    ) -> None:
        super().reset(cfg, base, episode_dir)
        self._unified = None
        self._pending = []

    def _stacked(self) -> list[int]:
        return [i for i in self._pending if not self._cost_rounds[i].retired]

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
            }
        )
        snapshot["pending"] = list(self._pending)
        return snapshot

    def restore(self, snapshot: dict[str, Any], root: Path) -> None:
        super().restore(snapshot, root)
        # Snapshots from before consolidation moved between attempts carry no
        # pending rounds.
        self._pending = list(snapshot.get("pending", []))
        unified = snapshot["unified"]
        self._unified = (
            None
            if unified is None
            else GeneratedPythonCost(
                unified["code"],
                dict(unified["params"]),
                self._generated[unified["round"]].context,
                unified["description"],
            )
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
            self._unified = None

    def learn(self, ctx: RoundContext) -> LearnOutcome:
        return self.record(ctx, self.generate(ctx))

    def record(
        self, ctx: RoundContext, generation: CostGenerationResult
    ) -> LearnOutcome:
        outcome = super().record(ctx, generation)
        if outcome.cost_accepted:
            self._pending.append(len(self._cost_rounds) - 1)
        return outcome

    def consolidate(self) -> None:
        """Replace the stacked rounds and the unified cost with one combined cost.

        A single learned round needs no combination: its own cost becomes the
        unified one. When a combination fails the stacked costs stay in effect,
        so the next attempt still plans with everything learned.
        """
        if not self._pending:
            return
        combined = (
            self._generated[0] if len(self._cost_rounds) == 1 else self._combine()
        )
        if combined is not None:
            self._unified = combined
            self._pending = []

    def _combine(self) -> GeneratedPythonCost | None:
        """The combined cost, or ``None`` when it failed.

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
        return combinator.generate(install=False)
