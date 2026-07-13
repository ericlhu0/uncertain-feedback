"""Persist feedback rounds and unify their generated costs with a Codex agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uncertain_feedback.planners.mpc.costs.agent_costs import AgentCostGenerator
from uncertain_feedback.planners.mpc.costs.cost_feedback import EvalState
from uncertain_feedback.planners.mpc.costs.cost_generator import evaluate_candidate_cost
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
    replace_generated_costs,
)
from uncertain_feedback.planners.mpc.costs.prompts import build_combine_task_body

_COMBINE_TIMEOUT_SECONDS = 60.0 * 60.0


@dataclass(frozen=True)
class CostRound:  # pylint: disable=too-many-instance-attributes
    """Serializable context for one feedback-triggered cost generation round."""

    index: int
    goal: tuple[float, float, float] | None
    feedback_text: str
    trigger_step: int
    round_dir: Path
    state_path: Path
    cost_code: str
    params: dict[str, Any]
    summaries: dict[str, Any]
    image_paths: tuple[Path, ...]
    trajectory_index: int = 0
    trigger_reason: str = "discomfort"
    trigger_violation: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe record with absolute artifact paths."""
        return {
            "index": self.index,
            "goal": list(self.goal) if self.goal is not None else None,
            "feedback_text": self.feedback_text,
            "trigger_step": self.trigger_step,
            "round_dir": str(self.round_dir.resolve()),
            "state_path": str(self.state_path.resolve()),
            "cost_code": self.cost_code,
            "params": self.params,
            "summaries": self.summaries,
            "image_paths": [str(path.resolve()) for path in self.image_paths],
            "trajectory_index": self.trajectory_index,
            "trigger_reason": self.trigger_reason,
            "trigger_violation": self.trigger_violation,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CostRound":
        """Rebuild a round from :meth:`to_json` output."""
        return cls(
            index=int(data["index"]),
            goal=(
                tuple(float(value) for value in data["goal"])  # type: ignore[arg-type]
                if data.get("goal") is not None
                else None
            ),
            feedback_text=str(data["feedback_text"]),
            trigger_step=int(data["trigger_step"]),
            round_dir=Path(data["round_dir"]),
            state_path=Path(data["state_path"]),
            cost_code=str(data["cost_code"]),
            params=dict(data["params"]),
            summaries=dict(data["summaries"]),
            image_paths=tuple(Path(path) for path in data["image_paths"]),
            trajectory_index=int(data.get("trajectory_index", 0)),
            trigger_reason=str(data.get("trigger_reason", "discomfort")),
            trigger_violation=(
                float(data["trigger_violation"])
                if data.get("trigger_violation") is not None
                else None
            ),
        )


class CombineCostGenerator(AgentCostGenerator):
    """Ask Codex to replace per-round costs with one cross-round cost."""

    def __init__(
        self,
        *args: Any,
        rounds: Sequence[CostRound],
        timeout_seconds: float = _COMBINE_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, timeout_seconds=timeout_seconds, **kwargs)
        self.rounds = tuple(rounds)

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            prompt_text, image_paths = build_combine_task_body(
                [round_.to_json() for round_ in self.rounds]
            )
            (self.run_dir / "TASK.md").write_text(
                self._task_md(
                    prompt_text,
                    iterate=True,
                    image_input=[str(path) for path in image_paths],
                ),
                encoding="utf-8",
            )
            self._run_codex()
            self._record_stage_log()
            self._record_iteration_log(required=True)

            response_path = self.run_dir / "response.json"
            if not response_path.exists():
                raise GeneratedCostValidationError(
                    "codex did not produce response.json; see codex.log"
                )
            raw = response_path.read_text(encoding="utf-8")
            (self.run_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            response, cost = self.parse_cost(raw)
            scores = self._score_rounds(
                response.code, response.params, response.description
            )
            with open(self.run_dir / "scores.json", "w", encoding="utf-8") as file:
                json.dump(scores, file, indent=2, sort_keys=True)
            self.save_response(response)
            if install:
                self.install(cost)
            self._on_success(cost, installed=install)
            return cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None

    def _score_rounds(
        self, code: str, params: dict[str, Any], description: str
    ) -> dict[str, Any]:
        per_round: dict[str, float] = {}
        for round_ in self.rounds:
            state = EvalState.load(round_.state_path)
            context = state.make_generated_context()
            cost = GeneratedPythonCost(code, params, context, description)
            score, _ = evaluate_candidate_cost(context, cost, state.make_rollout_fn())
            per_round[str(round_.index)] = float(score)
        return {
            "per_round": per_round,
            "mean": float(np.mean(list(per_round.values()))),
        }

    def _task_md(
        self,
        prompt_text: str,
        *,
        iterate: bool,
        image_input: list[str] | None,
    ) -> str:
        del iterate
        images = "\n".join(f"- `{path}`" for path in image_input or [])
        commands = []
        render_script = (
            Path(__file__).resolve().parents[3]
            / "experiments"
            / "render_cost_comparison.py"
        )
        for round_ in self.rounds:
            commands.append(
                "uv run --project "
                f"{Path(__file__).resolve().parents[5]} python "
                f"{render_script} "
                f"--state {round_.state_path.resolve()} --response response.json "
                f"--out comparison_round_{round_.index}.png "
                f"--angles-out angles_round_{round_.index}.png"
            )
        rendered_commands = "\n".join(f"```\n{command}\n```" for command in commands)
        return (
            "# Multi-round cost-combination task\n\n"
            "Write the final unified cost JSON to `response.json`. Open every image "
            "below before drafting it.\n\n"
            f"## Visual context files\n\n{images}\n\n"
            "## Required logs\n\n"
            "Maintain `stage_log.md` with `## Evidence synthesis` and "
            "`## Final unified cost` sections. Maintain `ITERATION_LOG.md`; for each "
            "candidate, run every command below, open every comparison and angles "
            "image, and record each round's score, goal-reach result, mismatch, and "
            "resulting revision. The one unified cost must score well on every round.\n\n"
            f"{rendered_commands}\n\n---\n\n{prompt_text}\n"
        )

    def install(self, cost: GeneratedPythonCost) -> None:
        """Replace generated planner costs instead of appending another one."""
        if self.mpc is None:
            return
        existing = self.mpc._extra_costs  # pylint: disable=protected-access
        self.mpc.set_extra_costs(replace_generated_costs(existing, cost))
