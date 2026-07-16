"""Multi-turn (``turns``) staged cost generator.

Runs the interpret stage once to fix the preference, then holds a real conversation
that iterates the ground+author work: each turn it parses the returned cost, scores
it with the evaluation harness, and feeds that score (and rollout comparison) back so
the model can revise both the numeric choices and the code, keeping the best-scoring
cost. The interpretation stays fixed — only grounding and authoring iterate.

Candidates are selected by ranking consistency (:func:`rank_candidate_cost`): the
cost is applied directly to the trajectories whose preference order the user
revealed, requiring the chosen correction to cost strictly less than the original
plan and explicitly marked-wrong candidates. A cost that captures the intent wins
without having to recreate the correction trajectory. When the context has no comparison trajectories the loop
falls back to the L2 rollout score. Either way, a goal-reaching candidate beats a
non-reaching one unless stage one judged the correction to conflict with the goal.
"""

from __future__ import annotations

import json
from typing import Any

from uncertain_feedback.planners.mpc.costs.cost_generator import (
    CostGenerator,
    CostRanking,
    evaluate_and_render,
    evaluate_candidate_cost,
    goal_reach_report,
    parse_goal_conflict,
    rank_candidate_cost,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
    LlmCostResponse,
    build_rollout_joint_comparison,
)
from uncertain_feedback.planners.mpc.costs.prompts import build_refine_prompt

# Stop early once the score fails to improve this many turns in a row.
_NO_IMPROVE_PATIENCE = 2


def _goal_feedback(report: dict[str, Any] | None, goal_conflict: bool) -> str:
    """Per-turn message telling the model whether the rollout still reaches the goal."""
    if report is None:
        return ""
    dist, thr = report["distance"], report["threshold"]
    if report["reached"]:
        return (
            f"\n\nGoal check: the arm still reaches the goal (wrist ended {dist:.3f} m "
            f"away, within the {thr:.3f} m threshold)."
        )
    if goal_conflict:
        return (
            f"\n\nGoal check: the arm ended {dist:.3f} m from the goal (threshold "
            f"{thr:.3f} m). This is acceptable here because the correction implies the "
            "goal should not be reached this way."
        )
    return (
        f"\n\nGoal check: THIS COST FAILED TO REACH THE GOAL — the wrist ended "
        f"{dist:.3f} m short (it must finish within {thr:.3f} m). Your stage-one "
        "interpretation said the goal is still reachable, so the motion MUST still reach "
        "it. Revise the cost so it reaches the goal (e.g. relax or bound the constraint "
        "near the goal) while still honoring the correction."
    )


class TurnsCostGenerator(CostGenerator):
    """Iterative cost generator driving a stateful LLM conversation."""

    def __init__(self, *args: Any, max_turns: int = 6, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_turns = max(1, int(max_turns))

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            llm = self.make_llm()
            interpretation = self.interpret(llm)
            best = self._converse(llm, interpretation)
            if best is None:
                raise GeneratedCostValidationError(
                    "no valid cost produced across turns"
                )
            best_cost, best_response, best_ranking = best
            self.save_response(best_response)
            self.save_rationale(
                best_response,
                interpret_raw=interpretation,
                ground_raw=None,
                ranking=best_ranking,
            )
            self.require_original_plan_improvement(best_ranking)
            if install:
                self.install(best_cost)
            self._on_success(best_cost, installed=install)
            return best_cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None

    def _converse(
        self,
        llm: Any,
        interpretation: str,
    ) -> tuple[GeneratedPythonCost, LlmCostResponse, CostRanking | None] | None:
        # Only insist the rollout still reaches the goal when stage one judged the goal
        # reachable; if the correction conflicts with the goal, stopping short is fine.
        goal_conflict = parse_goal_conflict(interpretation)
        prompt_text = build_refine_prompt(interpretation, self.summaries)
        (self.run_dir / "refine_prompt.txt").write_text(prompt_text, encoding="utf-8")
        messages: list[dict[str, Any]] = [{"role": "user", "text": prompt_text}]

        best: tuple[GeneratedPythonCost, LlmCostResponse, CostRanking | None] | None = (
            None
        )
        best_key: tuple[float, float, float] | None = None
        no_improve = 0

        for turn in range(self.max_turns):
            prompt_snapshot = json.dumps(messages, indent=2)
            raw = llm.converse(messages)
            messages.append({"role": "assistant", "text": raw})
            turn_dir = self.run_dir / f"turn_{turn}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            (turn_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            self._append_stage_log(f"refine turn {turn}", prompt_snapshot, raw)

            try:
                response, cost = self.parse_cost(raw)
            except GeneratedCostValidationError as exc:
                messages.append(
                    {
                        "role": "user",
                        "text": (
                            f"That response was not a usable cost: {exc}. "
                            "Return a corrected JSON object following the same "
                            "contract."
                        ),
                    }
                )
                continue

            if self.use_images:
                score, image_path, rollout = evaluate_and_render(
                    self.context,
                    cost,
                    self.rollout_fn,
                    turn_dir / "comparison.png",
                    angle_path=turn_dir / "angles.png",
                    rollout_path=(
                        turn_dir / "rollout.npy"
                        if self.save_candidate_videos
                        else None
                    ),
                    video_path=(
                        turn_dir / "rollout.mp4"
                        if self.save_candidate_videos
                        else None
                    ),
                )
            else:
                score, rollout = evaluate_candidate_cost(
                    self.context, cost, self.rollout_fn
                )
                image_path = None
            report = goal_reach_report(self.context, rollout)
            # A goal-reaching candidate beats a non-reaching one (unless the correction
            # conflicts with the goal); within that, candidates order by ranking
            # consistency, falling back to the L2 rollout score.
            reach_rank = (
                1 if report is not None and not goal_conflict and not report["reached"]
                else 0
            )
            ranking = rank_candidate_cost(self.context, cost)
            rank_key = ranking.sort_key if ranking is not None else (score, 0.0)
            key = (float(reach_rank), *rank_key)
            (turn_dir / "cost.py").write_text(response.code, encoding="utf-8")
            payload: dict[str, Any] = {"turn": turn, "l2_score": score}
            if ranking is not None:
                payload["ranking"] = ranking.as_json()
            payload["goal_reach"] = report
            with open(turn_dir / "score.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            if ranking is not None:
                print(
                    f"[cost-gen][turns] turn {turn}: rank_accuracy="
                    f"{ranking.rank_accuracy:.2f} margin="
                    f"{ranking.normalized_margin:.2f} inert={ranking.inert} "
                    f"(l2={score:.4f})"
                )
            else:
                print(f"[cost-gen][turns] turn {turn}: score={score:.4f}")

            if best_key is None or key < best_key:
                best_key = key
                best = (cost, response, ranking)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= _NO_IMPROVE_PATIENCE:
                    break

            joint_block = ""
            if rollout is not None:
                comparison = build_rollout_joint_comparison(self.context, rollout)
                joint_block = (
                    "\n\nJoint feature comparison (rollout vs. target):\n"
                    + json.dumps(comparison, indent=2)
                )

            goal_block = _goal_feedback(report, goal_conflict)
            score_block = _feedback_score_block(ranking, score)
            if image_path is not None:
                messages.append(
                    {
                        "role": "user",
                        "text": (
                            f"{score_block} The "
                            "first attached image overlays the motion your cost "
                            "produced (red, 'cost rollout') against the entire "
                            "intended corrected path (green, 'target corrected "
                            "path': the pre-correction motion, the correction, and "
                            "the continuation to the goal). The "
                            "second image plots each arm joint angle over time for "
                            "the same two trajectories (green target vs red rollout) "
                            "so you can see the shape of the movement, not just "
                            "endpoints. Use them to see what your cost actually does, "
                            "revise it, and return the JSON object "
                            f"again.{joint_block}{goal_block}"
                        ),
                        "images": [str(image_path), str(turn_dir / "angles.png")],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "text": (
                            f"{score_block} Revise the cost accordingly and return "
                            f"the JSON object again.{joint_block}{goal_block}"
                        ),
                    }
                )

        return best


def _feedback_score_block(ranking: CostRanking | None, score: float) -> str:
    """Describe this turn's evaluation to the model."""
    if ranking is None:
        return (
            f"That cost scored {score:.4f} (lower is better, where the score "
            "measures how well the resulting motion matches the user's correction)."
        )
    if ranking.inert:
        return (
            "Your cost function returned (near-)identical values for every candidate "
            "trajectory — it never discriminates, so it cannot steer the arm. "
            "Per-trajectory costs: "
            f"{json.dumps(ranking.costs)}. Make the cost actually respond to the "
            "preference."
        )
    return (
        "Your cost function was evaluated directly on the candidate trajectories "
        "(each resampled to equidistant joint-space points, so only the path "
        "matters, not timing). Per-trajectory costs: "
        f"{json.dumps(ranking.costs)}. The user's chosen correction "
        "('chosen_correction') must cost less than the original plan they "
        "interrupted ('original_plan') and strictly less than the marked-wrong "
        "candidates "
        "('rejected_cluster_*'). Rank accuracy: "
        f"{ranking.rank_accuracy:.2f} (fraction of those orderings satisfied), "
        f"separation margin: {ranking.normalized_margin:.2f} (both higher is "
        "better)."
    )
