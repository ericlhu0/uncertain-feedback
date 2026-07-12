"""Multi-goal feedback experiment with cross-round generated-cost combination."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.experiments.experiment_pipeline import (
    evaluate_cost_conditions,
    generate_cost_for_cluster,
    generate_uq_correction,
    run_initial_rollout,
)
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CombineCostGenerator,
    CostRound,
    GeneratedPythonCost,
    MpcCostContext,
    artifact_run_dir,
    replace_generated_costs,
)
from uncertain_feedback.simulated_users import SimulatedUser
from uncertain_feedback.simulated_users.viz import render_hidden_bounds


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _cost_record(cost: GeneratedPythonCost | None) -> dict[str, Any] | None:
    if cost is None:
        return None
    return {
        "description": cost.description,
        "params": cost.params,
        "code": cost.code,
    }


def run_multi_round_experiment(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    mpc: LeftArmMPCMDMUQ,
    cfg: MpcRunConfig,
    user: SimulatedUser,
    gen: MotionGenerator,
    initial_pose: np.ndarray,
    initial_q: np.ndarray,
    context: MpcCostContext,
    artifact_base_dir: Path,
    *,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    mdm_frames: int | None = None,
    frozen_body: bool = False,
    save_video: bool = False,
) -> dict[str, Any]:
    """Run successive goal/correction rounds and learn one replacement cost."""
    root_dir = artifact_run_dir(artifact_base_dir, Path("multi_round_artifacts"))
    root_dir.mkdir(parents=True, exist_ok=True)
    _write_json(root_dir / "history.json", [])
    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access
    rounds: list[CostRound] = []
    unified: GeneratedPythonCost | None = None
    round_summaries: list[dict[str, Any]] = []
    combine_runs: list[dict[str, Any]] = []

    print(f"[multi-round] {user.name}: artifacts -> {root_dir}", flush=True)
    for index, goal_values in enumerate(cfg.cartesian.goals):
        goal = (
            float(goal_values[0]),
            float(goal_values[1]),
            float(goal_values[2]),
        )
        goal_cfg = replace(cfg, cartesian=replace(cfg.cartesian, goals=[list(goal)]))
        round_dir = root_dir / f"round_{index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        installed = replace_generated_costs(base_extra_costs, unified)
        initial = run_initial_rollout(
            goal_cfg,
            user,
            initial_q,
            context,
            installed,
            body_pos,
            spine3_pos,
            spine3_aa,
            round_dir,
            log_prefix="[multi-round]",
        )
        round_summary: dict[str, Any] = {
            "index": index,
            "goal": list(goal),
            "trigger_step": initial.trigger_step,
            "trigger_violation": initial.trigger_violation,
            "unified_cost_before_round": _cost_record(unified),
        }
        if initial.trigger_step is None or initial.q_feedback is None:
            round_summary["result"] = "no_violation"
            round_summaries.append(round_summary)
            continue

        correction = generate_uq_correction(
            mpc,
            goal_cfg,
            user,
            gen,
            initial_pose,
            initial.q_feedback,
            context,
            round_dir,
            body_pos,
            mdm_frames=mdm_frames,
            frozen_body=frozen_body,
            log_prefix="[multi-round]",
        )
        np.save(round_dir / "correction.npy", correction.correction_traj)
        uq_result = correction.uq_result
        generation = generate_cost_for_cluster(
            mpc=None,
            cfg=goal_cfg,
            instruction=user.feedback_text,
            cluster_traj=correction.correction_traj,
            current_q=initial.q_feedback,
            q_history=initial.q_history,
            context=context,
            base_extra_costs=base_extra_costs,
            cost_dir=round_dir / "cost_generation",
            body_pos=body_pos,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            candidate_trajs=uq_result.cluster_means,
            highlight_label=uq_result.chosen_label,
            install=False,
            save_candidate_videos=save_video,
            log_prefix="[multi-round]",
        )
        generated = generation.generated_cost
        round_summary["chosen_cluster"] = uq_result.chosen_label
        round_summary["generated_cost"] = _cost_record(generated)
        if generated is None:
            round_summary["result"] = "cost_generation_failed"
            round_summaries.append(round_summary)
            continue

        state_path = round_dir / "state.pkl"
        generation.eval_state.save(state_path)
        cost_round = CostRound(
            index=index,
            goal=goal,
            feedback_text=user.feedback_text,
            trigger_step=initial.trigger_step,
            round_dir=round_dir.resolve(),
            state_path=state_path.resolve(),
            cost_code=generated.code,
            params=generated.params,
            summaries=generation.summaries,
            image_paths=tuple(path.resolve() for path in generation.images.values()),
        )
        rounds.append(cost_round)
        _write_json(root_dir / "history.json", [round_.to_json() for round_ in rounds])

        if len(rounds) == 1:
            unified = generated
        else:
            combine_dir = root_dir / f"combine_round_{index:02d}"
            combinator = CombineCostGenerator(
                context=generation.generated_context,
                instruction=user.feedback_text,
                summaries=generation.summaries,
                run_dir=combine_dir,
                images=generation.images,
                use_images=cfg.llm_cost.use_images,
                model=cfg.llm_cost.model,
                strict=cfg.llm_cost.strict,
                mpc=None,
                rollout_fn=generation.eval_state.make_rollout_fn(),
                eval_state=generation.eval_state,
                save_candidate_videos=save_video,
                codex_cmd=cfg.llm_cost.codex_cmd,
                rounds=rounds,
            )
            combined = combinator.generate(install=False)
            combine_summary: dict[str, Any] = {
                "round_index": index,
                "artifact_dir": str(combine_dir),
                "succeeded": combined is not None,
            }
            scores_path = combine_dir / "scores.json"
            if scores_path.exists():
                combine_summary["scores"] = json.loads(
                    scores_path.read_text(encoding="utf-8")
                )
            combine_runs.append(combine_summary)
            if combined is not None:
                unified = combined
            else:
                print(
                    f"[multi-round] {user.name}: combination failed at round "
                    f"{index}; keeping the previous unified cost",
                    flush=True,
                )
        round_summary["result"] = "ok"
        round_summary["unified_cost_after_round"] = _cost_record(unified)
        round_summaries.append(round_summary)

    final_results: dict[str, Any] = {}
    for index, goal_values in enumerate(cfg.cartesian.goals):
        goal_cfg = replace(
            cfg, cartesian=replace(cfg.cartesian, goals=[list(goal_values)])
        )
        eval_dir = root_dir / f"final_goal_{index:02d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        final_results[f"goal_{index}"] = evaluate_cost_conditions(
            goal_cfg,
            user,
            initial_q,
            context,
            base_extra_costs,
            unified,
            eval_dir,
            body_pos,
            spine3_pos,
            spine3_aa,
            save_video=save_video,
            log_prefix="[multi-round]",
        )
        trajectories = {
            name: np.load(eval_dir / name / "goal_0.npy")
            for name in ("base", "generated", "oracle")
            if (eval_dir / name / "goal_0.npy").exists()
        }
        if trajectories:
            render_hidden_bounds(
                user,
                context,
                trajectories,
                root_dir / f"hidden_bounds_goal_{index}.png",
            )

    summary = {
        "result": "ok",
        "persona": user.name,
        "feedback_text": user.feedback_text,
        "goals": cfg.cartesian.goals,
        "rounds": round_summaries,
        "feedback_round_count": len(rounds),
        "combine_runs": combine_runs,
        "unified_cost": _cost_record(unified),
        "results": final_results,
        "artifact_dir": str(root_dir),
    }
    _write_json(root_dir / "multi_round_summary.json", summary)
    print(f"[multi-round] {user.name}: artifacts saved to {root_dir}", flush=True)
    return summary
