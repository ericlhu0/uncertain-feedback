"""Fully automated simulated-user episodes with a multi-round re-trigger loop.

One episode plays a persona against one goal: an oracle path (MPC + hidden
cost from the initial pose) serves as the user's internal ideal; whenever the
robot's motion triggers discomfort, a deterministic attribution step contrasts
the robot's nominal continuation with the oracle window, the configured
verbalizer phrases it, and the oracle-path chooser picks a cluster and
magnitude. Everything except the words is verbalizer-invariant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.experiments.experiment_pipeline import (
    generate_cost_for_cluster,
    generate_uq_correction,
    goal_reach,
    rollout_to_goal,
    run_initial_rollout,
    save_rollout,
)
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc import LeftArmMPCMDMUQ
from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.config import MpcRunConfig, SimulatedUserConfig
from uncertain_feedback.planners.mpc.costs import (
    CombineCostGenerator,
    CompositeTrajectoryCost,
    CostRound,
    GeneratedPythonCost,
    MpcCostContext,
    artifact_run_dir,
    replace_generated_costs,
)
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    CorrectionIntent,
    HiddenCostTerm,
    SimulatedUser,
    Utterance,
    attribute_correction,
    choose_correction,
    first_violation_step,
    verbalize_everyday,
    verbalize_joint_resolved,
    verbalize_vague,
    violation_metrics,
)

_LOG_PREFIX = "[episode]"

# The verbalizer callable bound at episode setup: (intent, q_trigger, round_index).
BoundVerbalizer = Callable[[CorrectionIntent, np.ndarray, int], Utterance | None]


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _intent_record(intent: CorrectionIntent) -> dict[str, Any]:
    return {
        "join_index": intent.join_index,
        "feature_deltas": intent.feature_deltas,
        "wrist_offset": intent.wrist_offset.tolist(),
        "elbow_offset": intent.elbow_offset.tolist(),
    }


def _choice_record(choice: ChoiceResult) -> dict[str, Any]:
    return {
        "label": choice.label,
        "magnitude": choice.magnitude,
        "acceptable": {str(label): flag for label, flag in choice.acceptable.items()},
        "scores": {str(label): score for label, score in choice.scores.items()},
        "no_acceptable_cluster": choice.no_acceptable_cluster,
    }


def build_verbalizer(
    sim_cfg: SimulatedUserConfig,
    cfg: MpcRunConfig,
    context: MpcCostContext,
    oracle_path: np.ndarray,
    episode_key: str,
    cache_dir: Path,
) -> BoundVerbalizer:
    """Bind the configured verbalizer to its episode state (rng, VLM, oracle)."""
    if sim_cfg.verbalizer == "vague":
        return lambda intent, q_trigger, round_index: verbalize_vague(intent)
    if sim_cfg.verbalizer == "joint_resolved":
        return lambda intent, q_trigger, round_index: verbalize_joint_resolved(intent)
    if sim_cfg.verbalizer == "everyday":
        rng = np.random.default_rng(sim_cfg.seed)
        return lambda intent, q_trigger, round_index: verbalize_everyday(intent, rng)
    if sim_cfg.verbalizer == "visual":
        # Imported lazily so non-visual episodes never touch the OpenAI client.
        from uncertain_feedback.llm.openai_model import (  # pylint: disable=import-outside-toplevel
            OpenAIModel,
        )
        from uncertain_feedback.simulated_users.visual import (  # pylint: disable=import-outside-toplevel
            VisualVerbalizer,
        )

        if cfg.llm_cost.model is None:
            raise ValueError("simulated_user.verbalizer: visual needs llm_cost.model.")
        visual = VisualVerbalizer(
            OpenAIModel(
                model=cfg.llm_cost.model,
                system_prompt="You answer with exactly one short spoken sentence.",
            ),
            cache_dir,
        )
        return lambda intent, q_trigger, round_index: visual.verbalize(
            intent,
            q_trigger,
            oracle_path,
            context,
            episode_key,
            round_index,
            window=sim_cfg.nominal_steps,
        )
    raise ValueError(f"Unknown simulated_user.verbalizer {sim_cfg.verbalizer!r}.")


def _combine_costs(
    cfg: MpcRunConfig,
    utterance_text: str,
    generation: Any,
    cost_rounds: list[CostRound],
    combine_dir: Path,
    save_video: bool,
) -> GeneratedPythonCost | None:
    combinator = CombineCostGenerator(
        context=generation.generated_context,
        instruction=utterance_text,
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
        rounds=cost_rounds,
    )
    return combinator.generate(install=False)


def run_episode(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
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
    """Run one persona/goal episode with automated feedback until resolution."""
    sim_cfg = cfg.simulated_user
    root_dir = artifact_run_dir(artifact_base_dir, Path("episode_artifacts"))
    root_dir.mkdir(parents=True, exist_ok=True)
    base_extra_costs = mpc._extra_costs  # pylint: disable=protected-access
    if not cfg.cartesian.goals:
        raise ValueError("Episode experiments require cartesian.goals.")
    goal = np.asarray(cfg.cartesian.goals[0], dtype=np.float64)
    episode_key = f"{user.name}_seed{cfg.seed}"
    print(f"{_LOG_PREFIX} {user.name}: artifacts -> {root_dir}", flush=True)

    oracle_costs = CompositeTrajectoryCost(
        [*base_extra_costs.terms(), HiddenCostTerm(user=user, context=context)]
    )
    oracle_path = rollout_to_goal(
        cfg,
        initial_q,
        goal,
        context,
        oracle_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
        progress_label=f"{user.name} oracle_path",
        log_prefix=_LOG_PREFIX,
    )
    np.save(root_dir / "oracle_path.npy", oracle_path)

    verbalize = build_verbalizer(
        sim_cfg, cfg, context, oracle_path, episode_key, root_dir / "visual_cache"
    )

    summary: dict[str, Any] = {
        "persona": user.name,
        "verbalizer": sim_cfg.verbalizer,
        "goal": goal.tolist(),
        "oracle_steps": int(oracle_path.shape[0] - 1),
    }

    initial = run_initial_rollout(
        cfg,
        user,
        initial_q,
        context,
        base_extra_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
        root_dir,
        log_prefix=_LOG_PREFIX,
    )
    summary["trigger_step"] = initial.trigger_step
    if initial.trigger_step is None or initial.q_feedback is None:
        reach = goal_reach(context, cfg, initial.initial_traj, goal)
        summary.update(
            {
                "result": "no_violation",
                "rounds": [],
                "rounds_used": 0,
                "capped": False,
                "joint_success": reach["reached"],
            }
        )
        _write_json(root_dir / "episode_summary.json", summary)
        return summary

    q_feedback = initial.q_feedback
    q_history = [np.asarray(q, dtype=np.float64) for q in initial.q_history]
    min_join = 0
    rounds: list[dict[str, Any]] = []
    cost_rounds: list[CostRound] = []
    unified: GeneratedPythonCost | None = None
    result = "capped"
    joint_success = False

    for round_index in range(sim_cfg.max_rounds):
        round_dir = root_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_record: dict[str, Any] = {"index": round_index, "min_join": min_join}

        nominal_plan = rollout_to_goal(
            cfg,
            q_feedback,
            goal,
            context,
            base_extra_costs,
            body_pos,
            spine3_pos,
            spine3_aa,
            steps=sim_cfg.nominal_steps,
            stop_at_goal=False,
            log_prefix=_LOG_PREFIX,
        )
        np.save(round_dir / "nominal_plan.npy", nominal_plan)

        intent = attribute_correction(
            oracle_path, nominal_plan, q_feedback, context, min_join=min_join
        )
        round_record["intent"] = _intent_record(intent)
        utterance = verbalize(intent, q_feedback, round_index)
        if utterance is None:
            print(
                f"{_LOG_PREFIX} {user.name}: nothing left to say at round "
                f"{round_index}; ending episode",
                flush=True,
            )
            round_record["utterance"] = None
            rounds.append(round_record)
            _write_json(round_dir / "round_summary.json", round_record)
            result = "no_feedback_content"
            break
        round_record["utterance"] = {"text": utterance.text, "form": utterance.form}
        print(
            f"{_LOG_PREFIX} {user.name}: round {round_index} says "
            f"({utterance.form}): '{utterance.text}'",
            flush=True,
        )

        choices: list[ChoiceResult] = []
        round_join = min_join

        def _select(means: dict[int, np.ndarray]) -> tuple[int, float]:
            choice = choose_correction(
                user,
                context,
                means,
                oracle_path,
                min_join=round_join,  # pylint: disable=cell-var-from-loop
                threshold=cfg.corrections.trigger_threshold,
                magnitudes=sim_cfg.magnitudes,
            )
            choices.append(choice)  # pylint: disable=cell-var-from-loop
            return choice.label, choice.magnitude

        correction = generate_uq_correction(
            mpc,
            cfg,
            user,
            gen,
            initial_pose,
            q_feedback,
            context,
            round_dir,
            body_pos,
            mdm_frames=mdm_frames,
            frozen_body=frozen_body,
            feedback_text=utterance.text,
            cluster_selector=_select,
            log_prefix=_LOG_PREFIX,
        )
        round_record["choice"] = _choice_record(choices[-1])
        correction_q = canonical_arm_q(correction.correction_traj, context)
        np.save(round_dir / "correction.npy", correction_q)

        generation = generate_cost_for_cluster(
            mpc=None,
            cfg=cfg,
            instruction=utterance.text,
            cluster_traj=correction.correction_traj,
            current_q=q_feedback,
            q_history=q_history,
            context=context,
            base_extra_costs=base_extra_costs,
            cost_dir=round_dir / "cost_generation",
            body_pos=body_pos,
            spine3_pos=spine3_pos,
            spine3_aa=spine3_aa,
            candidate_trajs=correction.uq_result.cluster_means,
            highlight_label=correction.uq_result.chosen_label,
            install=False,
            save_candidate_videos=save_video,
            log_prefix=_LOG_PREFIX,
        )
        generated = generation.generated_cost
        round_record["generated_cost"] = (
            {"description": generated.description} if generated is not None else None
        )
        if generated is not None:
            state_path = round_dir / "state.pkl"
            generation.eval_state.save(state_path)
            cost_rounds.append(
                CostRound(
                    index=round_index,
                    goal=(float(goal[0]), float(goal[1]), float(goal[2])),
                    feedback_text=utterance.text,
                    trigger_step=len(q_history) - 1,
                    round_dir=round_dir.resolve(),
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
            if len(cost_rounds) == 1:
                unified = generated
            else:
                combined = _combine_costs(
                    cfg,
                    utterance.text,
                    generation,
                    cost_rounds,
                    root_dir / f"combine_round_{round_index:02d}",
                    save_video,
                )
                round_record["combine_succeeded"] = combined is not None
                if combined is not None:
                    unified = combined

        q_history.extend(np.asarray(correction_q[1:], dtype=np.float64))
        installed = replace_generated_costs(base_extra_costs, unified)
        continuation = rollout_to_goal(
            cfg,
            correction_q[-1],
            goal,
            context,
            installed,
            body_pos,
            spine3_pos,
            spine3_aa,
            progress_label=f"{user.name} round {round_index} continuation",
            log_prefix=_LOG_PREFIX,
        )
        np.save(round_dir / "continuation.npy", continuation)
        round_record["continuation_metrics"] = violation_metrics(
            user, context, continuation
        )

        retrigger = first_violation_step(
            user, context, continuation, cfg.corrections.trigger_threshold
        )
        round_record["retrigger_step"] = retrigger
        rounds.append(round_record)
        _write_json(round_dir / "round_summary.json", round_record)
        if retrigger is None:
            q_history.extend(np.asarray(continuation[1:], dtype=np.float64))
            reach = goal_reach(context, cfg, continuation, goal)
            result = "ok" if reach["reached"] else "goal_not_reached"
            joint_success = bool(reach["reached"])
            break
        q_history.extend(np.asarray(continuation[1 : retrigger + 1], dtype=np.float64))
        q_feedback = np.asarray(continuation[retrigger], dtype=np.float64)
        min_join = intent.join_index
        print(
            f"{_LOG_PREFIX} {user.name}: re-trigger at continuation step "
            f"{retrigger} (round {round_index})",
            flush=True,
        )

    executed = np.asarray(q_history, dtype=np.float64)
    save_rollout(
        executed,
        root_dir,
        "executed",
        context,
        body_pos,
        goal,
        save_video,
        log_prefix=_LOG_PREFIX,
    )
    summary.update(
        {
            "result": result,
            "rounds": rounds,
            "rounds_used": len(rounds),
            "capped": result == "capped",
            "joint_success": joint_success,
            "executed_metrics": violation_metrics(user, context, executed),
        }
    )
    _write_json(root_dir / "episode_summary.json", summary)
    print(
        f"{_LOG_PREFIX} {user.name}: episode {result} after {len(rounds)} round(s); "
        f"artifacts saved to {root_dir}",
        flush=True,
    )
    return summary
