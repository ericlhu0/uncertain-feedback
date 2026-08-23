"""One simulated interaction episode: goals, triggers, corrections, learning."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.approaches.base import Approach
from evaluation.metrics import round_row
from evaluation.rig import EvalRig, base_extra_costs, cfg_with_goal
from evaluation.structs import InteractionTask, RoundContext
from evaluation.verbalize import bind_verbalizer
from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    HiddenCostTerm,
    SimulatedUser,
    attribute_correction,
    choose_correction,
    first_violation_step,
    violation_metrics,
)

_LOG = "[evaluation]"


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)


def run_episode(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    rig: EvalRig,
    user: SimulatedUser,
    task: InteractionTask,
    approach: Approach,
    episode_dir: Path,
) -> dict[str, Any]:
    """Play one persona through the task's goal sequence against ``approach``.

    Per goal: an oracle path (base + hidden cost) defines the persona's ideal;
    the approach plans with its current learned costs; discomfort triggers a
    feedback round (attribute -> verbalize -> ground -> choose -> learn ->
    continue) until resolution or the round cap. Learned costs persist across
    the goal sequence, so later goals measure accumulated personalization.

    Returns ``{"rows": per-round records, "summary": episode record}``.
    """
    episode_dir.mkdir(parents=True, exist_ok=True)
    cfg = rig.cfg
    assert cfg.cartesian is not None
    threshold = cfg.corrections.trigger_threshold
    sim_cfg = cfg.simulated_user
    base = base_extra_costs(rig, user)
    oracle_costs = CompositeTrajectoryCost(
        [*base.terms(), HiddenCostTerm(user=user, context=rig.context)]
    )

    rows: list[dict[str, Any]] = []
    goal_results: list[dict[str, Any]] = []
    event_index = 0
    q_current = np.asarray(rig.q0, dtype=np.float64)
    executed: list[np.ndarray] = [q_current]
    chooser_rng = np.random.default_rng(task.seed)

    for goal_index, goal in enumerate(task.goals):
        goal_arr = np.asarray(goal, dtype=np.float64)
        goal_cfg = cfg_with_goal(cfg, goal_arr)
        goal_dir = episode_dir / f"goal_{goal_index:02d}"
        goal_dir.mkdir(parents=True, exist_ok=True)

        oracle_path = rollout_to_goal(
            goal_cfg,
            q_current,
            goal_arr,
            rig.context,
            oracle_costs,
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            progress_label=f"{task.persona} goal {goal_index} oracle",
            log_prefix=_LOG,
        )
        np.save(goal_dir / "oracle_path.npy", oracle_path)
        episode_key = (
            f"{task.persona}_{task.verbalizer}_seed{task.seed}_goal{goal_index}"
        )
        verbalize = bind_verbalizer(
            task, cfg, rig.context, oracle_path, episode_key, goal_dir / "visual_cache"
        )

        rollout = rollout_to_goal(
            goal_cfg,
            q_current,
            goal_arr,
            rig.context,
            approach.planning_costs(),
            rig.body_pos,
            rig.spine3_pos,
            rig.spine3_aa,
            progress_label=f"{task.persona} goal {goal_index} rollout",
            log_prefix=_LOG,
        )
        np.save(goal_dir / "initial_rollout.npy", rollout)
        trigger = first_violation_step(user, rig.context, rollout, threshold)
        if trigger is None:
            executed.extend(np.asarray(rollout[1:], dtype=np.float64))
            q_current = np.asarray(rollout[-1], dtype=np.float64)
            reach = goal_reach(rig.context, goal_cfg, rollout, goal_arr)
            goal_results.append(
                {
                    "goal_index": goal_index,
                    "result": "no_violation",
                    "reached": bool(reach["reached"]),
                    "rounds_used": 0,
                }
            )
            continue

        q_feedback = np.asarray(rollout[trigger], dtype=np.float64)
        q_history = [np.asarray(q, dtype=np.float64) for q in rollout[: trigger + 1]]
        executed.extend(q_history[1:])
        min_join = 0
        result = "capped"
        reached = False
        rounds_used = 0

        for round_index in range(task.max_rounds):
            round_dir = goal_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            nominal_plan = rollout_to_goal(
                goal_cfg,
                q_feedback,
                goal_arr,
                rig.context,
                approach.planning_costs(),
                rig.body_pos,
                rig.spine3_pos,
                rig.spine3_aa,
                steps=sim_cfg.nominal_steps,
                stop_at_goal=False,
                log_prefix=_LOG,
            )
            intent = attribute_correction(
                oracle_path, nominal_plan, q_feedback, rig.context, min_join=min_join
            )
            utterance = verbalize(intent, q_feedback, event_index)
            if utterance is None:
                result = "no_feedback_content"
                break
            print(
                f"{_LOG} {task.persona} goal {goal_index} round {round_index} "
                f"says ({utterance.form}): {utterance.text!r}",
                flush=True,
            )

            choices: list[ChoiceResult] = []
            round_join = min_join

            def _select(means: dict[int, np.ndarray]) -> tuple[int, float]:
                choice = choose_correction(
                    user,
                    rig.context,
                    means,
                    oracle_path,  # pylint: disable=cell-var-from-loop
                    min_join=round_join,  # pylint: disable=cell-var-from-loop
                    threshold=threshold,
                    magnitudes=sim_cfg.magnitudes,
                    mode=sim_cfg.chooser,
                    intent=intent,  # pylint: disable=cell-var-from-loop
                    rng=chooser_rng,
                )
                choices.append(choice)  # pylint: disable=cell-var-from-loop
                return choice.label, choice.magnitude

            ground_t0 = time.perf_counter()
            grounding = approach.ground(
                utterance.text, q_feedback, nominal_plan, _select, goal_arr
            )
            ground_seconds = time.perf_counter() - ground_t0
            choice = choices[-1]
            rejected = frozenset(
                label
                for label, acceptable in choice.acceptable.items()
                if not acceptable and label != grounding.chosen_label
            )

            learn_t0 = time.perf_counter()
            outcome = approach.learn(
                RoundContext(
                    round_dir=round_dir,
                    goal=goal_arr,
                    utterance_text=utterance.text,
                    grounding=grounding,
                    q_feedback=q_feedback,
                    q_history=list(q_history),
                    event_index=event_index,
                    rejected_labels=rejected,
                    nominal_plan=nominal_plan,
                )
            )
            learn_seconds = time.perf_counter() - learn_t0

            correction_q = canonical_arm_q(grounding.correction_traj, rig.context)
            np.save(round_dir / "correction.npy", correction_q)
            q_history.extend(np.asarray(correction_q[1:], dtype=np.float64))
            executed.extend(np.asarray(correction_q[1:], dtype=np.float64))

            continuation = rollout_to_goal(
                goal_cfg,
                np.asarray(correction_q[-1], dtype=np.float64),
                goal_arr,
                rig.context,
                approach.planning_costs(),
                rig.body_pos,
                rig.spine3_pos,
                rig.spine3_aa,
                progress_label=(
                    f"{task.persona} goal {goal_index} round {round_index} "
                    "continuation"
                ),
                log_prefix=_LOG,
            )
            np.save(round_dir / "continuation.npy", continuation)
            retrigger = first_violation_step(user, rig.context, continuation, threshold)

            rows.append(
                round_row(
                    task=task,
                    goal_index=goal_index,
                    round_index=round_index,
                    event_index=event_index,
                    user=user,
                    context=rig.context,
                    utterance_text=utterance.text,
                    utterance_form=utterance.form,
                    grounding=grounding,
                    choice=choice,
                    outcome=outcome,
                    continuation=continuation,
                    retrigger_step=retrigger,
                    ground_seconds=ground_seconds,
                    learn_seconds=learn_seconds,
                )
            )
            event_index += 1
            rounds_used = round_index + 1

            if retrigger is None:
                q_history.extend(np.asarray(continuation[1:], dtype=np.float64))
                executed.extend(np.asarray(continuation[1:], dtype=np.float64))
                reach = goal_reach(rig.context, goal_cfg, continuation, goal_arr)
                reached = bool(reach["reached"])
                result = "ok" if reached else "goal_not_reached"
                break
            q_history.extend(
                np.asarray(continuation[1 : retrigger + 1], dtype=np.float64)
            )
            executed.extend(
                np.asarray(continuation[1 : retrigger + 1], dtype=np.float64)
            )
            q_feedback = np.asarray(continuation[retrigger], dtype=np.float64)
            min_join = intent.join_index

        q_current = np.asarray(q_history[-1], dtype=np.float64)
        goal_results.append(
            {
                "goal_index": goal_index,
                "result": result,
                "reached": reached,
                "rounds_used": rounds_used,
            }
        )

    executed_arr = np.asarray(executed, dtype=np.float64)
    np.save(episode_dir / "executed.npy", executed_arr)
    summary: dict[str, Any] = {
        "persona": task.persona,
        "verbalizer": task.verbalizer,
        "seed": task.seed,
        "approach": approach.name,
        "goals": [list(goal) for goal in task.goals],
        "goal_results": goal_results,
        "feedback_events": event_index,
        "all_goals_resolved": all(
            record["result"] in ("ok", "no_violation") for record in goal_results
        ),
        "all_goals_reached": all(record["reached"] for record in goal_results),
        "executed_metrics": {
            key: float(value)
            for key, value in violation_metrics(user, rig.context, executed_arr).items()
        },
    }
    _write_json(episode_dir / "episode_summary.json", summary)
    return {"rows": rows, "summary": summary}
