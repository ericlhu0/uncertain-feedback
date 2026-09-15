"""One simulated interaction episode: goals, triggers, corrections, learning."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.approaches.base import Approach
from evaluation.approaches.cost_gen.structs import RoundContext
from evaluation.benchmarks.structs import FeedbackRound, Interaction, InteractionTask
from evaluation.benchmarks.verbalize import bind_verbalizer
from evaluation.metrics.grounding.structs import GroundingResult
from evaluation.metrics.cost_learning.success import goal_row
from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost, MpcCostContext
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.planners.rig import PlanningRig, base_extra_costs, cfg_with_goal
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    HiddenCostTerm,
    SimulatedUser,
    attribute_correction,
    choose_correction,
    compute_violations,
    first_violation_step,
    violation_metrics,
)

_LOG = "[evaluation]"


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)


def feedback_anchor(
    user: SimulatedUser, context: MpcCostContext, trajectory: np.ndarray, trigger: int
) -> int:
    """The last frame before ``trigger`` with no violation, else ``trigger`` itself.

    A correction anchored on a frame already past the bound can never be
    acceptable to the simulated user, whose test is the peak violation over
    every frame it contains (see ``build_sampled_case``).
    """
    clean = np.nonzero(compute_violations(user, context, trajectory[:trigger]) <= 0)[0]
    return int(clean[-1]) if clean.size > 0 else trigger


def _save_round_trajectories(
    round_dir: Path,
    q_feedback: np.ndarray,
    nominal_plan: np.ndarray,
    grounding: GroundingResult,
    correction_q: np.ndarray,
    continuation: np.ndarray,
) -> None:
    """Every trajectory the round produced, so it can be re-scored later.

    ``candidate_<label>`` and ``correction`` are left-arm axis-angles (T, 3, 3);
    the rest are canonical q (T, 7). ``samples``/``sample_labels`` are the raw
    generator draws when the grounder sampled (see ``UqClusterResult.samples``).
    """
    extras: dict[str, np.ndarray] = {}
    if grounding.samples is not None:
        extras["samples"] = np.asarray(grounding.samples, dtype=np.float32)
    if grounding.sample_labels is not None:
        extras["sample_labels"] = np.asarray(grounding.sample_labels)
    np.savez_compressed(
        round_dir / "trajectories.npz",
        q_feedback=q_feedback,
        nominal_plan=nominal_plan,
        correction=grounding.correction_traj,
        correction_q=correction_q,
        continuation=continuation,
        **{f"candidate_{label}": traj for label, traj in grounding.candidates.items()},
        **extras,
    )


def _episode_summary(interactions: list[Interaction]) -> dict[str, Any]:
    """The episode record: one persona's goal sequence under one approach."""
    first = interactions[0]
    executed = np.concatenate(
        [first.executed, *(item.executed[1:] for item in interactions[1:])]
    )
    return {
        "persona": first.task.persona,
        "verbalizer": first.task.verbalizer,
        "seed": first.task.seed,
        "approach": first.approach,
        "goals": [list(goal) for goal in first.task.goals],
        "goal_results": [goal_row(item) for item in interactions],
        "feedback_events": sum(item.rounds_used for item in interactions),
        "all_goals_resolved": all(item.resolved for item in interactions),
        "all_goals_reached": all(item.reached for item in interactions),
        "executed_metrics": {
            key: float(value)
            for key, value in violation_metrics(
                first.user, first.context, executed
            ).items()
        },
    }


def run_episode(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    rig: PlanningRig,
    user: SimulatedUser,
    task: InteractionTask,
    approach: Approach,
    episode_dir: Path,
) -> list[Interaction]:
    """Play one persona through the task's goal sequence against ``approach``.

    Per goal: an oracle path (base + hidden cost) defines the persona's ideal;
    the approach plans with its current learned costs; discomfort triggers a
    feedback round anchored on the last clean frame before the trigger (attribute -> verbalize -> ground -> choose -> learn ->
    continue) until resolution or the round cap. Learned costs persist across
    the goal sequence, so later goals measure accumulated personalization.

    Returns one :class:`Interaction` per goal, the record every metric reads,
    and pickles the list to ``interactions.pkl`` for ``analyze_results.py``;
    ``episode_summary.json`` and ``executed.npy`` are written alongside, each
    goal's ``oracle_path.npy`` / ``initial_rollout.npy`` under ``goal_NN/`` and
    every round's ``trajectories.npz`` under ``goal_NN/round_NN/``.
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

    interactions: list[Interaction] = []
    event_index = 0
    q_current = np.asarray(
        rig.q0 if task.start_q is None else task.start_q, dtype=np.float64
    )
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
        approach.begin_goal(goal_arr, oracle_path)
        episode_key = (
            f"{task.persona}_{task.verbalizer}_seed{task.seed}_goal{goal_index}"
        )
        verbalize = bind_verbalizer(
            task,
            cfg,
            rig.context,
            oracle_path,
            episode_key,
            goal_dir / "visual_cache",
            body_pos=rig.body_pos,
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
        rounds: list[FeedbackRound] = []
        if trigger is None:
            reach = goal_reach(rig.context, goal_cfg, rollout, goal_arr)
            interactions.append(
                Interaction(
                    task=task,
                    approach=approach.name,
                    user=user,
                    context=rig.context,
                    goal_index=goal_index,
                    goal=goal_arr,
                    oracle_path=oracle_path,
                    initial_rollout=rollout,
                    trigger_step=None,
                    rounds=(),
                    result="no_violation",
                    reached=bool(reach["reached"]),
                    executed=np.asarray(rollout, dtype=np.float64),
                )
            )
            q_current = np.asarray(rollout[-1], dtype=np.float64)
            continue

        feedback_step = feedback_anchor(user, rig.context, rollout, trigger)
        q_feedback = np.asarray(rollout[feedback_step], dtype=np.float64)
        q_history = [
            np.asarray(q, dtype=np.float64) for q in rollout[: feedback_step + 1]
        ]
        min_join = 0
        result = "capped"
        reached = False

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
                utterance.text, q_feedback, nominal_plan, _select
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
            q_history.extend(np.asarray(correction_q[1:], dtype=np.float64))

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
            retrigger = first_violation_step(user, rig.context, continuation, threshold)
            _save_round_trajectories(
                round_dir, q_feedback, nominal_plan, grounding, correction_q, continuation
            )
            grounding = replace(grounding, samples=None, sample_labels=None)

            rounds.append(
                FeedbackRound(
                    round_index=round_index,
                    event_index=event_index,
                    round_dir=round_dir,
                    q_feedback=q_feedback,
                    nominal_plan=nominal_plan,
                    intent=intent,
                    utterance=utterance,
                    grounding=grounding,
                    choice=choice,
                    outcome=outcome,
                    correction_q=correction_q,
                    continuation=continuation,
                    retrigger_step=retrigger,
                    ground_seconds=ground_seconds,
                    learn_seconds=learn_seconds,
                )
            )
            event_index += 1

            if retrigger is None:
                q_history.extend(np.asarray(continuation[1:], dtype=np.float64))
                reach = goal_reach(rig.context, goal_cfg, continuation, goal_arr)
                reached = bool(reach["reached"])
                result = "ok" if reached else "goal_not_reached"
                break
            feedback_step = feedback_anchor(user, rig.context, continuation, retrigger)
            q_history.extend(
                np.asarray(continuation[1 : feedback_step + 1], dtype=np.float64)
            )
            q_feedback = np.asarray(continuation[feedback_step], dtype=np.float64)
            min_join = intent.join_index

        q_current = np.asarray(q_history[-1], dtype=np.float64)
        interactions.append(
            Interaction(
                task=task,
                approach=approach.name,
                user=user,
                context=rig.context,
                goal_index=goal_index,
                goal=goal_arr,
                oracle_path=oracle_path,
                initial_rollout=rollout,
                trigger_step=int(trigger),
                rounds=tuple(rounds),
                result=result,
                reached=reached,
                executed=np.asarray(q_history, dtype=np.float64),
            )
        )

    summary = _episode_summary(interactions)
    np.save(
        episode_dir / "executed.npy",
        np.concatenate(
            [interactions[0].executed, *(i.executed[1:] for i in interactions[1:])]
        ),
    )
    _write_json(episode_dir / "episode_summary.json", summary)
    with open(episode_dir / "interactions.pkl", "wb") as file:
        pickle.dump(interactions, file)
    return interactions
