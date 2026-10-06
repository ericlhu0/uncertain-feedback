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
from evaluation.metrics.cost_learning.success import goal_row
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.arm_features import canonical_arm_q
from uncertain_feedback.planners.mpc.config import MpcRunConfig, cfg_with_goal
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import (
    StallProposal,
    goal_reach,
    propose_unblocked_path,
    rollout_to_goal,
)
from uncertain_feedback.simulated_users import (
    ChoiceResult,
    HiddenCostTerm,
    SimulatedUser,
    attribute_correction,
    choose_correction,
    feedback_anchor,
    first_violation_step,
    violation_metrics,
)

_LOG = "[evaluation]"


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)


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


def _stall_proposal(
    cfg: MpcRunConfig,
    at: Human,
    costs: CompositeTrajectoryCost,
    user: SimulatedUser,
    threshold: float,
    label: str,
) -> tuple[StallProposal, bool] | None:
    """Offer the stalled arm a path with the blocking learned costs left out.

    The persona approves it when it stays under the trigger threshold, the same
    test the chooser applies to candidate corrections. ``None`` when there is no
    learned cost to leave out.
    """
    proposed = propose_unblocked_path(cfg, at, costs)
    if proposed is None:
        return None
    approved = first_violation_step(user, at, proposed.human.history, threshold) is None
    print(
        f"{_LOG} {label} stalled; proposal leaving out {len(proposed.dropped)} "
        f"learned cost(s) {'approved' if approved else 'rejected'}",
        flush=True,
    )
    return proposed, approved


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
                first.user, first.human, executed
            ).items()
        },
    }


def run_episode(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    cfg: MpcRunConfig,
    human: Human,
    user: SimulatedUser,
    task: InteractionTask,
    approach: Approach,
    episode_dir: Path,
) -> list[Interaction]:
    """Play one persona through the task's goal sequence against ``approach``.

    ``human`` supplies the body (and, without ``task.start_q``, the start arm).
    Per goal: an oracle path (base + hidden cost) defines the persona's ideal;
    the approach plans with its current learned costs; discomfort triggers a
    feedback round anchored on the last clean frame before the trigger (attribute -> verbalize -> ground -> choose -> learn ->
    continue) until resolution or the round cap. Learned costs persist across
    the goal sequence, so later goals measure accumulated personalization.

    With ``cartesian.stall`` set, a rollout that stalls short of the goal gets a
    proposal (the path with the blocking learned costs left out): the persona
    approves it and the arm follows it, or rejects it and the next round
    corrects it, with the proposal as that round's nominal plan; once that
    round's cost is learned, the left-out costs are retired
    (:meth:`CostGen.retire`), their rounds kept for consolidation.

    Returns one :class:`Interaction` per goal, the record every metric reads,
    and pickles the list to ``interactions.pkl`` for ``analyze_results.py``;
    ``episode_summary.json`` and ``executed.npy`` are written alongside, each
    goal's ``oracle_path.npy`` / ``initial_rollout.npy`` under ``goal_NN/`` and
    every round's ``trajectories.npz`` under ``goal_NN/round_NN/``.
    """
    episode_dir.mkdir(parents=True, exist_ok=True)
    assert cfg.cartesian is not None
    threshold = cfg.corrections.trigger_threshold
    sim_cfg = cfg.simulated_user
    base = base_extra_costs(cfg.costs, human, user)
    oracle_costs = CompositeTrajectoryCost(
        [*base.terms(), HiddenCostTerm(user=user, human=human)]
    )

    interactions: list[Interaction] = []
    event_index = 0
    q_current = np.asarray(
        human.q if task.start_q is None else task.start_q, dtype=np.float64
    )
    chooser_rng = np.random.default_rng(task.seed)
    propose = cfg.cartesian.stall is not None

    for goal_index, goal in enumerate(task.goals):
        goal_arr = np.asarray(goal, dtype=np.float64)
        goal_cfg = cfg_with_goal(cfg, goal_arr)
        goal_dir = episode_dir / f"goal_{goal_index:02d}"
        goal_dir.mkdir(parents=True, exist_ok=True)

        oracle_path = rollout_to_goal(
            goal_cfg,
            human.reset_human_with_q(q_current),
            goal_arr,
            oracle_costs,
            progress_label=f"{task.persona} goal {goal_index} oracle",
            log_prefix=_LOG,
        ).history
        np.save(goal_dir / "oracle_path.npy", oracle_path)
        approach.begin_goal(goal_arr, oracle_path)
        episode_key = (
            f"{task.persona}_{task.verbalizer}_seed{task.seed}_goal{goal_index}"
        )
        verbalize = bind_verbalizer(
            task, cfg, human, oracle_path, episode_key, goal_dir / "visual_cache"
        )

        rolled = rollout_to_goal(
            goal_cfg,
            human.reset_human_with_q(q_current),
            goal_arr,
            approach.planning_costs(),
            stop_on_stall=True,
            progress_label=f"{task.persona} goal {goal_index} rollout",
            log_prefix=_LOG,
        )
        rollout = rolled.history
        np.save(goal_dir / "initial_rollout.npy", rollout)
        trigger = first_violation_step(user, human, rollout, threshold)
        rounds: list[FeedbackRound] = []
        pending_plan: np.ndarray | None = None
        # Left out by a rejected proposal; deleted once the correction that
        # follows has been learned, their rounds kept for consolidation.
        pending_retire: tuple[GeneratedPythonCost, ...] = ()
        proposal_approved = False
        offer = (
            _stall_proposal(
                goal_cfg,
                rolled,
                approach.planning_costs(),
                user,
                threshold,
                f"{task.persona} goal {goal_index} rollout",
            )
            if trigger is None
            and propose
            and not goal_reach(human, goal_cfg, rollout, goal_arr)["reached"]
            else None
        )
        if offer is not None and offer[1]:
            rolled = rolled.step(offer[0].human.history[1:])
            proposal_approved = True
        elif offer is not None:
            trigger = len(rollout) - 1
            pending_plan = offer[0].human.history
            pending_retire = offer[0].dropped
        if trigger is None:
            reach = goal_reach(human, goal_cfg, rolled.history, goal_arr)
            interactions.append(
                Interaction(
                    task=task,
                    approach=approach.name,
                    user=user,
                    human=human,
                    goal_index=goal_index,
                    goal=goal_arr,
                    oracle_path=oracle_path,
                    initial_rollout=rollout,
                    trigger_step=None,
                    rounds=(),
                    result="no_violation",
                    reached=bool(reach["reached"]),
                    executed=rolled.history,
                    proposal_approved=proposal_approved,
                )
            )
            q_current = rolled.q
            continue

        # The executed motion so far; its last frame is where feedback is given.
        executed = rolled.rewind(feedback_anchor(user, human, rollout, trigger))
        min_join = 0
        result = "capped"
        reached = False

        for round_index in range(task.max_rounds):
            round_dir = goal_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            at_feedback = executed
            q_feedback = at_feedback.q
            if pending_plan is not None:
                nominal_plan, pending_plan = pending_plan, None
            else:
                nominal_plan = rollout_to_goal(
                    goal_cfg,
                    at_feedback,
                    goal_arr,
                    approach.planning_costs(),
                    steps=sim_cfg.nominal_steps,
                    stop_at_goal=False,
                    log_prefix=_LOG,
                ).history
            intent = attribute_correction(
                oracle_path, nominal_plan, q_feedback, human, min_join=min_join
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
                    human,
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
                utterance.text, at_feedback, nominal_plan, _select
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
                    human=at_feedback,
                    event_index=event_index,
                    rejected_labels=rejected,
                    nominal_plan=nominal_plan,
                    retire=pending_retire,
                )
            )
            learn_seconds = time.perf_counter() - learn_t0
            pending_retire = ()

            correction_q = canonical_arm_q(grounding.correction_traj, human)
            executed = executed.step(correction_q[1:])

            continuation = rollout_to_goal(
                goal_cfg,
                executed,
                goal_arr,
                approach.planning_costs(),
                stop_on_stall=True,
                progress_label=(
                    f"{task.persona} goal {goal_index} round {round_index} "
                    "continuation"
                ),
                log_prefix=_LOG,
            ).history
            retrigger = first_violation_step(user, human, continuation, threshold)
            _save_round_trajectories(
                round_dir,
                q_feedback,
                nominal_plan,
                grounding,
                correction_q,
                continuation,
            )
            grounding = replace(grounding, samples=None, sample_labels=None)

            stalled_at = (
                None if retrigger is not None else executed.step(continuation[1:])
            )
            reached = stalled_at is not None and bool(
                goal_reach(human, goal_cfg, continuation, goal_arr)["reached"]
            )
            offer = (
                _stall_proposal(
                    goal_cfg,
                    stalled_at,
                    approach.planning_costs(),
                    user,
                    threshold,
                    f"{task.persona} goal {goal_index} round {round_index}",
                )
                if stalled_at is not None and not reached and propose
                else None
            )

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
                    proposal_rejected=offer is not None and not offer[1],
                )
            )
            event_index += 1

            if stalled_at is None:
                assert retrigger is not None
                feedback_step = feedback_anchor(user, human, continuation, retrigger)
                executed = executed.step(continuation[1 : feedback_step + 1])
                min_join = intent.join_index
                continue
            executed = stalled_at
            if offer is not None and not offer[1]:
                pending_plan = offer[0].human.history
                pending_retire = offer[0].dropped
                min_join = intent.join_index
                continue
            if offer is not None:
                executed = executed.step(offer[0].human.history[1:])
                reached = bool(
                    goal_reach(human, goal_cfg, offer[0].human.history, goal_arr)[
                        "reached"
                    ]
                )
                proposal_approved = True
            result = "ok" if reached else "goal_not_reached"
            break

        q_current = executed.q
        interactions.append(
            Interaction(
                task=task,
                approach=approach.name,
                user=user,
                human=human,
                goal_index=goal_index,
                goal=goal_arr,
                oracle_path=oracle_path,
                initial_rollout=rollout,
                trigger_step=int(trigger),
                rounds=tuple(rounds),
                result=result,
                reached=reached,
                executed=executed.history,
                proposal_approved=proposal_approved,
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
