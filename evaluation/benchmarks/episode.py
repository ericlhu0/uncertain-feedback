"""One simulated interaction episode: goals, triggers, corrections, learning."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, replace
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
from uncertain_feedback.planners.mpc.goal_spaces import GoalRegion
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


def _stall_proposal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: MpcRunConfig,
    at: Human,
    goal: GoalRegion,
    costs: CompositeTrajectoryCost,
    user: SimulatedUser,
    threshold: float,
    label: str,
) -> tuple[StallProposal, bool] | None:
    """Offer the stalled arm a path with the blocking learned costs left out.

    The persona approves it when it stays under the trigger threshold, the same
    test the chooser applies to candidate corrections. ``None`` when there is no
    learned cost to leave out, or when leaving them out brings the arm no
    closer to the goal (by ``cartesian.stall.min_gain``): then no learned cost
    is what holds it back, and there is nothing worth asking about.
    """
    assert cfg.cartesian is not None and cfg.cartesian.stall is not None
    proposed = propose_unblocked_path(cfg, at, costs)
    if proposed is None:
        return None
    path = proposed.human.history
    here = goal_reach(at, cfg, at.q[None], goal)["distance"]
    there = goal_reach(at, cfg, path, goal)
    if not there["reached"] and here - there["distance"] < cfg.cartesian.stall.min_gain:
        print(
            f"{_LOG} {label} stalled; leaving out the learned costs gets no closer "
            f"to the goal ({here:.4f} -> {there['distance']:.4f}), nothing proposed",
            flush=True,
        )
        return None
    approved = first_violation_step(user, at, path, threshold) is None
    print(
        f"{_LOG} {label} stalled; proposal leaving out {len(proposed.dropped)} "
        f"learned cost(s) {'approved' if approved else 'rejected'}",
        flush=True,
    )
    return proposed, approved


@dataclass(frozen=True)
class _Motion:
    """The arm's way toward the goal from one pose until it needs the persona.

    ``path`` starts at that pose: replans under every learned cost, with each
    stall proposal the persona approved followed in between
    (``approved_spans``, inclusive frame ranges of ``path``). It ends at
    ``trigger`` (its first frame over the threshold), at the goal
    (``reached``), at a stall with nothing to propose, or at a stall whose
    proposal the persona turned down (``rejected``).
    """

    path: np.ndarray
    trigger: int | None
    reached: bool
    approved_spans: tuple[tuple[int, int], ...]
    rejected: StallProposal | None


def _move_to_goal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: MpcRunConfig,
    start: Human,
    goal: GoalRegion,
    approach: Approach,
    user: SimulatedUser,
    approvals_left: int,
    label: str,
) -> _Motion:
    """Replan to the goal under the learned costs, past any approved stalls.

    A stall short of the goal gets a proposal when ``cartesian.stall`` is set
    (:func:`_stall_proposal`); the arm follows an approved one and replans from
    its end under every learned cost, so the goal goes on. At most
    ``approvals_left`` proposals are approved, so a goal the planner keeps
    stalling on still ends.
    """
    assert cfg.cartesian is not None
    threshold = cfg.corrections.trigger_threshold
    path = start.q[None]
    at = start
    spans: list[tuple[int, int]] = []
    while True:
        segment = rollout_to_goal(
            cfg,
            at,
            goal,
            approach.planning_costs(),
            stop_on_stall=True,
            progress_label=f"{label} after approval {len(spans)}" if spans else label,
            log_prefix=_LOG,
        ).history
        trigger = first_violation_step(user, at, segment, threshold)
        offset = len(path) - 1
        path = np.concatenate([path, segment[1:]])
        if trigger is not None:
            return _Motion(path, offset + trigger, False, tuple(spans), None)
        at = at.step(segment[1:])
        if goal_reach(at, cfg, segment, goal)["reached"]:
            return _Motion(path, None, True, tuple(spans), None)
        offer = (
            _stall_proposal(
                cfg, at, goal, approach.planning_costs(), user, threshold, label
            )
            if cfg.cartesian.stall is not None and len(spans) < approvals_left
            else None
        )
        if offer is None:
            return _Motion(path, None, False, tuple(spans), None)
        proposal, approved = offer
        if not approved:
            return _Motion(path, None, False, tuple(spans), proposal)
        followed = proposal.human.history
        spans.append((len(path) - 1, len(path) + len(followed) - 2))
        path = np.concatenate([path, followed[1:]])
        at = at.step(followed[1:])
        if goal_reach(at, cfg, followed, goal)["reached"]:
            return _Motion(path, None, True, tuple(spans), None)


def _executed_spans(
    spans: tuple[tuple[int, int], ...], offset: int, last: int
) -> list[tuple[int, int]]:
    """A motion's approved spans as ``executed`` frames, its path cut at ``last``."""
    return [
        (offset + start, offset + min(end, last))
        for start, end in spans
        if start < last
    ]


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
        "goals": [repr(goal) for goal in first.task.goals],
        "goal_results": [goal_row(item) for item in interactions],
        "feedback_events": sum(item.rounds_used for item in interactions),
        "approvals": sum(item.approvals for item in interactions),
        "user_turns": sum(item.user_turns for item in interactions),
        "violation_count": sum(item.violation_count for item in interactions),
        "robot_seconds": sum(item.robot_seconds for item in interactions),
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
    proposal (the path with the blocking learned costs left out) if that gets
    it closer to the goal. The persona approves it and the arm follows it, then
    goes on toward the goal under every learned cost (at most ``max_rounds``
    approvals per goal); or rejects it and the next round corrects it, with the
    proposal as that round's nominal plan; once that round's cost is learned,
    the left-out costs are retired (:meth:`CostGen.retire`), their rounds kept
    for consolidation.

    Returns one :class:`Interaction` per goal, the record every metric reads,
    and pickles the list to ``interactions.pkl`` for ``analyze_results.py``;
    ``episode_summary.json`` and ``executed.npy`` are written alongside, as is
    ``learned_state.json``, what the approach learned once it has consolidated
    (:meth:`Approach.consolidate`, :meth:`Approach.save`), which a later run
    resumes from with ``resume_from``; each
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

    for goal_index, task_goal in enumerate(task.goals):
        # Start-relative goals anchor here, not on each replan's start.
        goal = task_goal.bind(human.arm_aa_from_q(q_current), human)
        goal_cfg = cfg_with_goal(cfg, goal)
        goal_dir = episode_dir / f"goal_{goal_index:02d}"
        goal_dir.mkdir(parents=True, exist_ok=True)

        oracle_path = rollout_to_goal(
            goal_cfg,
            human.reset_human_with_q(q_current),
            goal,
            oracle_costs,
            progress_label=f"{task.persona} goal {goal_index} oracle",
            log_prefix=_LOG,
        ).history
        np.save(goal_dir / "oracle_path.npy", oracle_path)
        approach.begin_goal(goal, oracle_path)
        episode_key = (
            f"{task.persona}_{task.verbalizer}_seed{task.seed}_goal{goal_index}"
        )
        verbalize = bind_verbalizer(
            task, cfg, human, oracle_path, episode_key, goal_dir / "visual_cache"
        )

        start = human.reset_human_with_q(q_current)
        plan_t0 = time.perf_counter()
        before = _move_to_goal(
            goal_cfg,
            start,
            goal,
            approach,
            user,
            task.max_rounds,
            f"{task.persona} goal {goal_index} rollout",
        )
        plan_seconds = time.perf_counter() - plan_t0
        rollout = before.path
        np.save(goal_dir / "initial_rollout.npy", rollout)
        trigger = before.trigger
        approvals = len(before.approved_spans)
        rounds: list[FeedbackRound] = []
        pending_plan: np.ndarray | None = None
        # Left out by a rejected proposal; deleted once the correction that
        # follows has been learned, their rounds kept for consolidation.
        pending_retire: tuple[GeneratedPythonCost, ...] = ()
        if before.rejected is not None:
            trigger = len(rollout) - 1
            pending_plan = before.rejected.human.history
            pending_retire = before.rejected.dropped
        if trigger is None:
            interactions.append(
                Interaction(
                    task=task,
                    approach=approach.name,
                    user=user,
                    human=human,
                    goal_index=goal_index,
                    goal=goal,
                    oracle_path=oracle_path,
                    initial_rollout=rollout,
                    trigger_step=None,
                    rounds=(),
                    result="no_violation" if before.reached else "goal_not_reached",
                    reached=before.reached,
                    executed=rollout,
                    approvals=approvals,
                    approved_spans=before.approved_spans,
                    plan_seconds=plan_seconds,
                    trigger_threshold=threshold,
                )
            )
            q_current = rollout[-1]
            continue

        # The executed motion so far; its last frame is where feedback is given.
        anchor = feedback_anchor(user, human, rollout, trigger)
        executed = start.step(rollout[1 : anchor + 1])
        approved_spans = _executed_spans(before.approved_spans, 0, anchor)
        min_join = 0
        result = "capped"
        reached = False

        for round_index in range(task.max_rounds):
            round_dir = goal_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            at_feedback = executed
            q_feedback = at_feedback.q
            round_plan_t0 = time.perf_counter()
            if pending_plan is not None:
                nominal_plan, pending_plan = pending_plan, None
            else:
                nominal_plan = rollout_to_goal(
                    goal_cfg,
                    at_feedback,
                    goal,
                    approach.planning_costs(),
                    steps=sim_cfg.nominal_steps,
                    stop_at_goal=False,
                    log_prefix=_LOG,
                ).history
            round_plan_seconds = time.perf_counter() - round_plan_t0
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
            # A grounder may play its pick at another size than the user chose
            # (llm_keypoint runs its edit at 1x); record the grounder's size.
            choice = replace(choices[-1], magnitude=grounding.magnitude)
            rejected = frozenset(
                label
                for label, acceptable in choice.acceptable.items()
                if not acceptable and label != grounding.chosen_label
            )

            learn_t0 = time.perf_counter()
            outcome = approach.learn(
                RoundContext(
                    round_dir=round_dir,
                    goal=goal,
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

            round_plan_t0 = time.perf_counter()
            after = _move_to_goal(
                goal_cfg,
                executed,
                goal,
                approach,
                user,
                task.max_rounds - approvals,
                f"{task.persona} goal {goal_index} round {round_index} continuation",
            )
            round_plan_seconds += time.perf_counter() - round_plan_t0
            continuation = after.path
            approvals += len(after.approved_spans)
            _save_round_trajectories(
                round_dir,
                q_feedback,
                nominal_plan,
                grounding,
                correction_q,
                continuation,
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
                    retrigger_step=after.trigger,
                    ground_seconds=ground_seconds,
                    learn_seconds=learn_seconds,
                    proposal_rejected=after.rejected is not None,
                    approvals=len(after.approved_spans),
                    plan_seconds=round_plan_seconds,
                )
            )
            event_index += 1

            offset = len(executed.history) - 1
            if after.trigger is not None:
                feedback_step = feedback_anchor(
                    user, human, continuation, after.trigger
                )
                approved_spans += _executed_spans(
                    after.approved_spans, offset, feedback_step
                )
                executed = executed.step(continuation[1 : feedback_step + 1])
                min_join = intent.join_index
                continue
            approved_spans += _executed_spans(
                after.approved_spans, offset, len(continuation) - 1
            )
            executed = executed.step(continuation[1:])
            if after.rejected is not None:
                pending_plan = after.rejected.human.history
                pending_retire = after.rejected.dropped
                min_join = intent.join_index
                continue
            reached = after.reached
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
                goal=goal,
                oracle_path=oracle_path,
                initial_rollout=rollout,
                trigger_step=int(trigger),
                rounds=tuple(rounds),
                result=result,
                reached=reached,
                executed=executed.history,
                approvals=approvals,
                approved_spans=tuple(approved_spans),
                stall_rejected=before.rejected is not None,
                plan_seconds=plan_seconds,
                trigger_threshold=threshold,
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
