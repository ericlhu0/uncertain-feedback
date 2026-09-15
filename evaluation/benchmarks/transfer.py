"""Bound transfer: learn on the first goal, then score the menus proposed on later ones."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.approaches.base import Approach
from evaluation.benchmarks.episode import feedback_anchor, run_episode
from evaluation.benchmarks.structs import InteractionTask, MenuProbe
from evaluation.benchmarks.verbalize import bind_verbalizer
from uncertain_feedback.planners.mpc.costs import CompositeTrajectoryCost
from uncertain_feedback.planners.mpc.rollout import rollout_to_goal
from uncertain_feedback.planners.rig import PlanningRig, base_extra_costs, cfg_with_goal
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    SimulatedUser,
    attribute_correction,
    choose_correction,
    first_violation_step,
)

_LOG = "[transfer]"


def probe_menu(
    rig: PlanningRig,
    user: SimulatedUser,
    task: InteractionTask,
    approach: Approach,
    goal_index: int,
    q_start: np.ndarray,
    goal_dir: Path,
) -> tuple[np.ndarray, MenuProbe | None]:
    """Provoke feedback on one goal with the unlearned plan and collect the menu.

    The nominal rollout and plan use the base comfort costs only, so the
    feedback situation is the same for every approach and exists even when the
    learned cost would have avoided it. Returns the goal's oracle path (the next
    goal starts where it ends) and the probe, or ``None`` when the goal never
    provokes the bound or leaves the persona nothing to say.
    """
    cfg = rig.cfg
    sim_cfg = cfg.simulated_user
    threshold = cfg.corrections.trigger_threshold
    goal = np.asarray(task.goals[goal_index], dtype=np.float64)
    goal_cfg = cfg_with_goal(cfg, goal)
    goal_dir.mkdir(parents=True, exist_ok=True)
    base = base_extra_costs(rig, user)
    oracle_costs = CompositeTrajectoryCost(
        [*base.terms(), HiddenCostTerm(user=user, context=rig.context)]
    )
    rollout_kwargs = dict(
        body_pos=rig.body_pos,
        spine3_pos=rig.spine3_pos,
        spine3_aa=rig.spine3_aa,
        log_prefix=_LOG,
    )
    label = f"{task.persona} goal {goal_index}"

    oracle_path = rollout_to_goal(
        goal_cfg,
        q_start,
        goal,
        rig.context,
        oracle_costs,
        progress_label=f"{label} oracle",
        **rollout_kwargs,
    )
    np.save(goal_dir / "oracle_path.npy", oracle_path)
    approach.begin_goal(goal, oracle_path)

    rollout = rollout_to_goal(
        goal_cfg,
        q_start,
        goal,
        rig.context,
        base,
        progress_label=f"{label} unlearned rollout",
        **rollout_kwargs,
    )
    trigger = first_violation_step(user, rig.context, rollout, threshold)
    if trigger is None:
        print(f"{_LOG} {label}: unlearned plan never violates; no probe.", flush=True)
        return oracle_path, None

    step = feedback_anchor(user, rig.context, rollout, trigger)
    q_feedback = np.asarray(rollout[step], dtype=np.float64)
    nominal_plan = rollout_to_goal(
        goal_cfg,
        q_feedback,
        goal,
        rig.context,
        base,
        steps=sim_cfg.nominal_steps,
        stop_at_goal=False,
        **rollout_kwargs,
    )
    intent = attribute_correction(oracle_path, nominal_plan, q_feedback, rig.context)
    verbalize = bind_verbalizer(
        task,
        cfg,
        rig.context,
        oracle_path,
        f"{task.persona}_{task.verbalizer}_seed{task.seed}_probe{goal_index}",
        goal_dir / "visual_cache",
        body_pos=rig.body_pos,
    )
    utterance = verbalize(intent, q_feedback, goal_index)
    if utterance is None:
        print(f"{_LOG} {label}: no feedback content; no probe.", flush=True)
        return oracle_path, None
    print(f"{_LOG} {label} says ({utterance.form}): {utterance.text!r}", flush=True)

    rng = np.random.default_rng(task.seed)

    def _select(means: dict[int, np.ndarray]) -> tuple[int, float]:
        choice = choose_correction(
            user,
            rig.context,
            means,
            oracle_path,
            threshold=threshold,
            magnitudes=sim_cfg.magnitudes,
            mode=sim_cfg.chooser,
            intent=intent,
            rng=rng,
        )
        return choice.label, choice.magnitude

    grounding = approach.ground(utterance.text, q_feedback, nominal_plan, _select)
    np.savez_compressed(
        goal_dir / "menu.npz",
        q_feedback=q_feedback,
        nominal_plan=nominal_plan,
        **{f"candidate_{lab}": traj for lab, traj in grounding.candidates.items()},
    )
    probe = MenuProbe(
        task=task,
        approach=approach.name,
        user=user,
        context=rig.context,
        goal_index=goal_index,
        goal=goal,
        utterance=utterance,
        q_feedback=q_feedback,
        nominal_plan=nominal_plan,
        candidates=grounding.candidates,
        chosen_label=grounding.chosen_label,
        learned_terms=len(approach.cost_gen.learned_terms()),
    )
    return oracle_path, probe


def run_transfer(
    rig: PlanningRig,
    user: SimulatedUser,
    task: InteractionTask,
    approach: Approach,
    episode_dir: Path,
) -> list[MenuProbe]:
    """Learn on the task's first goal, then probe every later goal's menu.

    The learning goal is a full :func:`run_episode` on ``task.goals[0]`` under
    ``episode_dir/learn``; the approach keeps whatever it learned. Each later
    goal starts where the persona's oracle execution of the previous goal
    ended, so the test reaches do not depend on how well learning went.
    Probes are pickled to ``probes.pkl``.
    """
    learn_task = replace(task, goals=task.goals[:1])
    learned = run_episode(rig, user, learn_task, approach, episode_dir / "learn")
    q_start = np.asarray(learned[0].oracle_path[-1], dtype=np.float64)

    probes: list[MenuProbe] = []
    for goal_index in range(1, len(task.goals)):
        oracle_path, probe = probe_menu(
            rig,
            user,
            task,
            approach,
            goal_index,
            q_start,
            episode_dir / f"probe_{goal_index:02d}",
        )
        if probe is not None:
            probes.append(probe)
        q_start = np.asarray(oracle_path[-1], dtype=np.float64)
    with open(episode_dir / "probes.pkl", "wb") as file:
        pickle.dump(probes, file)
    return probes
