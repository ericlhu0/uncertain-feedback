"""Bound transfer: learn on the first goal, then score the menus proposed on later ones."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.approaches.base import Approach
from evaluation.benchmarks.episode import run_episode
from evaluation.benchmarks.structs import InteractionTask, MenuProbe
from evaluation.benchmarks.verbalize import bind_verbalizer
from uncertain_feedback.planners.mpc.config import MpcRunConfig, cfg_with_goal
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import rollout_to_goal
from uncertain_feedback.simulated_users import (
    HiddenCostTerm,
    SimulatedUser,
    attribute_correction,
    choose_correction,
    feedback_anchor,
    first_violation_step,
)

_LOG = "[transfer]"


def probe_menu(
    cfg: MpcRunConfig,
    human: Human,
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
    sim_cfg = cfg.simulated_user
    threshold = cfg.corrections.trigger_threshold
    goal = np.asarray(task.goals[goal_index], dtype=np.float64)
    goal_cfg = cfg_with_goal(cfg, goal)
    goal_dir.mkdir(parents=True, exist_ok=True)
    base = base_extra_costs(cfg.costs, human, user)
    oracle_costs = CompositeTrajectoryCost(
        [*base.terms(), HiddenCostTerm(user=user, human=human)]
    )
    start = human.reset_human_with_q(q_start)
    label = f"{task.persona} goal {goal_index}"

    oracle_path = rollout_to_goal(
        goal_cfg,
        start,
        goal,
        oracle_costs,
        progress_label=f"{label} oracle",
        log_prefix=_LOG,
    ).history
    np.save(goal_dir / "oracle_path.npy", oracle_path)
    approach.begin_goal(goal, oracle_path)

    rolled = rollout_to_goal(
        goal_cfg,
        start,
        goal,
        base,
        progress_label=f"{label} unlearned rollout",
        log_prefix=_LOG,
    )
    rollout = rolled.history
    trigger = first_violation_step(user, human, rollout, threshold)
    if trigger is None:
        print(f"{_LOG} {label}: unlearned plan never violates; no probe.", flush=True)
        return oracle_path, None

    at_feedback = rolled.rewind(feedback_anchor(user, human, rollout, trigger))
    q_feedback = at_feedback.q
    nominal_plan = rollout_to_goal(
        goal_cfg,
        at_feedback,
        goal,
        base,
        steps=sim_cfg.nominal_steps,
        stop_at_goal=False,
        log_prefix=_LOG,
    ).history
    intent = attribute_correction(oracle_path, nominal_plan, q_feedback, human)
    verbalize = bind_verbalizer(
        task,
        cfg,
        human,
        oracle_path,
        f"{task.persona}_{task.verbalizer}_seed{task.seed}_probe{goal_index}",
        goal_dir / "visual_cache",
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
            human,
            means,
            oracle_path,
            threshold=threshold,
            magnitudes=sim_cfg.magnitudes,
            mode=sim_cfg.chooser,
            intent=intent,
            rng=rng,
        )
        return choice.label, choice.magnitude

    grounding = approach.ground(utterance.text, at_feedback, nominal_plan, _select)
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
        human=human,
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
    cfg: MpcRunConfig,
    human: Human,
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
    learned = run_episode(cfg, human, user, learn_task, approach, episode_dir / "learn")
    q_start = np.asarray(learned[0].oracle_path[-1], dtype=np.float64)

    probes: list[MenuProbe] = []
    for goal_index in range(1, len(task.goals)):
        oracle_path, probe = probe_menu(
            cfg,
            human,
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
