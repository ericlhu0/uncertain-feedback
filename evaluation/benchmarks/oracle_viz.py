"""Render the oracle correction a grounder is scored against.

Two case sources, both ending in the same contrast at the trigger step — what
the robot would have kept doing, against what the hidden bound says it should
do instead:

``sampled``
    The scenario generator in
    :mod:`uncertain_feedback.data_collection.dataset_auto_correction.clips`.
    A start arm configuration and a Cartesian goal are drawn, a naive reach is
    rolled between them, and a hidden bound is sampled that the reach is
    *guaranteed* to cross (:func:`sample_violating_bound` places it in the gap
    the rollout opens at a drawn crossing frame). The crossing induces the
    trigger step, and the oracle replans from there under that bound. The draw
    order matches ``ClipSource.generate``, so case ``i`` at seed ``s`` is run
    ``i`` of the clip set generated with that seed.

``persona``
    The ``evaluation/benchmarks`` interaction benchmark: bounds fixed per
    persona, trigger wherever the persona's curated goal first provokes them.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    ClipSource,
    sample_violating_bound,
)
from uncertain_feedback.data_collection.dataset_auto_correction.motion_facts import (
    motion_facts,
)
from uncertain_feedback.planners.mpc.arm_features import arm_aa_from_state
from uncertain_feedback.planners.mpc.config import MpcRunConfig, cfg_with_goal
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.simulated_users import (
    ATTRIBUTED_FEATURES,
    HiddenCostTerm,
    SimulatedUser,
    attribute_correction,
    first_violation_step,
    render_hidden_bounds,
    verbalize_everyday,
)
from uncertain_feedback.utils.plot import ArmVisualizer

_LOG = "[oracle-viz]"


@dataclass(frozen=True)
class OracleCase:
    """One trigger and the two futures that leave it."""

    label: str
    user: SimulatedUser
    goal: np.ndarray
    trigger_step: int
    q_feedback: np.ndarray
    oracle_correction: np.ndarray
    nominal_continuation: np.ndarray
    utterance: str
    detail: dict[str, Any]


def build_sampled_case(
    source: ClipSource, index: int, oracle_steps: int | None = None
) -> OracleCase:
    """Draw evaluation case ``index``: scenario, violating bound, oracle replan.

    Mirrors ``ClipSource.generate`` draw for draw — scenario, bound, then the
    correction-window length — so the case is reproducible from its index and
    lines up with the clip set of the same seed.

    The correction starts one frame before the bound is crossed, the last naive
    frame with positive clearance, rather than at the trigger the clip set cuts
    on: a correction anchored on a frame already past the pain threshold can
    never be acceptable to the simulated user, whose test is the peak violation
    over every frame it contains. Both steps are recorded in ``detail``.

    ``oracle_steps`` overrides the planner config's step budget for the oracle
    replan only. The clip config's 300 is a cap the detour around a bound can
    exhaust before reaching the goal, which leaves the case with no completed
    target motion; raising it here changes neither the draws nor the naive reach,
    so the sampled bound and trigger still match the clip set.
    """
    cfg = source.cfg
    rng = np.random.default_rng([cfg.seed, index])
    goal, naive = source.sample_scenario(rng, f"case {index}")
    sampled = sample_violating_bound(rng, naive, source.human, cfg, source.threshold)
    window = int(rng.integers(cfg.correction_frames[0], cfg.correction_frames[1] + 1))

    oracle_costs = CompositeTrajectoryCost(
        [
            *source.base.terms(),
            HiddenCostTerm(user=sampled.user, human=source.human),
        ]
    )
    start = sampled.crossing_step - 1
    q_feedback = np.asarray(naive[start], dtype=np.float64)
    continuation = rollout_to_goal(
        cfg_with_goal(source.run_cfg, goal),
        source.human.reset_human_with_q(q_feedback),
        goal,
        oracle_costs,
        steps=oracle_steps,
        progress_label=f"case {index} continuation",
        log_prefix=_LOG,
    ).history
    reach = goal_reach(
        source.human, cfg_with_goal(source.run_cfg, goal), continuation, goal
    )
    oracle = np.asarray(continuation[: window + 1], dtype=np.float64)
    nominal = np.asarray(naive[start : start + window + 1], dtype=np.float64)
    # motion_facts measures from frame n_prefix - 1, so 1 means the whole window.
    facts = motion_facts(oracle, source.human, 1)
    return OracleCase(
        label=f"case_{index:03d}",
        user=sampled.user,
        goal=goal,
        trigger_step=sampled.trigger_step,
        q_feedback=q_feedback,
        oracle_correction=oracle,
        nominal_continuation=nominal,
        utterance=facts.text,
        detail={
            "feature": sampled.feature,
            "bound_type": sampled.bound_type,
            "bound_value": round(sampled.value, 4),
            "peak_violation": round(sampled.peak_violation, 4),
            "crossing_step": sampled.crossing_step,
            "start_step": start,
            "naive_frames": int(len(naive)),
            "oracle_total_frames": int(len(continuation)),
            "oracle_reached": bool(reach["reached"]),
            "oracle_final_distance_m": round(float(reach["distance"]), 4),
            "oracle_frames": int(len(oracle)),
            # Shorter than the oracle window when the naive reach already
            # finished: its future genuinely ends there.
            "nominal_frames": int(len(nominal)),
        },
    )


def build_persona_case(
    cfg: MpcRunConfig,
    human: Human,
    user: SimulatedUser,
    goal: np.ndarray,
    seed: int = 0,
) -> OracleCase | None:
    """Replay one persona episode's first feedback round without any grounder.

    Mirrors :func:`evaluation.benchmarks.episode.run_episode` up to the point
    where the grounder is called. Returns ``None`` when the nominal rollout
    never violates the persona's fixed bounds, so there is no correction.
    """
    goal = np.asarray(goal, dtype=np.float64)
    goal_cfg = cfg_with_goal(cfg, goal)
    base = base_extra_costs(cfg.costs, human, user)
    oracle_costs = CompositeTrajectoryCost(
        [*base.terms(), HiddenCostTerm(user=user, human=human)]
    )

    oracle_path = rollout_to_goal(
        goal_cfg,
        human,
        goal,
        oracle_costs,
        progress_label=f"{user.name} oracle",
        log_prefix=_LOG,
    ).history
    nominal_rollout = rollout_to_goal(
        goal_cfg,
        human,
        goal,
        base,
        progress_label=f"{user.name} nominal",
        log_prefix=_LOG,
    ).history
    trigger = first_violation_step(
        user, human, nominal_rollout, cfg.corrections.trigger_threshold
    )
    if trigger is None:
        return None

    q_feedback = np.asarray(nominal_rollout[trigger], dtype=np.float64)
    nominal_plan = rollout_to_goal(
        goal_cfg,
        human.reset_human_with_q(q_feedback),
        goal,
        base,
        steps=cfg.simulated_user.nominal_steps,
        stop_at_goal=False,
        log_prefix=_LOG,
    ).history
    intent = attribute_correction(oracle_path, nominal_plan, q_feedback, human)
    window = oracle_path[intent.join_index : intent.join_index + len(nominal_plan)]
    utterance = verbalize_everyday(intent, np.random.default_rng(seed))
    return OracleCase(
        label=user.name,
        user=user,
        goal=goal,
        trigger_step=trigger,
        q_feedback=q_feedback,
        oracle_correction=np.asarray(window, dtype=np.float64),
        nominal_continuation=np.asarray(nominal_plan, dtype=np.float64),
        utterance="" if utterance is None else utterance.text,
        detail={
            "join_index": int(intent.join_index),
            "feature_deltas_rad": {
                name: round(float(intent.feature_deltas[name]), 4)
                for name in ATTRIBUTED_FEATURES
            },
            "wrist_offset_m": [round(float(v), 4) for v in intent.wrist_offset],
            "elbow_offset_m": [round(float(v), 4) for v in intent.elbow_offset],
        },
    )


def case_summary(case: OracleCase) -> dict[str, Any]:
    """The numbers behind the picture."""
    return {
        "label": case.label,
        "goal": [round(float(v), 4) for v in case.goal],
        "trigger_step": int(case.trigger_step),
        "utterance": case.utterance,
        **case.detail,
    }


def render_case(case: OracleCase, human: Human, out_dir: Path) -> None:
    """Write the overlay still plus oracle and nominal playback videos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    viz = ArmVisualizer(fk=human.fk)
    oracle_aa = arm_aa_from_state(case.oracle_correction, human)
    nominal_aa = arm_aa_from_state(case.nominal_continuation, human)

    viz.render_oracle_overlay(
        out_dir / f"{case.label}_overlay.png",
        oracle_traj=oracle_aa,
        nominal_traj=nominal_aa,
        current_q=arm_aa_from_state(case.q_feedback, human),
        spine3_pos=human.spine3_pos,
        spine3_aa=human.spine3_aa,
        body_pos=human.posture,
        goal_pos=case.goal,
        title="\n".join(textwrap.wrap(f"{case.label} — {case.utterance}", width=110)),
    )
    # A bound on a rotation barely moves the wrist, so the positional overlay
    # alone cannot show those corrections; this traces both futures through the
    # bound that actually fired.
    render_hidden_bounds(
        case.user,
        human,
        {"nominal": nominal_aa, "oracle": oracle_aa},
        out_dir / f"{case.label}_bound.png",
    )
    for name, traj, color in (
        ("oracle", oracle_aa, "green"),
        ("nominal", nominal_aa, "firebrick"),
    ):
        viz.render_rollout_video(
            traj,
            out_dir / f"{case.label}_{name}.mp4",
            spine3_pos=human.spine3_pos,
            spine3_aa=human.spine3_aa,
            body_pos=human.posture,
            cartesian_goal=case.goal,
            frame_colors=[color] * len(traj),
            fps=12,
        )
