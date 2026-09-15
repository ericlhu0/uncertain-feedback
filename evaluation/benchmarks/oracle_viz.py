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
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    MpcCostContext,
)
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.planners.rig import PlanningRig, base_extra_costs, cfg_with_goal
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
    sampled = sample_violating_bound(rng, naive, source.context, cfg, source.threshold)
    window = int(rng.integers(cfg.correction_frames[0], cfg.correction_frames[1] + 1))

    oracle_costs = CompositeTrajectoryCost(
        [
            *source.base.terms(),
            HiddenCostTerm(user=sampled.user, context=source.context),
        ]
    )
    start = sampled.crossing_step - 1
    q_feedback = np.asarray(naive[start], dtype=np.float64)
    continuation = rollout_to_goal(
        cfg_with_goal(source.run_cfg, goal),
        q_feedback,
        goal,
        source.context,
        oracle_costs,
        source.body_pos,
        source.context.spine3_pos,
        source.context.spine3_aa,
        steps=oracle_steps,
        progress_label=f"case {index} continuation",
        log_prefix=_LOG,
    )
    reach = goal_reach(
        source.context, cfg_with_goal(source.run_cfg, goal), continuation, goal
    )
    oracle = np.asarray(continuation[: window + 1], dtype=np.float64)
    nominal = np.asarray(naive[start : start + window + 1], dtype=np.float64)
    # motion_facts measures from frame n_prefix - 1, so 1 means the whole window.
    facts = motion_facts(oracle, source.context, 1)
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
    rig: PlanningRig, user: SimulatedUser, goal: np.ndarray, seed: int = 0
) -> OracleCase | None:
    """Replay one persona episode's first feedback round without any grounder.

    Mirrors :func:`evaluation.benchmarks.episode.run_episode` up to the point
    where the grounder is called. Returns ``None`` when the nominal rollout
    never violates the persona's fixed bounds, so there is no correction.
    """
    cfg = rig.cfg
    goal = np.asarray(goal, dtype=np.float64)
    goal_cfg = cfg_with_goal(cfg, goal)
    base = base_extra_costs(rig, user)
    oracle_costs = CompositeTrajectoryCost(
        [*base.terms(), HiddenCostTerm(user=user, context=rig.context)]
    )

    oracle_path = rollout_to_goal(
        goal_cfg,
        rig.q0,
        goal,
        rig.context,
        oracle_costs,
        rig.body_pos,
        rig.spine3_pos,
        rig.spine3_aa,
        progress_label=f"{user.name} oracle",
        log_prefix=_LOG,
    )
    nominal_rollout = rollout_to_goal(
        goal_cfg,
        rig.q0,
        goal,
        rig.context,
        base,
        rig.body_pos,
        rig.spine3_pos,
        rig.spine3_aa,
        progress_label=f"{user.name} nominal",
        log_prefix=_LOG,
    )
    trigger = first_violation_step(
        user, rig.context, nominal_rollout, cfg.corrections.trigger_threshold
    )
    if trigger is None:
        return None

    q_feedback = np.asarray(nominal_rollout[trigger], dtype=np.float64)
    nominal_plan = rollout_to_goal(
        goal_cfg,
        q_feedback,
        goal,
        rig.context,
        base,
        rig.body_pos,
        rig.spine3_pos,
        rig.spine3_aa,
        steps=cfg.simulated_user.nominal_steps,
        stop_at_goal=False,
        log_prefix=_LOG,
    )
    intent = attribute_correction(oracle_path, nominal_plan, q_feedback, rig.context)
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


def render_case(
    case: OracleCase,
    context: MpcCostContext,
    body_pos: np.ndarray | None,
    out_dir: Path,
) -> None:
    """Write the overlay still plus oracle and nominal playback videos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    viz = ArmVisualizer(fk=context.fk)
    oracle_aa = arm_aa_from_state(case.oracle_correction, context)
    nominal_aa = arm_aa_from_state(case.nominal_continuation, context)

    viz.render_oracle_overlay(
        out_dir / f"{case.label}_overlay.png",
        oracle_traj=oracle_aa,
        nominal_traj=nominal_aa,
        current_q=arm_aa_from_state(case.q_feedback, context),
        spine3_pos=context.spine3_pos,
        spine3_aa=context.spine3_aa,
        body_pos=body_pos,
        goal_pos=case.goal,
        title="\n".join(textwrap.wrap(f"{case.label} — {case.utterance}", width=110)),
    )
    # A bound on a rotation barely moves the wrist, so the positional overlay
    # alone cannot show those corrections; this traces both futures through the
    # bound that actually fired.
    render_hidden_bounds(
        case.user,
        context,
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
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=body_pos,
            cartesian_goal=case.goal,
            frame_colors=[color] * len(traj),
            fps=12,
        )
