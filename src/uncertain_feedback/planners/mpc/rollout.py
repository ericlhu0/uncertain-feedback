"""Headless MPC rollout primitives shared by every pipeline stage.

The single stepping loop (:func:`run_planning_loop`) plus the goal-seeking rollouts
built on it: a comfort-only reference toward the original Cartesian goal, the full
corrected path shown to a cost generator, and the candidate-cost rollout closure the
cost evaluator scores. Nothing here knows about cost generation, LLMs, or simulated
users — the stages above import these, not the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
    MpcCostContext,
)
from uncertain_feedback.planners.mpc.goal_spaces import CartesianConfig
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.planners.mpc.mpc import ArmMPC


@dataclass
class LoopResult:
    """Joint configs visited by :func:`run_planning_loop`."""

    q_history: list[np.ndarray]
    error: str | None = None
    reached_goal: bool = False


StepHook = Callable[[int, np.ndarray, list[np.ndarray]], None]


def run_planning_loop(
    mpc: ArmMPC,
    q0: np.ndarray,
    n_steps: int,
    *,
    on_pre_step: StepHook | None = None,
    on_post_step: StepHook | None = None,
    stop_on_runtime_error: bool = False,
    stop_at_goal: bool = True,
    progress: bool = False,
    progress_desc: str = "MPC",
) -> LoopResult:
    """Step ``mpc`` forward up to ``n_steps``, returning the visited joint configs.

    This is the single stepping primitive shared by the live single run and the
    headless per-cluster experiment rollouts. The planner drives its own
    live/captured visualization (per the ``visualize``/``capture`` flags it was
    built with), so a live and a saved rollout share one rendering path.

    ``on_pre_step(step, q, q_history)`` runs before each ``mpc.step`` (the single
    run uses it to trigger MDM/LLM generation at ``text_time``); ``on_post_step``
    runs after (deferred LLM install, or per-step bookkeeping like frame colors).
    ``q_history`` holds the configs visited so far. With ``stop_on_runtime_error``
    a ``RuntimeError`` from ``mpc.step`` ends the loop and is recorded on the
    result instead of propagating.

    With ``stop_at_goal`` (the default), the loop ends as soon as the planner
    reports it has reached its final goal (``mpc.goal_reached``) and any MDM
    correction has finished playing (``mpc.mdm_ready_to_terminate``), rather than
    always running the full ``n_steps`` and idling at the goal. ``n_steps`` is
    therefore an upper bound. ``LoopResult.reached_goal`` records whether the loop
    stopped this way.

    Each ``mpc.step`` realizes its commanded configuration through the
    planner's execution env, so ``q_history`` records achieved configurations.
    """
    q = np.asarray(q0, dtype=np.float64).copy()
    q_history: list[np.ndarray] = []
    iterator: Iterable[int] = range(n_steps)
    if progress:
        from tqdm import (  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel
            tqdm,
        )

        iterator = tqdm(iterator, desc=progress_desc, unit="step")
    error: str | None = None
    reached_goal = False
    for step in iterator:
        if on_pre_step is not None:
            on_pre_step(step, q, q_history)
        try:
            q = mpc.step(q)
        except RuntimeError as exc:
            if not stop_on_runtime_error:
                raise
            error = str(exc)
            break
        q_history.append(q.copy())
        if on_post_step is not None:
            on_post_step(step, q, q_history)
        # Stop once the goal is reached (and any correction has finished), so the
        # rollout doesn't idle at the goal for the remaining step budget.
        if stop_at_goal and mpc.mdm_ready_to_terminate and mpc.goal_reached(q):
            reached_goal = True
            break
    return LoopResult(q_history=q_history, error=error, reached_goal=reached_goal)


def rollout_reference_trajectory(
    cfg: MpcRunConfig,
    current_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    on_step: Callable[[np.ndarray, np.ndarray | None], None] | None = None,
) -> np.ndarray | None:
    """Roll the MPC toward its original Cartesian goal, ignoring the correction.

    Builds a headless goal-space-only :class:`ArmMPC` from ``current_q`` carrying only
    the configured comfort costs (no feedback correction, no LLM-generated cost) and
    steps it toward ``cfg.cartesian.goals`` so the cost generator can see what the arm
    was driving toward before the correction — and avoid blocking it. With no feedback
    phase (``mdm_ready_to_terminate`` is always ``True``) the loop stops as soon as the
    wrist reaches the goal, so the trajectory ends at the goal rather than idling there
    for the full ``cfg.steps``. Returns ``(T, 7)``, or ``None`` without a persistent
    Cartesian goal.
    """
    if cfg.cartesian is None:
        return None

    planner = ArmMPC(
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=False,
        fk=context.fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        body_pos=body_pos,
        extra_costs=base_extra_costs,
        seed=cfg.seed,
        initial_q=current_q,
        cartesian=cfg.cartesian,
    )
    q0 = np.asarray(current_q, dtype=np.float64).copy()
    result = run_planning_loop(
        planner,
        q0,
        max(1, cfg.steps),
        on_post_step=(
            None if on_step is None else lambda _step, q, _history: on_step(q, None)
        ),
        stop_on_runtime_error=True,
    )
    return np.asarray([q0, *result.q_history], dtype=np.float64)


def assemble_full_correction_traj(
    cfg: MpcRunConfig,
    q_history: list[np.ndarray],
    correction_traj: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> np.ndarray:
    """Assemble the entire corrected path: history → correction → goal continuation.

    This is the target shown (green) in the cost-feedback comparison so the cost
    generator sees the whole intended trajectory, not just the MDM correction
    segment. The three segments are the executed pre-correction history, the MDM
    correction itself, and a comfort-only goal-seeking continuation rolled from the
    correction's endpoint (so the arm still reaches the goal afterwards). The
    continuation is empty for planners without a Cartesian goal, leaving just
    history + correction. The duplicated seam frame at the correction endpoint is
    dropped.
    """
    correction_traj = np.asarray(correction_traj, dtype=np.float64)
    if correction_traj.shape[-2:] == (3, 3):
        correction_traj = context.fk.arm_aa_to_q_batch(correction_traj, spine3_aa)
    segments: list[np.ndarray] = []
    if q_history:
        segments.append(np.asarray(q_history, dtype=np.float64))
    segments.append(correction_traj)
    post = rollout_reference_trajectory(
        cfg,
        correction_traj[-1],
        context,
        base_extra_costs,
        body_pos,
        spine3_pos,
        spine3_aa,
    )
    if post is not None and len(post) > 1:
        segments.append(post[1:])
    return np.concatenate(segments, axis=0)


def make_cost_eval_rollout(
    cfg: MpcRunConfig,
    current_q: np.ndarray,
    context: MpcCostContext,
    base_extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
) -> Callable[[GeneratedPythonCost], np.ndarray | None]:
    """Return a closure rolling the goal-seeking MPC with a candidate cost installed.

    The returned function appends the candidate generated cost to the comfort costs
    and rolls toward the original Cartesian goal (reusing
    :func:`rollout_reference_trajectory`), yielding the ``(T, 3, 3)`` trajectory the
    cost evaluator compares against the MDM correction. Returns ``None`` for planners
    without a persistent Cartesian goal. Each call builds a fresh headless planner, so
    the live MPC's goals/warm-start are untouched.
    """

    def rollout(cost: GeneratedPythonCost) -> np.ndarray | None:
        extra = CompositeTrajectoryCost([*base_extra_costs.terms(), cost])
        rollout_q = rollout_reference_trajectory(
            cfg, current_q, context, extra, body_pos, spine3_pos, spine3_aa
        )
        if rollout_q is None:
            return None
        return q_to_arm_aa(rollout_q, context.fk.elbow_hinge_axis)

    return rollout


def rollout_to_goal(
    cfg: MpcRunConfig,
    q0: np.ndarray,
    goal: np.ndarray,
    context: MpcCostContext,
    extra_costs: CompositeTrajectoryCost,
    body_pos: np.ndarray | None,
    spine3_pos: np.ndarray | None,
    spine3_aa: np.ndarray | None,
    *,
    steps: int | None = None,
    stop_at_goal: bool = True,
    progress_label: str | None = None,
    log_prefix: str = "[experiment]",
) -> np.ndarray:
    """Roll a headless Cartesian MPC from ``q0`` toward one goal.

    ``steps`` overrides ``cfg.steps`` and ``stop_at_goal=False`` forces the
    full step budget (the episode loop's fixed-length nominal plan).
    """
    assert cfg.cartesian is not None
    planner = ArmMPC(
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=False,
        fk=context.fk,
        spine3_pos=spine3_pos,
        spine3_aa=spine3_aa,
        body_pos=body_pos,
        extra_costs=extra_costs,
        seed=cfg.seed,
        initial_q=q0,
        cartesian=CartesianConfig(
            goals=[list(np.asarray(goal, dtype=np.float64))],
            threshold=cfg.cartesian.threshold,
        ),
    )
    q0 = np.asarray(q0, dtype=np.float64).copy()
    n_steps = max(1, cfg.steps if steps is None else steps)

    def _progress(step: int, _q: np.ndarray, _q_history: list[np.ndarray]) -> None:
        if progress_label is not None and (step + 1) % 50 == 0:
            print(
                f"{log_prefix} {progress_label}: step {step + 1}/{n_steps}", flush=True
            )

    result = run_planning_loop(
        planner,
        q0,
        n_steps,
        on_post_step=_progress if progress_label is not None else None,
        stop_on_runtime_error=True,
        stop_at_goal=stop_at_goal,
    )
    return np.asarray([q0, *result.q_history], dtype=np.float64)


def goal_reach(
    context: MpcCostContext,
    cfg: MpcRunConfig,
    rollout: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    """Final spine3-relative wrist distance to ``goal``."""
    final = np.asarray(rollout[-1], dtype=np.float64)
    final_aa = (
        q_to_arm_aa(final, context.fk.elbow_hinge_axis)
        if final.shape == (7,)
        else final
    )
    arm_pos = context.fk.fk(
        final_aa,
        context.spine3_pos,
        context.spine3_aa,
    )
    wrist_rel = arm_pos[-1] - context.spine3_pos
    distance = float(np.linalg.norm(wrist_rel - np.asarray(goal, dtype=np.float64)))
    assert cfg.cartesian is not None
    return {
        "reached": distance < cfg.cartesian.threshold,
        "distance": distance,
        "threshold": float(cfg.cartesian.threshold),
    }
