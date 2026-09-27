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

from uncertain_feedback.planners.mpc.arm_features import arm_aa_from_state
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    GeneratedPythonCost,
)
from uncertain_feedback.planners.mpc.goal_spaces import CartesianConfig
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import WRIST_CHAIN_IDX
from uncertain_feedback.planners.mpc.mpc import ArmMPC


@dataclass
class LoopResult:
    """The person after :func:`run_planning_loop` stopped."""

    human: Human
    error: str | None = None
    reached_goal: bool = False


StepHook = Callable[[int, Human], None]


def run_planning_loop(
    mpc: ArmMPC,
    n_steps: int,
    *,
    on_pre_step: StepHook | None = None,
    on_post_step: StepHook | None = None,
    stop_on_runtime_error: bool = False,
    stop_at_goal: bool = True,
    progress: bool = False,
    progress_desc: str = "MPC",
) -> LoopResult:
    """Step ``mpc`` forward up to ``n_steps`` from the person it was built with.

    This is the single stepping primitive shared by the live single run and the
    headless per-cluster experiment rollouts. The planner drives its own
    live/captured visualization (per the ``visualize``/``capture`` flags it was
    built with), so a live and a saved rollout share one rendering path.

    ``on_pre_step(step, human)`` runs before each ``mpc.step`` (the single run
    uses it to trigger MDM/LLM generation at ``text_time``); ``on_post_step``
    runs after (deferred LLM install, or per-step bookkeeping like frame
    colors). ``human.history`` holds every state so far, start included. With
    ``stop_on_runtime_error`` a ``RuntimeError`` from ``mpc.step`` ends the loop
    and is recorded on the result instead of propagating.

    With ``stop_at_goal`` (the default), the loop ends as soon as the planner
    reports it has reached its final goal (``mpc.goal_reached``) and any MDM
    correction has finished playing (``mpc.mdm_ready_to_terminate``), rather than
    always running the full ``n_steps`` and idling at the goal. ``n_steps`` is
    therefore an upper bound. ``LoopResult.reached_goal`` records whether the loop
    stopped this way.

    Each ``mpc.step`` realizes its commanded configuration through the
    planner's execution env, so the history records achieved configurations.
    """
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
            on_pre_step(step, mpc.human)
        try:
            human = mpc.step()
        except RuntimeError as exc:
            if not stop_on_runtime_error:
                raise
            error = str(exc)
            break
        if on_post_step is not None:
            on_post_step(step, human)
        # Stop once the goal is reached (and any correction has finished), so the
        # rollout doesn't idle at the goal for the remaining step budget.
        if stop_at_goal and mpc.mdm_ready_to_terminate and mpc.goal_reached(human.q):
            reached_goal = True
            break
    return LoopResult(human=mpc.human, error=error, reached_goal=reached_goal)


def rollout_reference_trajectory(
    cfg: MpcRunConfig,
    human: Human,
    base_extra_costs: CompositeTrajectoryCost,
    on_step: Callable[[np.ndarray, np.ndarray | None], None] | None = None,
) -> Human | None:
    """Roll the MPC toward its original Cartesian goal, ignoring the correction.

    Builds a headless goal-space-only :class:`ArmMPC` from ``human``'s current
    arm state carrying only the configured comfort costs (no feedback
    correction, no LLM-generated cost) and steps it toward
    ``cfg.cartesian.goals`` so the cost generator can see what the arm was
    driving toward before the correction — and avoid blocking it. With no
    feedback phase (``mdm_ready_to_terminate`` is always ``True``) the loop stops
    as soon as the wrist reaches the goal, so the trajectory ends at the goal
    rather than idling there for the full ``cfg.steps``. Returns the person
    whose ``history`` is the rollout alone, starting at ``human.q``, or ``None``
    without a persistent Cartesian goal.
    """
    if cfg.cartesian is None:
        return None

    planner = ArmMPC(
        human.reset_human_with_q(human.q),
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=False,
        extra_costs=base_extra_costs,
        seed=cfg.seed,
        cartesian=cfg.cartesian,
    )
    result = run_planning_loop(
        planner,
        max(1, cfg.steps),
        on_post_step=(
            None if on_step is None else lambda _step, moved: on_step(moved.q, None)
        ),
        stop_on_runtime_error=True,
    )
    return result.human


def assemble_full_correction_traj(
    cfg: MpcRunConfig,
    human: Human,
    correction_traj: np.ndarray,
    base_extra_costs: CompositeTrajectoryCost,
) -> np.ndarray:
    """Assemble the entire corrected path: history → correction → goal continuation.

    This is the target shown (green) in the cost-feedback comparison so the cost
    generator sees the whole intended trajectory, not just the MDM correction
    segment. The three segments are ``human``'s executed history, the MDM
    correction itself, and a comfort-only goal-seeking continuation rolled from the
    correction's endpoint (so the arm still reaches the goal afterwards). The
    continuation is empty for planners without a Cartesian goal, leaving just
    history + correction. The duplicated seam frame at the correction endpoint is
    dropped.
    """
    correction_traj = np.asarray(correction_traj, dtype=np.float64)
    if correction_traj.shape[-2:] == (3, 3):
        correction_traj = human.q_from_arm_aa(correction_traj)
    segments = [human.history, correction_traj]
    post = rollout_reference_trajectory(
        cfg, human.reset_human_with_q(correction_traj[-1]), base_extra_costs
    )
    if post is not None and len(post.history) > 1:
        segments.append(post.history[1:])
    return np.concatenate(segments, axis=0)


def make_cost_eval_rollout(
    cfg: MpcRunConfig,
    human: Human,
    base_extra_costs: CompositeTrajectoryCost,
) -> Callable[[GeneratedPythonCost], np.ndarray | None]:
    """Return a closure rolling the goal-seeking MPC with a candidate cost installed.

    The returned function appends the candidate generated cost to the comfort costs
    and rolls toward the original Cartesian goal from ``human``'s current arm state
    (reusing :func:`rollout_reference_trajectory`), yielding the ``(T, 3, 3)``
    trajectory the cost evaluator compares against the MDM correction. Returns
    ``None`` for planners without a persistent Cartesian goal. Each call builds a
    fresh headless planner, so the live MPC's goals/warm-start are untouched.
    """

    def rollout(cost: GeneratedPythonCost) -> np.ndarray | None:
        extra = CompositeTrajectoryCost([*base_extra_costs.terms(), cost])
        rolled = rollout_reference_trajectory(cfg, human, extra)
        if rolled is None:
            return None
        return human.arm_aa_from_q(rolled.history)

    return rollout


def rollout_to_goal(
    cfg: MpcRunConfig,
    human: Human,
    goal: np.ndarray,
    extra_costs: CompositeTrajectoryCost,
    *,
    steps: int | None = None,
    stop_at_goal: bool = True,
    progress_label: str | None = None,
    log_prefix: str = "[experiment]",
) -> Human:
    """Roll a headless Cartesian MPC from ``human``'s arm state toward one goal.

    Returns the person whose ``history`` is the rollout alone, starting at
    ``human.q``. ``steps`` overrides ``cfg.steps`` and ``stop_at_goal=False``
    forces the full step budget (the episode loop's fixed-length nominal plan).
    """
    assert cfg.cartesian is not None
    planner = ArmMPC(
        human.reset_human_with_q(human.q),
        horizon=cfg.horizon,
        n_mpc_samples=cfg.n_mpc_samples,
        max_angle_delta=cfg.max_angle_delta,
        visualize=False,
        extra_costs=extra_costs,
        seed=cfg.seed,
        cartesian=CartesianConfig(
            goals=[list(np.asarray(goal, dtype=np.float64))],
            threshold=cfg.cartesian.threshold,
        ),
    )
    n_steps = max(1, cfg.steps if steps is None else steps)

    def _progress(step: int, _human: Human) -> None:
        if progress_label is not None and (step + 1) % 50 == 0:
            print(
                f"{log_prefix} {progress_label}: step {step + 1}/{n_steps}", flush=True
            )

    result = run_planning_loop(
        planner,
        n_steps,
        on_post_step=_progress if progress_label is not None else None,
        stop_on_runtime_error=True,
        stop_at_goal=stop_at_goal,
    )
    return result.human


def goal_reach(
    human: Human,
    cfg: MpcRunConfig,
    rollout: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    """Final spine3-relative wrist distance to ``goal``."""
    final = np.asarray(rollout[-1], dtype=np.float64)
    final_aa = human.arm_aa_from_q(final) if final.shape == (7,) else final
    arm_pos = human.fk.fk(final_aa, human.spine3_pos, human.spine3_aa)
    wrist_rel = arm_pos[-1] - human.spine3_pos
    distance = float(np.linalg.norm(wrist_rel - np.asarray(goal, dtype=np.float64)))
    assert cfg.cartesian is not None
    return {
        "reached": distance < cfg.cartesian.threshold,
        "distance": distance,
        "threshold": float(cfg.cartesian.threshold),
    }


def wrist_goal_distances(
    human: Human,
    trajectory: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Per-frame spine3-relative wrist distance to ``goal``, for q or axis-angle states."""
    arm_aa = arm_aa_from_state(np.asarray(trajectory, dtype=np.float64), human)
    positions = human.fk.fk_batch(arm_aa, human.spine3_pos, human.spine3_aa)
    wrist_rel = positions[:, WRIST_CHAIN_IDX, :] - human.spine3_pos
    return np.asarray(
        np.linalg.norm(wrist_rel - np.asarray(goal, dtype=np.float64), axis=-1),
        dtype=np.float64,
    )
