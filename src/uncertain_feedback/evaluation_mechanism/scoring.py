"""Score a generated cost against the preferences the user actually revealed.

This is the evaluation mechanism the cost generators use on their own output: rank a
candidate cost by how it orders the revealed preference pairs
(:func:`rank_candidate_cost`), measure how closely the cost's rollout tracks the
correction (:func:`evaluate_candidate_cost`), check the Cartesian goal is still
reached (:func:`goal_reach_report`), and render the rollout-vs-correction comparison
the ``agent`` backend inspects each turn (:func:`evaluate_and_render`).

Depends only on the planner layer — never on ``cost_generation``, which imports this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    canonical_arm_q,
    resample_equidistant,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostContext,
    GeneratedPythonCost,
)


@dataclass(frozen=True)
class CostRanking:
    """How a candidate cost orders the preferences the user actually revealed.

    ``rank_accuracy`` is the fraction of preference pairs the cost orders
    correctly; ``normalized_margin`` is the mean z-scored separation of the chosen
    correction below the alternatives (a scale-free tiebreak, not a gate).
    ``inert`` marks a cost that returned (near-)identical values for every
    trajectory — it never discriminates, so its ranking is vacuous.
    """

    rank_accuracy: float
    normalized_margin: float
    inert: bool
    costs: dict[str, float]

    @property
    def improves_original_plan(self) -> bool | None:
        """Whether the chosen correction costs strictly less than the original plan."""
        chosen = self.costs.get("chosen_correction")
        original = self.costs.get("original_plan")
        if chosen is None or original is None:
            return None
        return chosen < original

    @property
    def sort_key(self) -> tuple[float, float]:
        """Lower-is-better selection key: rank accuracy first, margin as tiebreak."""
        if self.inert:
            return (math.inf, math.inf)
        return (1.0 - self.rank_accuracy, -self.normalized_margin)

    def as_json(self) -> dict[str, Any]:
        """JSON-safe payload for score/rationale artifacts."""
        return {
            "rank_accuracy": self.rank_accuracy,
            "normalized_margin": self.normalized_margin,
            "inert": self.inert,
            "improves_original_plan": self.improves_original_plan,
            "costs": self.costs,
        }


def rank_candidate_cost(
    context: GeneratedCostContext, cost: GeneratedPythonCost
) -> CostRanking | None:
    """Evaluate a candidate cost by ranking consistency, not trajectory matching.

    The cost function itself is applied to the trajectories whose preference order
    the user revealed: the chosen correction (``mdm_traj``) must cost strictly less
    than the original plan the user interrupted (``reference_traj``) and every
    cluster the user explicitly marked undesirable (``rejected_trajs``).
    Any cost that captures the intent satisfies this; recreating the correction
    trajectory is not required. All trajectories are resampled to equidistant
    joint-space points first (timing is a pipeline artifact, not intent) and
    compared after z-normalization (generated costs have arbitrary scale).

    Returns ``None`` when the context has no comparison trajectories (no reference
    rollout and no marked-undesirable clusters), in which case callers fall back to
    the L2 rollout score.
    """
    chosen = canonical_arm_q(context.mdm_traj, context)
    if chosen.ndim != 2 or chosen.shape[0] == 0:
        return None
    trajs: dict[str, np.ndarray] = {}
    if context.reference_traj is not None and context.reference_traj.size > 0:
        trajs["original_plan"] = canonical_arm_q(context.reference_traj, context)
    for i, rejected in enumerate(context.rejected_trajs):
        trajs[f"rejected_cluster_{i}"] = canonical_arm_q(rejected, context)
    if not trajs:
        return None
    trajs["chosen_correction"] = chosen
    n = max(traj.shape[0] for traj in trajs.values())
    batch = np.stack([resample_equidistant(traj, n) for traj in trajs.values()])
    raw = np.asarray(cost(batch), dtype=np.float64)
    costs = {name: float(value) for name, value in zip(trajs, raw)}
    if not np.all(np.isfinite(raw)) or float(raw.std()) < 1e-12:
        return CostRanking(0.0, 0.0, True, costs)
    z = dict(zip(trajs, (raw - raw.mean()) / raw.std()))
    z_chosen = z["chosen_correction"]
    pairs: list[bool] = []
    margins: list[float] = []
    if "original_plan" in z:
        pairs.append(bool(z_chosen < z["original_plan"]))
        margins.append(float(z["original_plan"] - z_chosen))
    for name in trajs:
        if name.startswith("rejected_cluster_"):
            pairs.append(bool(z_chosen < z[name]))
            margins.append(float(z[name] - z_chosen))
    return CostRanking(
        rank_accuracy=float(np.mean(pairs)),
        normalized_margin=float(np.mean(margins)),
        inert=False,
        costs=costs,
    )


def _score_rollout(context: GeneratedCostContext, rollout: np.ndarray) -> float:
    """Mean per-frame FK L2 distance between a rollout and the MDM correction.

    Both trajectories are resampled to equidistant joint-space points
    (:func:`resample_equidistant`) so the comparison is purely about path — MDM
    output is systematically slower than a fresh rollout, so frame-wise timing is
    a pipeline artifact. ``math.inf`` when either trajectory is empty / malformed.
    """
    rollout = canonical_arm_q(rollout, context)
    target = canonical_arm_q(context.mdm_traj, context)
    if (
        rollout.ndim != 2
        or rollout.shape[0] == 0
        or target.ndim != 2
        or target.shape[0] == 0
    ):
        return math.inf
    n = max(rollout.shape[0], target.shape[0])
    rollout_positions = context.fk_batch(resample_equidistant(rollout, n))
    mdm_positions = context.fk_batch(resample_equidistant(target, n))
    return float(np.linalg.norm(rollout_positions - mdm_positions, axis=-1).mean())


def goal_reach_report(
    context: GeneratedCostContext, rollout: np.ndarray | None
) -> dict[str, Any] | None:
    """Whether a candidate rollout still reaches the Cartesian goal.

    Reproduces the MPC's own criterion (``ArmMPCCartesian.goal_reached``): forward-
    kinematics the final rollout frame, take the spine3-relative wrist, and compare its
    distance to ``context.cartesian_goal`` against ``context.cartesian_threshold``.
    Returns ``None`` when no Cartesian goal is available (non-Cartesian planners) or the
    rollout is empty/malformed, so callers degrade to no goal feedback.
    """
    if context.cartesian_goal is None or context.cartesian_threshold is None:
        return None
    if rollout is None:
        return None
    rollout = canonical_arm_q(rollout, context)
    if rollout.ndim != 2 or rollout.shape[0] == 0:
        return None
    arm_pos = context.fk.fk(
        context.arm_aa(rollout[-1]), context.spine3_pos, context.spine3_aa
    )
    wrist_rel = arm_pos[-1] - context.spine3_pos
    distance = float(np.linalg.norm(wrist_rel - context.cartesian_goal))
    return {
        "reached": distance < context.cartesian_threshold,
        "distance": distance,
        "threshold": float(context.cartesian_threshold),
    }


def evaluate_candidate_cost(
    context: GeneratedCostContext,
    cost: GeneratedPythonCost,
    rollout_fn: Callable[[GeneratedPythonCost], np.ndarray | None] | None = None,
) -> tuple[float, np.ndarray | None]:
    """Score a candidate cost; lower is better.

    Rolls the goal-seeking MPC toward its original goal with this cost installed
    (via ``rollout_fn``), resamples the result to the corrected (MDM) trajectory's
    length, and returns the mean per-frame Cartesian (FK) L2 distance to it plus the
    raw rollout trajectory. A cost that steers the goal-seeking motion to match the
    user's correction scores low.

    Returns ``(math.inf, None)`` when no rollout is available (``rollout_fn`` is
    ``None`` or yields no trajectory, e.g. planners without a persistent Cartesian
    goal).
    """
    if rollout_fn is None:
        return math.inf, None
    rollout = rollout_fn(cost)
    if rollout is None:
        return math.inf, None
    return _score_rollout(context, rollout), rollout


def evaluate_and_render(
    context: GeneratedCostContext,
    cost: GeneratedPythonCost,
    rollout_fn: Callable[[GeneratedPythonCost], np.ndarray | None] | None,
    image_path: Path,
    *,
    angle_path: Path | None = None,
    rollout_path: Path | None = None,
    video_path: Path | None = None,
) -> tuple[float, Path | None, np.ndarray | None]:
    """Score a candidate cost and render the rollout-vs-correction comparison.

    Rolls the goal-seeking MPC **once** with ``cost`` installed, computes the L2
    score (:func:`_score_rollout`), and renders ``image_path`` overlaying that
    rollout against the MDM correction
    (:meth:`ArmVisualizer.render_cost_feedback_overlay`). When ``angle_path`` is
    given, also renders a joint-angle-over-time comparison there
    (:meth:`ArmVisualizer.render_joint_angle_comparison`). Returns
    ``(score, image_path, rollout)``, or ``(math.inf, None, None)`` when no rollout
    is available (planners without a persistent Cartesian goal) so callers degrade
    to text-only feedback.
    """
    from uncertain_feedback.utils.plot import (  # pylint: disable=import-outside-toplevel
        ArmVisualizer,
    )

    if rollout_fn is None:
        return math.inf, None, None
    rollout = rollout_fn(cost)
    if rollout is None:
        return math.inf, None, None
    rollout = canonical_arm_q(rollout, context)
    if rollout.ndim != 2 or rollout.shape[0] == 0:
        return math.inf, None, None
    score = _score_rollout(context, rollout)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = ArmVisualizer(context.fk)
    correction_traj = reference_with_correction_traj(context)
    visualizer.render_cost_feedback_overlay(
        image_path,
        rollout_traj=context.arm_aa(rollout),
        correction_traj=context.arm_aa(correction_traj),
        current_q=context.arm_aa(context.current_q),
        spine3_pos=context.spine3_pos,
        spine3_aa=context.spine3_aa,
        body_pos=context.body_pos,
    )
    if angle_path is not None:
        angle_path.parent.mkdir(parents=True, exist_ok=True)
        reference_series = (
            _joint_angle_series(context, context.reference_traj)
            if context.reference_traj is not None and context.reference_traj.size > 0
            else None
        )
        visualizer.render_joint_angle_comparison(
            angle_path,
            target_series=_joint_angle_series(context, correction_traj),
            rollout_series=_joint_angle_series(context, rollout),
            reference_series=reference_series,
        )
    if rollout_path is not None:
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rollout_path, rollout)
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.render_rollout_video(
            context.arm_aa(rollout),
            video_path,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=context.body_pos,
        )
    return score, image_path, rollout


def reference_with_correction_traj(context: GeneratedCostContext) -> np.ndarray:
    """Return the target path that includes the user's correction."""
    if context.full_correction_traj is not None:
        return context.full_correction_traj
    return context.mdm_traj


def _joint_angle_series(
    context: GeneratedCostContext,
    trajectory: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the shared anatomical feature series for a trajectory."""
    return context.feature_series(trajectory)
