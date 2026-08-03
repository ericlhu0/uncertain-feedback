"""JSON summaries and overlay images that ground a cost-generation prompt.

Everything the prompt layer needs to describe the correction numerically
(:func:`build_motion_summaries`), visually (:func:`render_prompt_images`), and — for
the multi-turn backend — to report how a scored rollout diverged from the target
(:func:`build_rollout_joint_comparison`). Read-only over
:class:`GeneratedCostContext`; nothing here compiles or scores a cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.planners.mpc.costs import GeneratedCostContext
from uncertain_feedback.utils.plot import ArmVisualizer

# The FK chain in position order; indices match ``context.*_positions`` rows.
_ARM_CHAIN_JOINTS = (
    ("spine3", 0),
    ("left_collar", 1),
    ("left_shoulder", 2),
    ("left_elbow", 3),
    ("left_wrist", 4),
)


def build_motion_summaries(
    context: GeneratedCostContext,
    cartesian_goal: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable current/recent/MDM trajectory summaries.

    When ``context.reference_traj`` is set, a ``"reference"`` summary of the
    original-goal rollout (the path the arm was taking before the correction,
    including its endpoint joint posture) is included. When ``cartesian_goal``
    (the spine3-relative wrist target) is given it is added under
    ``"cartesian_goal"``. When explicitly marked-wrong candidate trajectories are
    available, a rollout-labeled terminal joint-feature comparison is included under
    ``"candidate_comparison"``.
    """
    mdm_positions = context.mdm_positions
    current_positions = context.current_positions
    recent_positions = context.recent_positions
    spine3_pos = context.spine3_pos
    summaries: dict[str, Any] = {
        "current": _state_summary(
            context, context.current_q, current_positions, spine3_pos
        ),
        "mdm_traj": _trajectory_summary(
            context, context.mdm_traj, mdm_positions, spine3_pos
        ),
    }
    summaries["current"]["joint_features"] = _joint_feature_frame_summary(
        context,
        context.current_q,
    )
    summaries["mdm_traj"]["joint_features"] = _joint_feature_summary(
        context,
        context.mdm_traj,
    )
    if context.rejected_trajs:
        summaries["candidate_comparison"] = _candidate_comparison_summary(context)
    if context.recent_q.size > 0:
        summaries["recent"] = _trajectory_summary(
            context,
            context.recent_q,
            recent_positions,
            spine3_pos,
        )
        summaries["recent"]["joint_features"] = _joint_feature_summary(
            context,
            context.recent_q,
        )
    else:
        summaries["recent"] = {}
    if context.reference_traj is not None and context.reference_traj.size > 0:
        summaries["reference"] = _trajectory_summary(
            context,
            context.reference_traj,
            context.reference_positions,
            spine3_pos,
        )
        summaries["reference"]["joint_features"] = _joint_feature_summary(
            context,
            context.reference_traj,
        )
    if cartesian_goal is not None:
        summaries["cartesian_goal"] = np.asarray(
            cartesian_goal, dtype=np.float64
        ).tolist()
    return summaries


def render_prompt_images(
    context: GeneratedCostContext,
    output_dir: Path,
    candidate_trajs: dict[int, np.ndarray] | None = None,
    highlight_label: int | None = None,
    reference_traj: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
) -> dict[str, Path]:
    """Render the per-purpose overlay images grounding the LLM cost prompt.

    Produces up to three separately-readable images, each showing full-arm motion
    and keyed by the prompt placeholder that requests it:

    - ``current_cluster_traj_img``: only the chosen cluster's arm (always rendered).
    - ``other_clusters_traj_img``: the chosen cluster's motion alongside every
      marked-wrong candidate's terminal arm pose (rendered only when
      ``candidate_trajs`` has more than one cluster).
    - ``reference_traj_img``: the chosen cluster alongside the original-goal reference
      arm (rendered only when ``reference_traj`` is given).

    The gold goal star and orange current pose appear in every image. When
    ``candidate_trajs`` is ``None`` the highlighted path is ``context.mdm_traj``.
    Returns ``{placeholder: path}`` for only the images that were renderable, so the
    prompt layer can attach exactly the subset its template asks for.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_trajs = candidate_trajs if candidate_trajs else {0: context.mdm_traj}
    trajs = {label: context.arm_aa(traj) for label, traj in raw_trajs.items()}
    reference_aa = (
        context.arm_aa(reference_traj) if reference_traj is not None else None
    )
    label = highlight_label if highlight_label is not None else next(iter(trajs))
    visualizer = ArmVisualizer(context.fk)

    def _render(
        filename: str, *, include_others: bool, include_reference: bool
    ) -> Path:
        path = output_dir / filename
        visualizer.render_cluster_contrast_overlay(
            path,
            mdm_trajs=trajs,
            highlight_label=label,
            current_q=context.arm_aa(context.current_q),
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=context.body_pos,
            reference_traj=reference_aa,
            goal_pos=goal_pos,
            include_others=include_others,
            include_reference=include_reference,
        )
        return path

    images: dict[str, Path] = {
        "current_cluster_traj_img": _render(
            "current.png", include_others=False, include_reference=False
        )
    }
    if len(trajs) > 1:
        images["other_clusters_traj_img"] = _render(
            "others.png", include_others=True, include_reference=False
        )
    if reference_traj is not None:
        images["reference_traj_img"] = _render(
            "reference.png", include_others=False, include_reference=True
        )
    return images


def build_rollout_joint_comparison(
    context: GeneratedCostContext,
    rollout: np.ndarray,
) -> dict[str, Any]:
    """Return joint feature comparison between a scored rollout and the intended target.

    Uses ``context.full_correction_traj`` as the target when available (the full
    pre-correction + correction + goal-continuation path), falling back to
    ``context.mdm_traj``.
    """
    target_traj = (
        context.full_correction_traj
        if context.full_correction_traj is not None
        else context.mdm_traj
    )
    return {
        "rollout": _joint_feature_summary(
            context, np.asarray(rollout, dtype=np.float64)
        ),
        "target": _joint_feature_summary(context, target_traj),
    }


def _arm_state_frame_summary(
    context: GeneratedCostContext, trajectory: np.ndarray
) -> dict[str, Any]:
    q = context.canonical_q(trajectory)
    return {
        "clavicle": {
            "value": q[0:3].tolist(),
            "norm": float(np.linalg.norm(q[0:3])),
        },
        "shoulder": {
            "value": q[3:6].tolist(),
            "norm": float(np.linalg.norm(q[3:6])),
        },
        "elbow_flexion": float(q[6]),
    }


def _arm_state_trajectory_summary(
    context: GeneratedCostContext, trajectory: np.ndarray
) -> dict[str, Any]:
    t = context.canonical_q(trajectory)
    return {
        "clavicle": {
            "start": t[0, 0:3].tolist(),
            "end": t[-1, 0:3].tolist(),
            "norm_start": float(np.linalg.norm(t[0, 0:3])),
            "norm_end": float(np.linalg.norm(t[-1, 0:3])),
        },
        "shoulder": {
            "start": t[0, 3:6].tolist(),
            "end": t[-1, 3:6].tolist(),
            "norm_start": float(np.linalg.norm(t[0, 3:6])),
            "norm_end": float(np.linalg.norm(t[-1, 3:6])),
        },
        "elbow_flexion": {
            "start": float(t[0, 6]),
            "end": float(t[-1, 6]),
        },
    }


def _trajectory_summary(
    context: GeneratedCostContext,
    trajectory: np.ndarray,
    positions: np.ndarray,
    spine3_pos: np.ndarray,
) -> dict[str, Any]:
    return {
        "joint_angles": _array_stats(trajectory),
        "positions": _position_summary(positions, spine3_pos),
        "arm_state": _arm_state_trajectory_summary(context, trajectory),
    }


def _state_summary(
    context: GeneratedCostContext,
    q: np.ndarray,
    positions: np.ndarray,
    spine3_pos: np.ndarray,
) -> dict[str, Any]:
    return {
        "joint_angles": np.asarray(q, dtype=np.float64).tolist(),
        "positions": _position_frame_summary(positions, spine3_pos),
        "arm_state": _arm_state_frame_summary(context, q),
    }


def _joint_feature_summary(
    context: GeneratedCostContext,
    trajectory: np.ndarray,
) -> dict[str, Any]:
    return {
        name: _series_stats(values)
        for name, values in context.feature_series(trajectory).items()
    }


def _candidate_comparison_summary(
    context: GeneratedCostContext,
) -> dict[str, Any]:
    chosen = _joint_feature_summary(context, context.mdm_traj)
    current = _joint_feature_frame_summary(context, context.current_q)
    original = (
        _joint_feature_summary(context, context.reference_traj)
        if context.reference_traj is not None and context.reference_traj.size > 0
        else None
    )
    rejected = [
        _joint_feature_summary(context, trajectory)
        for trajectory in context.rejected_trajs
    ]
    comparison: dict[str, Any] = {}
    for feature, chosen_stats in chosen.items():
        rejected_ends = {
            f"rejected_cluster_{index}": float(stats[feature]["end"])
            for index, stats in enumerate(rejected)
        }
        values = np.asarray(list(rejected_ends.values()), dtype=np.float64)
        rejected_median = float(np.median(values))
        rejected_std = float(np.std(values))
        chosen_end = float(chosen_stats["end"])
        feature_comparison: dict[str, Any] = {
            "chosen_rollout": "mdm_traj",
            "chosen_end": chosen_end,
            "current_rollout": "current",
            "current_value": float(current[feature]),
            "chosen_minus_current": chosen_end - float(current[feature]),
            "rejected_ends": rejected_ends,
            "rejected_median": rejected_median,
            "rejected_std": rejected_std,
            "standardized_separation": (
                None
                if rejected_std <= 1e-12
                else float((chosen_end - rejected_median) / rejected_std)
            ),
        }
        if original is not None:
            original_end = float(original[feature]["end"])
            feature_comparison.update(
                {
                    "original_plan_rollout": "reference",
                    "original_plan_end": original_end,
                    "chosen_minus_original_plan": chosen_end - original_end,
                }
            )
        comparison[feature] = feature_comparison
    return comparison


def _joint_feature_frame_summary(
    context: GeneratedCostContext,
    q: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(np.asarray(values).reshape(-1)[0])
        for name, values in context.feature_series(q).items()
    }


def _position_summary(positions: np.ndarray, spine3_pos: np.ndarray) -> dict[str, Any]:
    return {
        name: _vector_series_stats(positions[:, idx] - spine3_pos)
        for name, idx in _ARM_CHAIN_JOINTS
    }


def _position_frame_summary(
    positions: np.ndarray, spine3_pos: np.ndarray
) -> dict[str, Any]:
    return {
        name: (positions[idx] - spine3_pos).tolist() for name, idx in _ARM_CHAIN_JOINTS
    }


def _series_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "start": float(values[0]),
        "end": float(values[-1]),
    }


def _vector_series_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "x": _series_stats(values[:, 0]),
        "y": _series_stats(values[:, 1]),
        "z": _series_stats(values[:, 2]),
        "start": values[0].tolist(),
        "end": values[-1].tolist(),
        "delta": (values[-1] - values[0]).tolist(),
    }


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "start": values[0].tolist(),
        "end": values[-1].tolist(),
    }
