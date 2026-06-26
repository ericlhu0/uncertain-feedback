"""Shared primitives for generated Python trajectory costs.

These are provider- and strategy-agnostic building blocks reused by every cost
generator (single-turn ``llm``, multi-turn ``turns``, and the ``codex`` agent):
the runtime context handed to generated code, the executable cost wrapper, the
JSON response parser/compiler, trajectory summaries, and overlay rendering.

``llm_costs`` re-exports these names for backward compatibility, so existing
imports from ``...costs.llm_costs`` keep working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.costs.base import (
    MpcCostContext,
    TrajectoryCost,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK
from uncertain_feedback.utils.plot import ArmVisualizer

# Canonical names first (used for summaries); short aliases follow.
_JOINT_NAMES = {
    "spine3": 0,
    "left_collar": 1,
    "left_shoulder": 2,
    "left_elbow": 3,
    "left_wrist": 4,
    "collar": 1,
    "shoulder": 2,
    "elbow": 3,
    "wrist": 4,
}


class GeneratedCostValidationError(ValueError):
    """Raised when generated cost code or parameters are unsafe/invalid."""


@dataclass(frozen=True)
class LlmCostResponse:
    """Parsed response from the cost-generator LLM."""

    description: str
    code: str
    params: dict[str, Any]
    explanation: str = ""
    recipient_explanation: str = ""


@dataclass(frozen=True)
class GeneratedCostContext:
    """Read-only runtime context exposed to generated cost code."""

    fk: SmplLeftArmFK
    spine3_pos: np.ndarray
    spine3_aa: np.ndarray
    current_q: np.ndarray
    mdm_traj: np.ndarray
    recent_q: np.ndarray
    body_pos: np.ndarray | None = None
    reference_traj: np.ndarray | None = None

    @property
    def current_positions(self) -> np.ndarray:
        """Current arm-chain positions with shape ``(5, 3)``."""
        return self.fk.fk(self.current_q, self.spine3_pos, self.spine3_aa)

    @property
    def mdm_positions(self) -> np.ndarray:
        """Generated arm-chain positions with shape ``(T, 5, 3)``."""
        return self.fk_batch(self.mdm_traj)

    @property
    def reference_positions(self) -> np.ndarray:
        """Original-goal reference arm-chain positions with shape ``(T, 5, 3)``.

        Empty ``(0, 5, 3)`` when no reference trajectory is available.
        """
        if self.reference_traj is None:
            return np.empty((0, 5, 3), dtype=np.float64)
        return self.fk_batch(self.reference_traj)

    @property
    def recent_positions(self) -> np.ndarray:
        """Recent executed arm-chain positions with shape ``(T, 5, 3)``."""
        if self.recent_q.size == 0:
            return np.empty((0, 5, 3), dtype=np.float64)
        return self.fk_batch(self.recent_q)

    def fk_batch(self, trajectory: np.ndarray) -> np.ndarray:
        """Return arm-chain positions for ``(..., 3, 3)`` axis-angle frames."""
        trajectory = np.asarray(trajectory, dtype=np.float64)
        leading = trajectory.shape[:-2]
        flat = trajectory.reshape((-1, 3, 3))
        positions = self.fk.fk_batch(flat, self.spine3_pos, self.spine3_aa)
        return positions.reshape((*leading, 5, 3))

    def fk_rollouts(self, q_trajs: np.ndarray) -> np.ndarray:
        """Return arm-chain positions for rollout states."""
        return self.fk_batch(q_trajs)

    def joint_index(self, name: str) -> int:
        """Return arm-chain joint index for a known joint name."""
        try:
            return _JOINT_NAMES[name]
        except KeyError as exc:
            raise KeyError(f"Unknown generated-cost joint name: {name!r}") from exc

    def elbow_flexion_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return elbow bend as local elbow rotation-vector magnitude.

        Accepts any leading shape ending in ``(3, 3)`` and returns that leading
        shape. This is a coarse SMPL-space flexion proxy, not a clinical joint
        angle decomposition.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        return np.linalg.norm(trajectory[..., 1, :], axis=-1)

    def shoulder_abduction_adduction_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return signed shoulder abduction/adduction proxy in the spine3 frame.

        This is the lateral component angle of the shoulder-to-elbow direction:
        positive values move toward ``+x`` (left-arm abduction / away from the
        torso), and negative values move toward ``-x`` (adduction / across the
        torso).
        """
        upper_arm = self._upper_arm_direction_spine_frame(trajectory)
        return np.arcsin(np.clip(upper_arm[..., 0], -1.0, 1.0))

    def shoulder_flexion_extension_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return signed shoulder flexion/extension proxy in the spine3 frame.

        This is the depth component angle of the shoulder-to-elbow direction:
        positive values move toward ``+z`` and negative values move toward
        ``-z``.
        """
        upper_arm = self._upper_arm_direction_spine_frame(trajectory)
        return np.arcsin(np.clip(upper_arm[..., 2], -1.0, 1.0))

    def shoulder_internal_external_rotation_angles(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """Return signed shoulder twist around the T-pose upper-arm axis.

        This is an approximate internal/external rotation proxy from the local
        shoulder axis-angle. Positive sign follows the T-pose shoulder-to-elbow
        axis convention.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        leading = trajectory.shape[:-2]
        shoulder_rotvec = trajectory[..., 0, :].reshape(-1, 3)
        axis = self._tpose_upper_arm_axis()
        angles = _twist_angles_about_axis(shoulder_rotvec, axis)
        return angles.reshape(leading)

    def _upper_arm_direction_spine_frame(self, trajectory: np.ndarray) -> np.ndarray:
        """Return unit shoulder-to-elbow directions in the spine3 frame."""
        positions = self.fk_batch(trajectory)
        upper_arm_world = (
            positions[..., _JOINT_NAMES["left_elbow"], :]
            - positions[..., _JOINT_NAMES["left_shoulder"], :]
        )
        leading = upper_arm_world.shape[:-1]
        spine_inv = Rotation.from_rotvec(self.spine3_aa).inv()
        upper_arm_local = spine_inv.apply(upper_arm_world.reshape(-1, 3)).reshape(
            (*leading, 3)
        )
        norms = np.linalg.norm(upper_arm_local, axis=-1, keepdims=True)
        return np.divide(
            upper_arm_local,
            norms,
            out=np.zeros_like(upper_arm_local),
            where=norms > 1e-12,
        )

    def _tpose_upper_arm_axis(self) -> np.ndarray:
        """Return unit T-pose shoulder-to-elbow axis in the spine3 frame."""
        tpose = self.fk.tpose_joints
        axis = tpose[_JOINT_NAMES["left_elbow"]] - tpose[_JOINT_NAMES["left_shoulder"]]
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm


def _twist_angles_about_axis(rotvecs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Return signed twist component of rotations about a unit axis."""
    rotvecs = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / axis_norm

    quats = Rotation.from_rotvec(rotvecs).as_quat()  # (x, y, z, w)
    vec = quats[:, :3]
    w = quats[:, 3]
    projected_vec = axis[np.newaxis, :] * (vec @ axis)[:, np.newaxis]
    twist_norm = np.sqrt(np.sum(projected_vec**2, axis=1) + w**2)
    safe_vec = np.divide(
        projected_vec,
        twist_norm[:, np.newaxis],
        out=np.zeros_like(projected_vec),
        where=twist_norm[:, np.newaxis] > 1e-12,
    )
    safe_w = np.divide(w, twist_norm, out=np.ones_like(w), where=twist_norm > 1e-12)
    signed_vec = safe_vec @ axis
    angles = 2.0 * np.arctan2(signed_vec, safe_w)
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class GeneratedPythonCost(TrajectoryCost):
    """Executable LLM-generated trajectory cost."""

    code: str
    params: dict[str, Any]
    context: GeneratedCostContext
    description: str = ""

    def __post_init__(self) -> None:
        func = compile_generated_cost(self.code)
        object.__setattr__(self, "_func", func)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        q_trajs = np.asarray(q_trajs, dtype=np.float64)
        raw = self._func(q_trajs, self.context, self.params)  # type: ignore[attr-defined]
        costs = np.asarray(raw, dtype=np.float64)
        expected_shape = (q_trajs.shape[0],)
        if costs.shape != expected_shape:
            raise GeneratedCostValidationError(
                "generated cost must return shape "
                f"{expected_shape}, got {costs.shape}"
            )
        if not np.all(np.isfinite(costs)):
            raise GeneratedCostValidationError(
                "generated cost returned non-finite values"
            )
        return costs


def parse_llm_cost_response(raw: str) -> LlmCostResponse:
    """Parse the LLM JSON response, accepting optional Markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise GeneratedCostValidationError("LLM response is not JSON") from exc
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise GeneratedCostValidationError("LLM response must be a JSON object")
    description = data.get("description", "")
    code = data.get("code")
    params = data.get("params", {})
    explanation = data.get("explanation", "")
    recipient_explanation = data.get("recipient_explanation", "")
    if not isinstance(description, str):
        raise GeneratedCostValidationError("description must be a string")
    if not isinstance(code, str):
        raise GeneratedCostValidationError("code must be a string")
    if not isinstance(params, dict):
        raise GeneratedCostValidationError("params must be an object")
    if not isinstance(explanation, str):
        raise GeneratedCostValidationError("explanation must be a string")
    if not isinstance(recipient_explanation, str):
        raise GeneratedCostValidationError("recipient_explanation must be a string")
    return LlmCostResponse(
        description,
        code,
        params,
        explanation,
        recipient_explanation,
    )


def compile_generated_cost(code: str) -> FunctionType:
    """Compile and exec generated Python cost source."""
    namespace: dict[str, Any] = {"np": np}
    locals_dict: dict[str, Any] = {}
    exec(compile(code, "<llm_generated_cost>", "exec"), namespace, locals_dict)  # pylint: disable=exec-used
    func = locals_dict.get("cost")
    if not isinstance(func, FunctionType):
        raise GeneratedCostValidationError("generated code must define cost")
    return func


def build_generated_cost_context(
    mpc_context: MpcCostContext,
    current_q: np.ndarray,
    mdm_traj: np.ndarray,
    q_history: list[np.ndarray],
    window: int,
    body_pos: np.ndarray | None = None,
    reference_traj: np.ndarray | None = None,
) -> GeneratedCostContext:
    """Build the runtime context passed to generated Python costs."""
    recent_q = np.asarray(q_history[-window:], dtype=np.float64)
    if recent_q.size == 0:
        recent_q = np.empty((0, 3, 3), dtype=np.float64)
    return GeneratedCostContext(
        fk=mpc_context.fk,
        spine3_pos=np.asarray(mpc_context.spine3_pos, dtype=np.float64),
        spine3_aa=np.asarray(mpc_context.spine3_aa, dtype=np.float64),
        current_q=np.asarray(current_q, dtype=np.float64),
        mdm_traj=np.asarray(mdm_traj, dtype=np.float64),
        recent_q=recent_q,
        body_pos=np.asarray(body_pos, dtype=np.float64) if body_pos is not None else None,
        reference_traj=(
            np.asarray(reference_traj, dtype=np.float64)
            if reference_traj is not None
            else None
        ),
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
    ``"cartesian_goal"``. Both are omitted otherwise, leaving summaries unchanged.
    """
    mdm_positions = context.mdm_positions
    current_positions = context.current_positions
    recent_positions = context.recent_positions
    spine3_pos = context.spine3_pos
    summaries: dict[str, Any] = {
        "current": _state_summary(context.current_q, current_positions, spine3_pos),
        "mdm_traj": _trajectory_summary(context.mdm_traj, mdm_positions, spine3_pos),
    }
    summaries["current"]["joint_features"] = _joint_feature_frame_summary(
        context,
        context.current_q,
    )
    summaries["mdm_traj"]["joint_features"] = _joint_feature_summary(
        context,
        context.mdm_traj,
    )
    if context.recent_q.size > 0:
        summaries["recent"] = _trajectory_summary(
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
    - ``other_clusters_traj_img``: the chosen cluster alongside every other candidate
      cluster's arm (rendered only when ``candidate_trajs`` has more than one cluster).
    - ``reference_traj_img``: the chosen cluster alongside the original-goal reference
      arm (rendered only when ``reference_traj`` is given).

    The gold goal star and orange current pose appear in every image. When
    ``candidate_trajs`` is ``None`` the highlighted path is ``context.mdm_traj``.
    Returns ``{placeholder: path}`` for only the images that were renderable, so the
    prompt layer can attach exactly the subset its template asks for.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trajs = candidate_trajs if candidate_trajs else {0: context.mdm_traj}
    label = highlight_label if highlight_label is not None else next(iter(trajs))
    visualizer = ArmVisualizer(context.fk)

    def _render(filename: str, *, include_others: bool, include_reference: bool) -> Path:
        path = output_dir / filename
        visualizer.render_cluster_contrast_overlay(
            path,
            mdm_trajs=trajs,
            highlight_label=label,
            current_q=context.current_q,
            spine3_pos=context.spine3_pos,
            spine3_aa=context.spine3_aa,
            body_pos=context.body_pos,
            reference_traj=reference_traj,
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


def _trajectory_summary(
    trajectory: np.ndarray,
    positions: np.ndarray,
    spine3_pos: np.ndarray,
) -> dict[str, Any]:
    return {
        "joint_angles": _array_stats(trajectory),
        "positions": _position_summary(positions, spine3_pos),
    }


def _state_summary(
    q: np.ndarray,
    positions: np.ndarray,
    spine3_pos: np.ndarray,
) -> dict[str, Any]:
    return {
        "joint_angles": np.asarray(q, dtype=np.float64).tolist(),
        "positions": _position_frame_summary(positions, spine3_pos),
    }


def _joint_feature_summary(
    context: GeneratedCostContext,
    trajectory: np.ndarray,
) -> dict[str, Any]:
    return {
        "elbow_flexion": _series_stats(context.elbow_flexion_angles(trajectory)),
        "shoulder_flexion_extension": _series_stats(
            context.shoulder_flexion_extension_angles(trajectory)
        ),
        "shoulder_abduction_adduction": _series_stats(
            context.shoulder_abduction_adduction_angles(trajectory)
        ),
        "shoulder_internal_external_rotation": _series_stats(
            context.shoulder_internal_external_rotation_angles(trajectory)
        ),
    }


def _joint_feature_frame_summary(
    context: GeneratedCostContext,
    q: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(np.asarray(values).reshape(-1)[0])
        for name, values in {
            "elbow_flexion": context.elbow_flexion_angles(q),
            "shoulder_flexion_extension": (
                context.shoulder_flexion_extension_angles(q)
            ),
            "shoulder_abduction_adduction": (
                context.shoulder_abduction_adduction_angles(q)
            ),
            "shoulder_internal_external_rotation": (
                context.shoulder_internal_external_rotation_angles(q)
            ),
        }.items()
    }


def _position_summary(positions: np.ndarray, spine3_pos: np.ndarray) -> dict[str, Any]:
    return {
        name: _vector_series_stats(positions[:, idx] - spine3_pos)
        for name, idx in list(_JOINT_NAMES.items())[:5]
    }


def _position_frame_summary(
    positions: np.ndarray, spine3_pos: np.ndarray
) -> dict[str, Any]:
    return {
        name: (positions[idx] - spine3_pos).tolist()
        for name, idx in list(_JOINT_NAMES.items())[:5]
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

