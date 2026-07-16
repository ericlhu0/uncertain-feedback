"""Shared primitives for generated Python trajectory costs.

These are provider- and strategy-agnostic building blocks reused by every cost
generator (single-turn ``llm``, multi-turn ``turns``, and the ``codex`` agent):
the runtime context handed to generated code, the executable cost wrapper, the
JSON response parser/compiler, trajectory summaries, and overlay rendering.

``llm_costs`` re-exports these names for backward compatibility, so existing
imports from ``...costs.llm_costs`` keep working.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.costs.base import (
    CompositeTrajectoryCost,
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

_NAMED_FEATURE_METHODS = {
    "elbow_flexion_angles": "elbow_flexion",
    "shoulder_flexion_extension_angles": "shoulder_flexion_extension",
    "shoulder_abduction_adduction_angles": "shoulder_abduction_adduction",
    "shoulder_elevation_angles": "shoulder_elevation",
    "shoulder_internal_external_rotation_angles": (
        "shoulder_internal_external_rotation"
    ),
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
    """Read-only runtime context exposed to generated cost code.

    ``rejected_trajs`` contains only candidates the person explicitly marked wrong.
    """

    fk: SmplLeftArmFK
    spine3_pos: np.ndarray
    spine3_aa: np.ndarray
    current_q: np.ndarray
    mdm_traj: np.ndarray
    recent_q: np.ndarray
    body_pos: np.ndarray | None = None
    reference_traj: np.ndarray | None = None
    full_correction_traj: np.ndarray | None = None
    cartesian_goal: np.ndarray | None = None
    cartesian_threshold: float | None = None
    rejected_trajs: tuple[np.ndarray, ...] = ()

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
        """Return elbow bend as the angle between upper arm and forearm.

        Accepts any leading shape ending in ``(3, 3)`` and returns that leading
        shape. 0 = fully extended; larger = more bent. Computed from the
        wrist-slot joint rotation — under this repo's FK convention (joint
        *j*'s rotation transforms the bone arriving at *j*) that slot rotates
        the forearm relative to the upper arm, so the bend is the angle it
        opens between the T-pose bone axes. Equivalent to the FK-position
        angle, without running FK.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        leading = trajectory.shape[:-2]
        flat = trajectory.reshape(-1, 3, 3)
        forearm = Rotation.from_rotvec(flat[:, 2]).apply(self._tpose_forearm_axis())
        cos = forearm @ self._tpose_upper_arm_axis()
        return np.arccos(np.clip(cos, -1.0, 1.0)).reshape(leading)

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

    def shoulder_elevation_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return shoulder elevation: upper-arm angle from straight down.

        0 = arm hanging at the side, pi/2 = horizontal, pi = straight
        overhead, regardless of the plane of elevation. This is the
        goniometric elevation the lateral/depth component proxies cannot
        capture (both read ~0 for a vertical upper arm).
        """
        upper_arm = self._upper_arm_direction_spine_frame(trajectory)
        return np.arccos(np.clip(-upper_arm[..., 1], -1.0, 1.0))

    def shoulder_internal_external_rotation_angles(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """Return signed shoulder twist around the T-pose upper-arm axis.

        Twist is decomposed from the composed collar∘shoulder∘elbow rotation
        because under this repo's FK convention that composition orients the
        upper-arm bone — the same physical twist can live in either controlled
        slot. Positive sign follows the right-hand rule about the T-pose
        shoulder-to-elbow axis, which is internal (medial) rotation twisting the
        forearm toward the body midline; negative is external rotation.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        leading = trajectory.shape[:-2]
        flat = trajectory.reshape(-1, 3, 3)
        upper_arm_rotvec = self._upper_arm_rotations(flat).as_rotvec()
        axis = self._tpose_upper_arm_axis()
        angles = _twist_angles_about_axis(upper_arm_rotvec, axis)
        return angles.reshape(leading)

    def _upper_arm_rotations(self, flat: np.ndarray) -> Rotation:
        """Return the composed collar∘shoulder∘elbow rotations for ``(N, 3, 3)``
        frames — the rotation orienting the upper-arm bone in the spine3 frame
        under this repo's FK convention."""
        collar = Rotation.from_rotvec(self.fk.collar_aa[None])
        return (
            collar
            * Rotation.from_rotvec(flat[:, 0])
            * Rotation.from_rotvec(flat[:, 1])
        )

    def _upper_arm_direction_spine_frame(self, trajectory: np.ndarray) -> np.ndarray:
        """Return unit shoulder-to-elbow directions in the spine3 frame.

        Computed by applying the composed joint rotations to the T-pose bone
        axis; identical to normalizing the FK elbow−shoulder position
        difference (the spine3 rotation cancels), without running FK.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        leading = trajectory.shape[:-2]
        flat = trajectory.reshape(-1, 3, 3)
        directions = self._upper_arm_rotations(flat).apply(
            self._tpose_upper_arm_axis()
        )
        return directions.reshape((*leading, 3))

    def _tpose_upper_arm_axis(self) -> np.ndarray:
        """Return unit T-pose shoulder-to-elbow axis in the spine3 frame."""
        tpose = self.fk.tpose_joints
        axis = tpose[_JOINT_NAMES["left_elbow"]] - tpose[_JOINT_NAMES["left_shoulder"]]
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm

    def _tpose_forearm_axis(self) -> np.ndarray:
        """Return unit T-pose elbow-to-wrist axis in the spine3 frame."""
        tpose = self.fk.tpose_joints
        axis = tpose[_JOINT_NAMES["left_wrist"]] - tpose[_JOINT_NAMES["left_elbow"]]
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


def generated_cost_feature_dependencies(code: str) -> tuple[str, ...]:
    """Return named joint features read directly from the runtime context."""
    tree = ast.parse(code)
    methods = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
    }
    return tuple(
        feature
        for method, feature in _NAMED_FEATURE_METHODS.items()
        if method in methods
    )


def replace_generated_costs(
    composite: CompositeTrajectoryCost,
    cost: GeneratedPythonCost | None,
) -> CompositeTrajectoryCost:
    """Replace every generated term while preserving hand-authored costs."""
    terms = [
        term for term in composite.terms() if not isinstance(term, GeneratedPythonCost)
    ]
    if cost is not None:
        terms.append(cost)
    return CompositeTrajectoryCost(terms)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Leniently parse a JSON object, accepting fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def parse_llm_cost_response(raw: str) -> LlmCostResponse:
    """Parse the LLM JSON response, accepting optional Markdown fences."""
    data = extract_json_object(raw)
    if data is None:
        raise GeneratedCostValidationError("LLM response is not JSON")
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
    full_correction_traj: np.ndarray | None = None,
    cartesian_goal: np.ndarray | None = None,
    cartesian_threshold: float | None = None,
    rejected_trajs: tuple[np.ndarray, ...] | None = None,
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
        full_correction_traj=(
            np.asarray(full_correction_traj, dtype=np.float64)
            if full_correction_traj is not None
            else None
        ),
        cartesian_goal=(
            np.asarray(cartesian_goal, dtype=np.float64)
            if cartesian_goal is not None
            else None
        ),
        cartesian_threshold=cartesian_threshold,
        rejected_trajs=tuple(
            np.asarray(traj, dtype=np.float64) for traj in (rejected_trajs or ())
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
    ``"cartesian_goal"``. When explicitly marked-wrong candidate trajectories are
    available, a rollout-labeled terminal joint-feature comparison is included under
    ``"candidate_comparison"``.
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
    if context.rejected_trajs:
        summaries["candidate_comparison"] = _candidate_comparison_summary(context)
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


def _shoulder_elbow_frame_summary(q: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    return {
        "shoulder": {"value": q[0].tolist(), "norm": float(np.linalg.norm(q[0]))},
        "elbow": {"value": q[1].tolist(), "norm": float(np.linalg.norm(q[1]))},
    }


def _shoulder_elbow_joint_summary(trajectory: np.ndarray) -> dict[str, Any]:
    t = np.asarray(trajectory, dtype=np.float64)
    return {
        "shoulder": {
            "start": t[0, 0].tolist(),
            "end": t[-1, 0].tolist(),
            "norm_start": float(np.linalg.norm(t[0, 0])),
            "norm_end": float(np.linalg.norm(t[-1, 0])),
        },
        "elbow": {
            "start": t[0, 1].tolist(),
            "end": t[-1, 1].tolist(),
            "norm_start": float(np.linalg.norm(t[0, 1])),
            "norm_end": float(np.linalg.norm(t[-1, 1])),
        },
    }


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
        "rollout": _joint_feature_summary(context, np.asarray(rollout, dtype=np.float64)),
        "target": _joint_feature_summary(context, target_traj),
    }



def build_joint_angle_series(
    context: GeneratedCostContext,
    trajectory: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the per-frame anatomical joint-feature series for a trajectory.

    Same four features as :func:`_joint_feature_summary` (elbow flexion, shoulder
    flexion/extension, abduction/adduction, internal/external rotation), but the
    full ``(T,)`` radian series per feature instead of summary statistics — used
    to plot joint angles over time.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    return {
        "elbow_flexion": context.elbow_flexion_angles(trajectory),
        "shoulder_flexion_extension": context.shoulder_flexion_extension_angles(
            trajectory
        ),
        "shoulder_abduction_adduction": context.shoulder_abduction_adduction_angles(
            trajectory
        ),
        "shoulder_elevation": context.shoulder_elevation_angles(trajectory),
        "shoulder_internal_external_rotation": (
            context.shoulder_internal_external_rotation_angles(trajectory)
        ),
    }


def _trajectory_summary(
    trajectory: np.ndarray,
    positions: np.ndarray,
    spine3_pos: np.ndarray,
) -> dict[str, Any]:
    return {
        "joint_angles": _array_stats(trajectory),
        "positions": _position_summary(positions, spine3_pos),
        "shoulder_elbow_values": _shoulder_elbow_joint_summary(trajectory),
    }


def _state_summary(
    q: np.ndarray,
    positions: np.ndarray,
    spine3_pos: np.ndarray,
) -> dict[str, Any]:
    return {
        "joint_angles": np.asarray(q, dtype=np.float64).tolist(),
        "positions": _position_frame_summary(positions, spine3_pos),
        "shoulder_elbow_values": _shoulder_elbow_frame_summary(q),
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
        "shoulder_elevation": _series_stats(
            context.shoulder_elevation_angles(trajectory)
        ),
        "shoulder_internal_external_rotation": _series_stats(
            context.shoulder_internal_external_rotation_angles(trajectory)
        ),
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
        for name, values in {
            "elbow_flexion": context.elbow_flexion_angles(q),
            "shoulder_flexion_extension": (
                context.shoulder_flexion_extension_angles(q)
            ),
            "shoulder_abduction_adduction": (
                context.shoulder_abduction_adduction_angles(q)
            ),
            "shoulder_elevation": context.shoulder_elevation_angles(q),
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
