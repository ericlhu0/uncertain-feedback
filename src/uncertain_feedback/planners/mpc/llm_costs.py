"""LLM-generated Python trajectory costs for MPC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from uncertain_feedback.planners.mpc.costs import (
    MpcCostContext,
    TrajectoryCost,
)
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_BONE_PAIRS_22,
    SmplLeftArmFK,
)
from uncertain_feedback.utils.plot import ArmVisualizer, _ORTHO_VIEWS, _draw_bones_2d

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

    @property
    def current_positions(self) -> np.ndarray:
        """Current arm-chain positions with shape ``(5, 3)``."""
        return self.fk.fk(self.current_q, self.spine3_pos, self.spine3_aa)

    @property
    def mdm_positions(self) -> np.ndarray:
        """Generated arm-chain positions with shape ``(T, 5, 3)``."""
        return self.fk_batch(self.mdm_traj)

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
    )


def build_motion_summaries(context: GeneratedCostContext) -> dict[str, Any]:
    """Return JSON-serializable current/recent/MDM trajectory summaries."""
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
    return summaries


_IMAGE_DESCRIPTION_PROMPT = (
    "Describe the arm trajectory shown in this image. "
    "Focus on: the overall direction of motion, the shape of the path, "
    "roughly where the arm starts and ends, and any notable features "
    "(e.g. arc, straight line, elbow going up/down, wrist reaching far). "
    "Be concise (2-4 sentences)."
)


def build_llm_cost_prompt(
    instruction: str,
    summaries: dict[str, Any],
    image_paths: list[Path],
    image_description: str = "",
) -> str:
    """Build the text prompt for the cost-generator LLM."""
    image_section = ""
    if image_paths:
        image_section = "An image of the trajectory is attached."
        if image_description:
            image_section += f"\n\nImage description: {image_description}"
    return f"""
You are a robot controller assisting a person with mobility limitations with arm movements. Your task is to generate a compact Python cost function that encodes the smallest set of user or caregiver preferences that realistically explains this correction. Keep in mind that the language instruction can be ambiguous and omit certain preferences that are implied by the motion. Think in terms of what the person likely cared about: what region the arm should reach, how high or far it should move, what joint-angle range should be allowed, or whether one clearly uncomfortable/support-related posture should be avoided. Do not encode every visible difference between the original and corrected trajectory. A feature value at the corrected trajectory is not the only preferable feature value; it is just one example inside a broader preferred set. Your goal is to infer that preferred set and add only the minimum cost terms needed to express it. Also keep in mind that joint limits are not independent from each other: for example, a user can be comfortable with extending their elbow in front of them but not to the side.

Instruction:
{instruction}

{image_section}

Guidelines:
- DO: first infer the most realistic human preference behind the correction in plain language. Then implement the fewest cost terms that capture that preference.
- DO: default to one cost term. Add a second or third term only when it captures a genuinely separate preference that is clearly implied by the instruction and trajectory.
- DO: use the MDM trajectory summaries to extract the *intent* behind the motion — e.g. how high the person wants to reach (wrist z range or lower bound), roughly where they want the arm to end (a preferred region, usually not an exact point), how much lateral reach is involved, joint angle limits, etc. Use these numbers to define preference regions, ranges, thresholds, or margins.
- DO: ground each preference term numerically in the generated trajectory data. The MDM trajectory tells you what this person's version of the instruction means quantitatively — treat those numbers as examples inside a preferred set, not as exact targets to imitate. You may draw targets from any part of the trajectory — start, end, or mid-trajectory — if that best represents the preference, but convert point values into a range, one-sided bound, or tolerance region whenever possible (e.g. "elbow stay above height X mid-motion").
- DO: include multiple cost terms only when the instruction and trajectory imply multiple distinct preferences, and keep the set small and natural. A typical cost should have one preference term; two or three should be uncommon.
- DO: keep params minimal and interpretable. Prefer trajectory-derived bounds, tolerances, region centers with radii, or thresholds. Avoid inventing separate tuning-weight params for each term unless there is a clear reason.
- DO: write costs grounded in what a caregiver would consider: reaching a goal, keeping the arm supported, or avoiding uncomfortable positions.
- DO: consider relative preferences only when implied by the instruction or trajectory, such as keeping the wrist/elbow near the torso (spine3) or near another arm joint. Express these as broad distance limits or preferred distance ranges, not exact pairwise distances, unless exact contact is clearly intended.
- DO: use named joint-angle features when they are the natural way to express the preference, such as elbow bend/flexion, shoulder flexion/extension, shoulder abduction/adduction, shoulder internal/external rotation, or avoiding an uncomfortable joint configuration. Prefer the helper methods in the Runtime API and broad ranges or one-sided bounds over exact full-arm pose matching.
- DO NOT: imitate the trajectory's shape or timing — no Gaussian time peaks tied to specific arc positions, no sweep patterns that encode the MDM trajectory geometry, no costs that enforce a specific mid-trajectory path.
- DO NOT: write costs that only make sense as "follow this exact path." A caregiver cares about whether the arm reaches a useful region and whether it's comfortable throughout, not about replicating the specific arc the MDM generated.
- DO: prefer penalties for being outside a box/ball/height band, below a minimum, above a maximum, outside a directional/lateral region, or outside a relative-distance range. Exact Cartesian point penalties are acceptable when the instruction clearly specifies a precise target, contact point, or placement.
- DO NOT: add generic smoothness, velocity, or acceleration penalties.
- DO NOT: add endpoint, wrist height, elbow support, progress, and posture terms all at once. Choose the smallest subset that explains the correction.

Framing examples:
- Caregiver-style (good): "Help the arm reach a comfortable high region and keep the elbow from dropping below the shoulder."
- Region-style (good): "Penalize the wrist only when it is below the preferred height band or outside the broad forward-reaching region."
- Relative-style (good when implied): "Keep the wrist from drifting too far away from the torso while reaching."
- Exact-point style (only when precise target is explicit): "Place the wrist exactly at a specified button or contact point."
- Exact-point style (avoid by default): "Force the wrist to end exactly at the MDM final wrist xyz coordinate."
- Trajectory-imitation (avoid): "Apply a Gaussian reward peaking at mid-horizon to encourage the upward arc, then enforce a leftward descending sweep to match the MDM path."

Runtime API:
- Define exactly: def cost(q_trajs, context, params):
- q_trajs has shape (n_rollouts, horizon + 1, 3, 3) for left_shoulder,
  left_elbow, left_wrist axis-angle states.
- Prefer these named joint-feature helpers when writing joint-space costs:
  context.elbow_flexion_angles(q_trajs)
  context.shoulder_flexion_extension_angles(q_trajs)
  context.shoulder_abduction_adduction_angles(q_trajs)
  context.shoulder_internal_external_rotation_angles(q_trajs)
  Each accepts any array ending in (3, 3), including q_trajs[:, 1:] or
  context.mdm_traj, and returns the matching leading shape. Values are radians.
  The shoulder helpers are preference-friendly component-angle approximations
  in the spine3 frame: flexion/extension is signed upper-arm movement along
  the z axis, abduction/adduction is signed lateral upper-arm movement along
  the x axis (positive is away from the torso for the left arm), and
  internal/external rotation is signed twist around the T-pose upper-arm axis.
- Raw joint indexing is also allowed for advanced cases: q_trajs[:, :, 0, :]
  is left_shoulder, q_trajs[:, :, 1, :] is left_elbow, and q_trajs[:, :, 2, :]
  is left_wrist. The final dimension is a 3D local axis-angle rotation vector,
  not separate named anatomical DOFs. Avoid selecting one arbitrary rotvec
  component and calling it flexion/abduction/rotation. The left_wrist row exists
  in the SMPL/HML state, but this simplified arm model has no hand joint beyond
  the wrist, so use wrist joint-angle terms only when a wrist-rotation
  preference is clearly implied.
- context.fk_rollouts(q_trajs) returns positions with shape
  (n_rollouts, horizon + 1, 5, 3) for spine3, left_collar, left_shoulder,
  left_elbow, left_wrist.
- context.mdm_positions has shape (T, 5, 3) — a numpy array. Use numpy
  indexing only: context.mdm_positions[:, 4, 2] gives wrist z over the MDM
  trajectory; context.mdm_positions[-1, 4, :] gives the final wrist position.
  Do NOT use dict-style access like context.mdm_positions['left_wrist'] — it
  will raise a TypeError at runtime.
- context.current_positions has shape (5, 3) — a numpy array. Use numpy
  indexing: context.current_positions[4] gives the current wrist position.
  Do NOT use dict-style access.
- context.joint_index(name) accepts spine3, collar, shoulder, elbow, wrist and
  left_* aliases and returns an integer index (0–4).
- If you need statistics such as maximum wrist z during the MDM trajectory,
  read those values from the Summaries section below and encode them as
  hardcoded floats or params in your cost function — do not try to compute
  them from context.mdm_positions using dict-style access.
- context.mdm_traj and context.recent_q/recent_positions are also available
  as numpy arrays.
- np is available. Do not import anything.
- Return a finite numpy array with shape (n_rollouts,).

Hard requirements:
- Return only JSON with keys: description, code, params, explanation, recipient_explanation.
- Prefer costs over future timesteps q_trajs[:, 1:], not only the initial state.
- Do not include smoothness, velocity, or acceleration costs.
- Avoid exact Cartesian point matching for wrists, elbows, or other body parts by default. If a Cartesian location is useful, prefer a region with a tolerance/radius or one-sided bounds, and penalize only violations outside that region. Use an exact point only when the instruction clearly calls for a precise target or contact point.
- Use the least number of cost terms possible. If one term explains the realistic human preference, use one term.
- Params should usually contain no more than three values; if more are needed, simplify the cost.
- The description field must state the caregiver preference the cost encodes in plain human language — how a caregiver would explain what they are trying to achieve. Do not describe the code structure or mathematical approach.
- The explanation field must describe in 3–6 sentences: (1) what preferences the cost captures and why they are implied by the instruction and trajectory, (2) which cost terms were chosen and what each one penalizes, and (3) which specific numbers were drawn from the trajectory data and where they came from (e.g. "the wrist max-z of 0.42 m came from the 95th percentile of the MDM wrist-z range"). This field is for developer understanding — be specific and concrete.
- The recipient_explanation field must be 1–3 brief, plain-language sentences you would say directly to the care recipient. Avoid technical terms, measurements, code, "cost", "penalty", and robot-control jargon. Explain how the new cost functions will guide their arm movement in layman terms; they should be understand what motions their arms will and won't be able to make with the cost functions.

Summaries:
{json.dumps(summaries, indent=2, sort_keys=True)}
""".strip()


def render_prompt_images(
    context: GeneratedCostContext,
    output_dir: Path,
) -> list[Path]:
    """Render trajectory-grounding overlay image for the LLM prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "overlay.png"
    _render_overlay(context, path)
    return [path]


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



def _render_overlay(context: GeneratedCostContext, path: Path) -> None:
    positions = context.mdm_positions  # (T, 5, 3) arm chain — wrist path / markers

    # Reference body: actual body pose if available, else translated T-pose.
    if context.body_pos is not None:
        ref_body = context.body_pos
    else:
        tpose = context.fk.tpose_all_joints
        ref_body = tpose + (context.spine3_pos - context.fk.tpose_spine3_pos)

    cur_full = context.fk.full_body_positions(
        context.current_q, context.spine3_pos, context.spine3_aa
    )

    # Equal-square axis limits across all three axes (matches ArmVisualizer.format_3d_axis).
    all_pts = np.concatenate([ref_body, positions.reshape(-1, 3), context.current_positions], axis=0)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    lims = [(center[i] - radius, center[i] + radius) for i in range(3)]

    n_samples = min(12, positions.shape[0])
    sample_indices = np.linspace(0, positions.shape[0] - 1, n_samples).round().astype(int)
    cmap = plt.get_cmap("Blues")
    denom = max(1, positions.shape[0] - 1)
    wrist_path = positions[:, _JOINT_NAMES["left_wrist"]]
    start_w = positions[0, _JOINT_NAMES["left_wrist"]]
    end_w = positions[-1, _JOINT_NAMES["left_wrist"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, view in zip(axes, _ORTHO_VIEWS):
        ax.set_aspect("equal")
        ax.set_title(view.title, fontsize=9)
        ax.set_xlabel(view.hl, fontsize=8)
        ax.set_ylabel(view.vl, fontsize=8)
        ax.set_xlim(*lims[view.hi])
        ax.set_ylim(*lims[view.vi])
        ax.tick_params(labelsize=7)

        # Static reference body (grey)
        _draw_bones_2d(ax, ref_body, ArmVisualizer.BODY_BONES, view.hi, view.vi,
                       ArmVisualizer.BODY_COLOR, alpha=0.45, lw=1.2)

        # MDM trajectory arm bones (blue gradient, sampled frames)
        for frame_idx in sample_indices:
            t = 0.3 + 0.7 * (frame_idx / denom)
            full = context.fk.full_body_positions(
                context.mdm_traj[frame_idx], context.spine3_pos, context.spine3_aa
            )
            _draw_bones_2d(ax, full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                           cmap(t), alpha=0.5, lw=1.2)

        # Wrist path and start/end markers
        ax.plot(wrist_path[:, view.hi], wrist_path[:, view.vi],
                color="steelblue", alpha=0.5, linewidth=1.0)
        ax.scatter(start_w[view.hi], start_w[view.vi], marker="o", color="lime", s=55, zorder=5)
        ax.scatter(end_w[view.hi], end_w[view.vi], marker="X", color="red", s=65, zorder=5)

        # Current pose arm (orange)
        _draw_bones_2d(ax, cur_full, LEFT_ARM_BONE_PAIRS_22, view.hi, view.vi,
                       "tab:orange", alpha=1.0, lw=2.2)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=positions.shape[0] - 1))
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=axes[-1], shrink=0.8, pad=0.04, label="frame (light=early, dark=late)")
    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color="tab:orange", linewidth=2, label="current"),
            plt.Line2D([0], [0], marker="o", color="lime", linestyle="", markersize=7, label="traj start"),
            plt.Line2D([0], [0], marker="X", color="red", linestyle="", markersize=7, label="traj end"),
        ],
        fontsize=7, loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
