"""LLM-generated Python trajectory costs for MPC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from uncertain_feedback.planners.mpc.costs import (
    MpcCostContext,
    TrajectoryCost,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK

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


@dataclass(frozen=True)
class GeneratedCostContext:
    """Read-only runtime context exposed to generated cost code."""

    fk: SmplLeftArmFK
    spine3_pos: np.ndarray
    spine3_aa: np.ndarray
    current_q: np.ndarray
    mdm_traj: np.ndarray
    recent_q: np.ndarray

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
    if not isinstance(description, str):
        raise GeneratedCostValidationError("description must be a string")
    if not isinstance(code, str):
        raise GeneratedCostValidationError("code must be a string")
    if not isinstance(params, dict):
        raise GeneratedCostValidationError("params must be an object")
    return LlmCostResponse(description, code, params)


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
    if context.recent_q.size > 0:
        summaries["recent"] = _trajectory_summary(
            context.recent_q,
            recent_positions,
            spine3_pos,
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
You generate Python source for an MPC trajectory cost for a SMPL left arm.

Instruction:
{instruction}

{image_section}

Runtime API:
- Define exactly: def cost(q_trajs, context, params):
- q_trajs has shape (n_rollouts, horizon + 1, 3, 3) for left_shoulder,
  left_elbow, left_wrist axis-angle states.
- context.fk_rollouts(q_trajs) returns positions with shape
  (n_rollouts, horizon + 1, 5, 3) for spine3, left_collar, left_shoulder,
  left_elbow, left_wrist.
- context.mdm_traj and context.mdm_positions contain the generated motion.
- context.current_q/current_positions and context.recent_q/recent_positions are
  available.
- context.joint_index(name) accepts spine3, collar, shoulder, elbow, wrist and
  left_* aliases.
- np is available. Do not import anything.
- Return a finite numpy array with shape (n_rollouts,).

Hard requirements:
- Return only JSON with keys: description, code, params.
- Prefer costs over future timesteps q_trajs[:, 1:], not only the initial state.

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
    positions = context.mdm_positions
    current = context.current_positions
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    n_samples = min(12, positions.shape[0])
    sample_indices = np.linspace(0, positions.shape[0] - 1, n_samples).round().astype(int)
    cmap = plt.get_cmap("Blues")
    denom = max(1, positions.shape[0] - 1)
    for frame_idx in sample_indices:
        # map to [0.3, 1.0] so early frames are light blue, not white
        t = 0.3 + 0.7 * (frame_idx / denom)
        _plot_arm(ax, positions[frame_idx], color=cmap(t), alpha=0.5, linewidth=1.2)

    wrist_path = positions[:, _JOINT_NAMES["left_wrist"]]
    ax.plot(wrist_path[:, 0], wrist_path[:, 1], wrist_path[:, 2],
            color="steelblue", alpha=0.5, linewidth=1.0)

    start = positions[0, _JOINT_NAMES["left_wrist"]]
    end = positions[-1, _JOINT_NAMES["left_wrist"]]
    ax.scatter(start[0], start[1], start[2], marker="o", color="lime", s=55, zorder=5)
    ax.scatter(end[0], end[1], end[2], marker="X", color="red", s=65, zorder=5)
    _plot_arm(ax, current, color="tab:orange", alpha=1.0, linewidth=2.2)

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=positions.shape[0] - 1)
    )
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=ax, shrink=0.65, pad=0.08, label="frame (light=early, dark=late)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="tab:orange", linewidth=2, label="current"),
            plt.Line2D([0], [0], marker="o", color="lime", linestyle="", markersize=7, label="traj start"),
            plt.Line2D([0], [0], marker="X", color="red", linestyle="", markersize=7, label="traj end"),
        ],
        fontsize=7, loc="upper left",
    )
    _format_3d_axis(ax, np.concatenate([positions.reshape(-1, 3), current], axis=0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_arm(
    ax: Any, positions: np.ndarray, color: str, alpha: float, linewidth: float
) -> None:
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        marker="o",
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        markersize=3,
    )


def _format_3d_axis(ax: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
