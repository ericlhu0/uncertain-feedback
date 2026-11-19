"""Compiled-constraint runtime for generated Python limits.

The planner-side half of the generated-cost pipeline, independent of how a limit
was authored: the runtime context handed to generated code
(:class:`GeneratedCostContext`), the executable wrapper that turns generated
margins into a planner cost (:class:`GeneratedPythonCost`), and the JSON response
parser/compiler. Generated code states limits as margins; the penalty is the
system's.

Prompt summaries and overlay images built on this context live in
:mod:`uncertain_feedback.cost_generation.summaries`; scoring lives in
:mod:`uncertain_feedback.evaluation_mechanism.scoring`.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from types import FunctionType
from typing import TYPE_CHECKING, Any

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    SPEED_FEATURE_NAMES,
    arm_aa_from_state,
    arm_feature_series,
    arm_motion_series,
    canonical_arm_q,
)
from uncertain_feedback.planners.mpc.costs.base import (
    CompositeTrajectoryCost,
    TrajectoryCost,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_NAMES,
    SmplLeftArmFK,
)

if TYPE_CHECKING:
    from torch import Tensor

    from uncertain_feedback.planners.mpc.goal_spaces.regions import GoalRegion

# The FK chain names plus their "left_"-stripped short aliases.
_JOINT_NAMES = {name: i for i, name in enumerate(LEFT_ARM_CHAIN_NAMES)} | {
    name.removeprefix("left_"): i
    for i, name in enumerate(LEFT_ARM_CHAIN_NAMES)
    if name.startswith("left_")
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

    Arm-state fields are canonical q arrays ending in ``(7,)``. Helper methods
    also accept decoded arrays ending in ``(3, 3)`` at generated-code/FK boundaries.
    ``rejected_trajs`` contains only candidates the person explicitly marked wrong.
    ``cartesian_goal`` is the goal region's representative wrist point (``None``
    for feature regions); ``cartesian_region`` is the region reaching is judged by.
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
    cartesian_region: GoalRegion | None = None
    # Planner penalty per unit of worst-step excess (``llm_cost.limit_weight``).
    limit_weight: float = 1000.0
    # Planner penalty per rad of path excess averaged over a rollout
    # (``llm_cost.pullback_weight``).
    pullback_weight: float = 10.0
    # ``llm_cost.enforcement``: "hard" or "soft".
    enforcement: str = "hard"

    @property
    def current_positions(self) -> np.ndarray:
        """Current arm-chain positions with shape ``(5, 3)``."""
        return self.fk_batch(self.current_q)

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
        """Return arm-chain positions for q or axis-angle arm states."""
        arm_aa = arm_aa_from_state(trajectory, self)
        leading = arm_aa.shape[:-2]
        flat = arm_aa.reshape((-1, 3, 3))
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
        """Return elbow bend for q or axis-angle arm states."""
        return self.feature_series(trajectory)["elbow_flexion"]

    def shoulder_abduction_adduction_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return signed shoulder abduction/adduction proxy in the spine3 frame.

        This is the lateral component angle of the shoulder-to-elbow direction:
        positive values move toward ``+x`` (left-arm abduction / away from the
        torso), and negative values move toward ``-x`` (adduction / across the
        torso).
        """
        return self.feature_series(trajectory)["shoulder_abduction_adduction"]

    def shoulder_flexion_extension_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return signed shoulder flexion/extension proxy in the spine3 frame.

        This is the depth component angle of the shoulder-to-elbow direction:
        positive values move toward ``+z`` and negative values move toward
        ``-z``.
        """
        return self.feature_series(trajectory)["shoulder_flexion_extension"]

    def shoulder_elevation_angles(self, trajectory: np.ndarray) -> np.ndarray:
        """Return shoulder elevation: upper-arm angle from straight down.

        0 = arm hanging at the side, pi/2 = horizontal, pi = straight
        overhead, regardless of the plane of elevation. This is the
        goniometric elevation the lateral/depth component proxies cannot
        capture (both read ~0 for a vertical upper arm).
        """
        return self.feature_series(trajectory)["shoulder_elevation"]

    def shoulder_internal_external_rotation_angles(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """Return shoulder-only twist from the canonical ``q[3:6]`` block."""
        return self.feature_series(trajectory)["shoulder_internal_external_rotation"]

    def feature_series(self, trajectory: np.ndarray) -> dict[str, np.ndarray]:
        """Return every canonical anatomical feature in one conversion pass."""
        return arm_feature_series(trajectory, self)

    def canonical_q(self, trajectory: np.ndarray) -> np.ndarray:
        """Return q-space states for a q or axis-angle arm trajectory."""
        return canonical_arm_q(trajectory, self)

    def arm_aa(self, trajectory: np.ndarray) -> np.ndarray:
        """Return FK-boundary axis-angle states for a q or legacy trajectory."""
        return arm_aa_from_state(trajectory, self)


@dataclass(frozen=True)
class GeneratedPythonCost(TrajectoryCost):
    """Executable LLM-generated limits, charged by the planner as a hard cost.

    The generated code defines constraints, not penalties: ``path_margins(features,
    params)`` and/or ``speed_margins(features, params)``, each returning ``{name:
    margin}`` with ``(N, T)`` torch margins in physical units (rad for a path
    limit; rad/s or m/s for a speed limit), positive when violated. Path margins
    see the pose features only; speed margins also see the speed features
    (``SPEED_FEATURE_NAMES``), measured over the step into each frame. The
    planner (features from IK) and MDM steering (features from the sample's
    positions, differentiable) evaluate the same margins.

    The planner's penalty is the system's: ``context.limit_weight`` times the
    worst step's excess, linear, summed over margins. A path margin is charged
    only beyond the excess the rollout's first frame already has, so an arm that
    starts past a limit can still plan, as long as it goes no further past it.
    ``context.pullback_weight`` times the path excess averaged over the rollout
    is added on top: a small cost that brings such an arm back inside. With
    ``context.enforcement == "soft"`` the planner instead charges each margin's
    squared excess averaged over the rollout, at unit weight.
    """

    code: str
    params: dict[str, Any]
    context: GeneratedCostContext
    description: str = ""
    _path: FunctionType | None = field(init=False, repr=False, compare=False)
    _speed: FunctionType | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        path, speed = compile_generated_cost(self.code)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_speed", speed)

    def margins(
        self, features: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The ``(path, speed)`` margins of ``(N, T)`` feature series."""
        pose = {name: features[name] for name in FEATURE_NAMES if name in features}
        return (
            {} if self._path is None else self._path(pose, self.params),
            {} if self._speed is None else self._speed(features, self.params),
        )

    def _excess(
        self, func: FunctionType | None, features: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Each margin of ``func`` over ``(N, T)`` features, clamped at zero."""
        import torch  # pylint: disable=import-outside-toplevel

        if func is None:
            return []
        shape = next(iter(features.values())).shape
        with torch.no_grad():
            margins = func(
                {name: torch.from_numpy(series) for name, series in features.items()},
                self.params,
            )
        excess = []
        for name, margin in margins.items():
            values = torch.as_tensor(margin, dtype=torch.float64).numpy()
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise GeneratedCostValidationError(
                    f"margin {name!r} must be finite with shape {shape}, "
                    f"got shape {values.shape}"
                )
            excess.append(np.maximum(values, 0.0))
        return excess

    def pose_excess(self, q: np.ndarray) -> np.ndarray:
        """Summed path-limit excess (rad) at each pose of ``q`` ``(M, 7)``, ``(M,)``."""
        features = self.context.feature_series(np.asarray(q, dtype=np.float64)[None])
        excess = self._excess(self._path, features)
        return np.sum(excess, axis=0)[0] if excess else np.zeros(len(q))

    def path_excess(self, q_trajs: np.ndarray) -> np.ndarray:
        """Worst-step path excess of ``(N, T, 7)`` rollouts beyond their start, ``(N,)``.

        Radians, summed over the path margins: how much further past each limit
        the rollout goes than its first frame already is.
        """
        features = self.context.feature_series(q_trajs)
        return np.sum(
            [
                np.maximum(excess[:, 1:] - excess[:, :1], 0.0).max(axis=1)
                for excess in self._excess(self._path, features)
            ],
            axis=0,
        ) + np.zeros(len(q_trajs))

    def path_lingering(self, q_trajs: np.ndarray) -> np.ndarray:
        """Path excess of ``(N, T, 7)`` rollouts averaged over their steps, ``(N,)``.

        Radians, summed over the path margins and counted from the limit itself.
        """
        features = self.context.feature_series(q_trajs)
        return np.sum(
            [
                excess[:, 1:].mean(axis=1)
                for excess in self._excess(self._path, features)
            ],
            axis=0,
        ) + np.zeros(len(q_trajs))

    def speed_excess(self, q_trajs: np.ndarray) -> np.ndarray:
        """Worst-step speed excess of ``(N, T, 7)`` rollouts, ``(N,)``.

        rad/s or m/s, summed over the speed margins; the frames must sit
        ``STEP_SECONDS`` apart.
        """
        features = arm_motion_series(q_trajs, self.context)
        return np.sum(
            [excess.max(axis=1) for excess in self._excess(self._speed, features)],
            axis=0,
        ) + np.zeros(len(q_trajs))

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        q_trajs = self.context.canonical_q(np.asarray(q_trajs, dtype=np.float64))
        if self.context.enforcement == "soft":
            path = self._excess(self._path, self.context.feature_series(q_trajs[:, 1:]))
            speed = self._excess(self._speed, arm_motion_series(q_trajs, self.context))
            return np.sum(
                [(excess**2).mean(axis=1) for excess in (*path, *speed)], axis=0
            ) + np.zeros(len(q_trajs))
        return self.context.limit_weight * (
            self.path_excess(q_trajs) + self.speed_excess(q_trajs)
        ) + self.context.pullback_weight * self.path_lingering(q_trajs)


def generated_cost_feature_dependencies(code: str) -> tuple[str, ...]:
    """Return the joint features generated code names (it reads them by key)."""
    strings = {
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    return tuple(
        name for name in (*FEATURE_NAMES, *SPEED_FEATURE_NAMES) if name in strings
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


def interp(x: Tensor, xp: list[float], fp: list[float]) -> Tensor:
    """``np.interp`` for tensors: linear between anchors, the end values beyond."""
    import torch  # pylint: disable=import-outside-toplevel

    xs = torch.as_tensor(xp, dtype=x.dtype, device=x.device)
    ys = torch.as_tensor(fp, dtype=x.dtype, device=x.device)
    x = x.clamp(xs[0], xs[-1])
    upper = torch.searchsorted(xs, x.detach().contiguous()).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[upper - 1], xs[upper], ys[upper - 1], ys[upper]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def compile_generated_cost(
    code: str,
) -> tuple[FunctionType | None, FunctionType | None]:
    """Compile generated source into its ``(path_margins, speed_margins)`` functions."""
    import torch  # pylint: disable=import-outside-toplevel

    namespace: dict[str, Any] = {"torch": torch, "interp": interp}
    # pylint: disable-next=exec-used
    exec(compile(code, "<llm_generated_limits>", "exec"), namespace)
    path, speed = namespace.get("path_margins"), namespace.get("speed_margins")
    if not isinstance(path, FunctionType) and not isinstance(speed, FunctionType):
        raise GeneratedCostValidationError(
            "generated code must define path_margins and/or speed_margins"
        )
    return (
        path if isinstance(path, FunctionType) else None,
        speed if isinstance(speed, FunctionType) else None,
    )


def build_generated_cost_context(
    human: Human,
    mdm_traj: np.ndarray,
    window: int,
    reference_traj: np.ndarray | None = None,
    full_correction_traj: np.ndarray | None = None,
    cartesian_goal: np.ndarray | None = None,
    cartesian_threshold: float | None = None,
    rejected_trajs: tuple[np.ndarray, ...] | None = None,
    cartesian_region: GoalRegion | None = None,
    limit_weight: float = 1000.0,
    pullback_weight: float = 10.0,
    enforcement: str = "hard",
) -> GeneratedCostContext:
    """Build the q-native runtime context generated Python costs see.

    The person's kinematics, current arm state and last ``window`` executed
    frames come from ``human``.
    """
    return GeneratedCostContext(
        fk=human.fk,
        spine3_pos=human.spine3_pos,
        spine3_aa=human.spine3_aa,
        current_q=human.q,
        mdm_traj=canonical_arm_q(mdm_traj, human),
        recent_q=human.history[-window:],
        body_pos=human.posture,
        reference_traj=(
            canonical_arm_q(reference_traj, human)
            if reference_traj is not None
            else None
        ),
        full_correction_traj=(
            canonical_arm_q(full_correction_traj, human)
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
            canonical_arm_q(traj, human) for traj in (rejected_trajs or ())
        ),
        cartesian_region=cartesian_region,
        limit_weight=limit_weight,
        pullback_weight=pullback_weight,
        enforcement=enforcement,
    )
