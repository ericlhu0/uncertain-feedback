"""Compiled-cost runtime for generated Python trajectory costs.

The planner-side half of the generated-cost pipeline, independent of how a cost was
authored: the runtime context handed to generated code
(:class:`GeneratedCostContext`), the executable cost wrapper
(:class:`GeneratedPythonCost`), and the JSON response parser/compiler.

Prompt summaries and overlay images built on this context live in
:mod:`uncertain_feedback.cost_generation.summaries`; scoring lives in
:mod:`uncertain_feedback.evaluation_mechanism.scoring`.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

import numpy as np

from uncertain_feedback.planners.mpc.arm_features import (
    arm_aa_from_state,
    arm_feature_series,
    canonical_arm_q,
)
from uncertain_feedback.planners.mpc.costs.base import (
    CompositeTrajectoryCost,
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

    Arm-state fields are canonical q arrays ending in ``(7,)``. Helper methods
    also accept decoded arrays ending in ``(3, 3)`` at generated-code/FK boundaries.
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
    """Executable LLM-generated trajectory cost."""

    code: str
    params: dict[str, Any]
    context: GeneratedCostContext
    description: str = ""
    _func: FunctionType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        func = compile_generated_cost(self.code)
        object.__setattr__(self, "_func", func)

    def __call__(self, q_trajs: np.ndarray) -> np.ndarray:
        q_trajs = np.asarray(q_trajs, dtype=np.float64)
        raw = self._func(self.context.arm_aa(q_trajs), self.context, self.params)
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
    exec(  # pylint: disable=exec-used
        compile(code, "<llm_generated_cost>", "exec"), namespace, locals_dict
    )
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
    """Build a q-native runtime context passed to generated Python costs."""
    recent_q = (
        np.stack([canonical_arm_q(q, mpc_context) for q in q_history[-window:]], axis=0)
        if q_history
        else np.empty((0, 7), dtype=np.float64)
    )
    return GeneratedCostContext(
        fk=mpc_context.fk,
        spine3_pos=np.asarray(mpc_context.spine3_pos, dtype=np.float64),
        spine3_aa=np.asarray(mpc_context.spine3_aa, dtype=np.float64),
        current_q=canonical_arm_q(current_q, mpc_context),
        mdm_traj=canonical_arm_q(mdm_traj, mpc_context),
        recent_q=recent_q,
        body_pos=(
            np.asarray(body_pos, dtype=np.float64) if body_pos is not None else None
        ),
        reference_traj=(
            canonical_arm_q(reference_traj, mpc_context)
            if reference_traj is not None
            else None
        ),
        full_correction_traj=(
            canonical_arm_q(full_correction_traj, mpc_context)
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
            canonical_arm_q(traj, mpc_context) for traj in (rejected_trajs or ())
        ),
    )
