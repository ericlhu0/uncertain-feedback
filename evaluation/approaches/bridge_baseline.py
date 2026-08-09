"""BRIDGE-style grounding baseline: landmark potential-field edits (no text-to-motion).

Adapted from BRIDGE (Wang et al., HRI '26): position modifications are
attractive/repulsive potential fields anchored at body landmarks, applied as a
one-step displacement of the arm's elbow/wrist waypoints and projected back to
arm axis-angles. Differences from BRIDGE, forced by this setting: the LLM
utterance interpreter is replaced by the harness's oracle selection over the
candidate family (an upper bound on any interpreter within the same family);
displacements are ramped in time so correction playback stays continuous from
the trigger state; and BRIDGE's velocity/force scopes have no analogue on
posture trajectories, so only the position scope is represented.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.approaches.base import Approach, ClusterSelector
from evaluation.structs import GroundingResult, InteractionTask
from uncertain_feedback.planners.mpc.costs import extract_json_object
from uncertain_feedback.planners.mpc.kinematics import q_to_arm_aa
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_ELBOW_CHAIN = 3
_WRIST_CHAIN = 4

# SMPL-22 indices for landmarks a seated-care correction plausibly references.
_LANDMARKS = (
    ("pelvis", 0),
    ("left_hip", 1),
    ("chest", 9),
    ("neck", 12),
    ("head", 15),
    ("right_shoulder", 17),
)


class BridgePotentialFieldApproach(Approach):
    """Candidates are potential-field edits of the nominal continuation.

    One candidate per (landmark, polarity): the elbow and wrist waypoints are
    displaced along the field gradient evaluated at their own positions —
    attraction pulls the arm toward the landmark, repulsion pushes it away
    within ``rho0`` — then projected back to axis-angles by position IK.
    """

    requires_generator = False

    def __init__(
        self,
        name: str = "bridge_baseline",
        learning: str = "immediate",
        attract_gain: float = 0.5,
        repel_gain: float = 0.05,
        rho0: float = 0.8,
        displacement_cap: float = 0.25,
    ) -> None:
        super().__init__(name=name, learning=learning)
        self._attract_gain = attract_gain
        self._repel_gain = repel_gain
        self._rho0 = rho0
        self._displacement_cap = displacement_cap

    def _attract(self, points: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        return self._cap(self._attract_gain * (landmark - points))

    def _repel(self, points: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        offset = points - landmark
        dist = np.linalg.norm(offset, axis=-1, keepdims=True)
        dist = np.maximum(dist, 1e-6)
        magnitude = self._repel_gain * (1.0 / dist - 1.0 / self._rho0) / dist**2
        magnitude = np.where(dist <= self._rho0, magnitude, 0.0)
        return self._cap(magnitude * offset / dist)

    def _cap(self, displacement: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(displacement, axis=-1, keepdims=True)
        factor = np.minimum(1.0, self._displacement_cap / np.maximum(norm, 1e-9))
        return displacement * factor

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del text, q_feedback
        rig = self.rig
        nominal_aa = q_to_arm_aa(nominal_plan, rig.fk.elbow_hinge_axis)
        arm_pos = rig.fk.fk_batch(nominal_aa, rig.spine3_pos, rig.spine3_aa)
        body = rig.body_pos if rig.body_pos is not None else rig.fk.tpose_all_joints
        ramp = np.linspace(0.0, 1.0, nominal_aa.shape[0])[:, None]
        candidates: dict[int, np.ndarray] = {0: nominal_aa}
        label = 1
        for _, joint_index in _LANDMARKS:
            landmark = np.asarray(body[joint_index], dtype=np.float64)
            for field in (self._attract, self._repel):
                positions = arm_pos.copy()
                for chain_index in (_ELBOW_CHAIN, _WRIST_CHAIN):
                    positions[:, chain_index] += ramp * field(
                        arm_pos[:, chain_index], landmark
                    )
                candidates[label] = rig.fk.arm_aa_from_positions_batch(
                    positions, rig.spine3_aa
                )
                label += 1
        chosen_label, magnitude = cluster_selector(candidates)
        correction = scale_trajectory(candidates[chosen_label], magnitude)
        return GroundingResult(
            candidates=candidates,
            chosen_label=chosen_label,
            magnitude=magnitude,
            correction_traj=correction,
        )


_INTERPRETER_SYSTEM_PROMPT = (
    "You interpret a care recipient's spoken correction about how a robot is "
    "repositioning their left arm. Map the utterance to trajectory "
    "modifications expressed as attraction toward or repulsion from body "
    "landmarks (the only modification type available). Landmarks: pelvis, "
    "left_hip, chest, neck, head, right_shoulder. Strengths: slight, default, "
    "strong. Use the trajectory context and conversation history to resolve "
    "ambiguity. If the utterance implies no landmark-relative position change "
    "you can express, return an empty modifications list. Return only a JSON "
    'object: {"modifications": [{"landmark": str, "mode": "attract"|"repel", '
    '"strength": "slight"|"default"|"strong"}], "reply": str}.'
)

_STRENGTH_FACTORS = {"slight": 0.5, "default": 1.0, "strong": 1.5}


class BridgeInterpreterApproach(BridgePotentialFieldApproach):
    """Language-faithful BRIDGE baseline: an LLM interprets the utterance.

    Unlike :class:`BridgePotentialFieldApproach` (oracle selection over the
    full edit family), the utterance itself must be mapped to (landmark,
    polarity, strength) modifications, as in BRIDGE's LLM interpreter with
    trajectory + conversation-history context. The simulated user only tunes
    playback magnitude of the single interpreted edit (standing in for
    BRIDGE's conversational refinement); it never sees the alternatives.
    """

    def _reset_grounding(self, task: InteractionTask) -> None:
        self._history: list[str] = []
        self._llm: Any = None

    def _interpret(self, text: str, waypoint_context: str) -> list[dict[str, str]]:
        if self._llm is None:
            from uncertain_feedback.llm import (  # pylint: disable=import-outside-toplevel
                OpenAIModel,
            )

            self._llm = OpenAIModel(
                model=self.rig.cfg.llm_cost.model,
                system_prompt=_INTERPRETER_SYSTEM_PROMPT,
                temperature=0.0,
                reasoning_effort="low",
            )
        prompt = (
            f"Planned trajectory (wrist waypoint -> nearest landmark):\n"
            f"{waypoint_context}\n"
            f"Conversation history:\n"
            + ("\n".join(self._history) if self._history else "(none)")
            + f'\nUser says: "{text}"'
        )
        for _ in range(2):
            raw = self._llm.get_full_output(prompt)
            data = extract_json_object(raw)
            if data is not None and isinstance(data.get("modifications"), list):
                self._history.append(f"user: {text}")
                self._history.append(f"robot: {data.get('reply', '')}")
                return [
                    mod
                    for mod in data["modifications"]
                    if mod.get("landmark") in dict(_LANDMARKS)
                    and mod.get("mode") in ("attract", "repel")
                ]
        self._history.append(f"user: {text}")
        return []

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        del q_feedback
        rig = self.rig
        nominal_aa = q_to_arm_aa(nominal_plan, rig.fk.elbow_hinge_axis)
        arm_pos = rig.fk.fk_batch(nominal_aa, rig.spine3_pos, rig.spine3_aa)
        body = rig.body_pos if rig.body_pos is not None else rig.fk.tpose_all_joints
        landmarks = {name: np.asarray(body[i], dtype=np.float64) for name, i in _LANDMARKS}

        wrist = arm_pos[:, _WRIST_CHAIN]
        stride = max(1, len(wrist) // 8)
        context_lines = []
        for frame in range(0, len(wrist), stride):
            nearest = min(
                landmarks, key=lambda n: np.linalg.norm(wrist[frame] - landmarks[n])
            )
            context_lines.append(f"waypoint {frame}: {nearest}")
        mods = self._interpret(text, "\n".join(context_lines))

        candidates: dict[int, np.ndarray] = {0: nominal_aa}
        chosen = 0
        if mods:
            ramp = np.linspace(0.0, 1.0, nominal_aa.shape[0])[:, None]
            positions = arm_pos.copy()
            for mod in mods:
                landmark = landmarks[mod["landmark"]]
                field = self._attract if mod["mode"] == "attract" else self._repel
                factor = _STRENGTH_FACTORS.get(mod.get("strength", "default"), 1.0)
                for chain_index in (_ELBOW_CHAIN, _WRIST_CHAIN):
                    positions[:, chain_index] += ramp * self._cap(
                        factor * field(arm_pos[:, chain_index], landmark)
                    )
            candidates[1] = rig.fk.arm_aa_from_positions_batch(positions, rig.spine3_aa)
            chosen = 1
        _, magnitude = cluster_selector({chosen: candidates[chosen]})
        correction = scale_trajectory(candidates[chosen], magnitude)
        return GroundingResult(
            candidates=candidates,
            chosen_label=chosen,
            magnitude=magnitude,
            correction_traj=correction,
        )
