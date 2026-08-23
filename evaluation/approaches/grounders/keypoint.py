"""LLM-keypoint grounding baseline: language -> one workspace target (no text-to-motion).

The utterance is interpreted by an LLM into a single 3D keypoint that one arm
part (elbow or wrist) should move toward — the "language to workspace keypoint"
grounding style. The correction ramps the chosen joint from the nominal plan
toward the keypoint and projects back to arm axis-angles by position IK. The
simulated user only tunes playback magnitude of the one proposed motion; it
never selects among alternatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evaluation.approaches.grounders.base import (
    LANDMARKS,
    ClusterSelector,
    Grounder,
)
from evaluation.rig import EvalRig
from evaluation.structs import GroundingResult, InteractionTask
from uncertain_feedback.planners.mpc.costs import extract_json_object
from uncertain_feedback.planners.mpc.kinematics import (
    ELBOW_CHAIN_IDX,
    WRIST_CHAIN_IDX,
    q_to_arm_aa,
)
from uncertain_feedback.simulated_users import SimulatedUser
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_KEYPOINT_SYSTEM_PROMPT = (
    "You interpret a care recipient's spoken correction about how a robot is "
    "repositioning their left arm. Choose ONE arm part (elbow or wrist) and a "
    "3D keypoint (meters, world frame: +X = person's left, +Y = up, +Z = "
    "person's front) that the part should move toward to satisfy the "
    "correction. Use the numeric scene context and conversation history to "
    "resolve ambiguity, and keep the keypoint within plausible reach of the "
    "current position. If the utterance implies no expressible spatial "
    "target, return a null keypoint. Return only a JSON object: "
    '{"joint": "elbow"|"wrist", "keypoint": [x, y, z] | null, "reply": str}.'
)

_JOINT_CHAINS = {"elbow": ELBOW_CHAIN_IDX, "wrist": WRIST_CHAIN_IDX}


class KeypointGrounder(Grounder):
    """Candidates are single LLM-proposed keypoint edits of the nominal plan."""

    def __init__(
        self,
        displacement_cap: float = 0.4,
        use_generator_rig: bool = False,
    ) -> None:
        super().__init__()
        self._displacement_cap = displacement_cap
        # Load the motion generator anyway (unused for grounding) so the rig
        # geometry (decoded pose spine3/body) matches the system arms exactly.
        if use_generator_rig:
            self.requires_generator = True

    def reset(
        self,
        rig: EvalRig,
        user: SimulatedUser,
        task: InteractionTask,
        episode_dir: Path,
    ) -> None:
        super().reset(rig, user, task, episode_dir)
        self._history: list[str] = []
        self._llm: Any = None

    def _interpret(self, text: str, scene_context: str) -> dict[str, Any] | None:
        """Return {"joint": str, "keypoint": (3,) array} or None for a no-op."""
        if self._llm is None:
            from uncertain_feedback.llm import (  # pylint: disable=import-outside-toplevel
                OpenAIModel,
            )

            self._llm = OpenAIModel(
                model=self.rig.cfg.llm_cost.model,
                system_prompt=_KEYPOINT_SYSTEM_PROMPT,
                temperature=0.0,
                reasoning_effort="low",
            )
        prompt = (
            f"Scene (positions in meters):\n{scene_context}\n"
            "Conversation history:\n"
            + ("\n".join(self._history) if self._history else "(none)")
            + f'\nUser says: "{text}"'
        )
        for _ in range(2):
            raw = self._llm.get_full_output(prompt)
            data = extract_json_object(raw)
            if data is None:
                continue
            self._history.append(f"user: {text}")
            self._history.append(f"robot: {data.get('reply', '')}")
            keypoint = data.get("keypoint")
            if (
                data.get("joint") in _JOINT_CHAINS
                and isinstance(keypoint, list)
                and len(keypoint) == 3
            ):
                return {
                    "joint": str(data["joint"]),
                    "keypoint": np.asarray(keypoint, dtype=np.float64),
                }
            return None
        self._history.append(f"user: {text}")
        return None

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
        goal: np.ndarray,
    ) -> GroundingResult:
        del q_feedback, goal
        rig = self.rig
        nominal_aa = q_to_arm_aa(nominal_plan, rig.fk.elbow_hinge_axis)
        arm_pos = rig.fk.fk_batch(nominal_aa, rig.spine3_pos, rig.spine3_aa)
        body = rig.body_pos if rig.body_pos is not None else rig.fk.tpose_all_joints

        def fmt(point: np.ndarray) -> str:
            return "[" + ", ".join(f"{v:.2f}" for v in point) + "]"

        scene_lines = [
            f"arm now: elbow {fmt(arm_pos[0, ELBOW_CHAIN_IDX])}, "
            f"wrist {fmt(arm_pos[0, WRIST_CHAIN_IDX])}",
            f"planned end: elbow {fmt(arm_pos[-1, ELBOW_CHAIN_IDX])}, "
            f"wrist {fmt(arm_pos[-1, WRIST_CHAIN_IDX])}",
        ]
        scene_lines += [
            f"landmark {name}: {fmt(np.asarray(body[i], dtype=np.float64))}"
            for name, i in LANDMARKS
        ]
        edit = self._interpret(text, "\n".join(scene_lines))

        candidates: dict[int, np.ndarray] = {0: nominal_aa}
        chosen = 0
        if edit is not None:
            chain = _JOINT_CHAINS[edit["joint"]]
            delta = edit["keypoint"] - arm_pos[-1, chain]
            norm = float(np.linalg.norm(delta))
            if norm > self._displacement_cap:
                delta = delta * (self._displacement_cap / norm)
            ramp = np.linspace(0.0, 1.0, nominal_aa.shape[0])[:, None]
            positions = arm_pos.copy()
            positions[:, chain] += ramp * delta
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
