"""Direct-trajectory pure-agent grounder: the LLM writes the correction itself.

This grounder produces the same feedback grounding as the system but with no
motion prior: one LLM call per round returns the most likely interpretations of
the utterance, each carrying a trajectory payload converted into one candidate
motion. One class covers the dense, sparse-waypoint, and single-waypoint
variants in either output space (elbow/wrist positions or the five anatomical
angles). The LLM output is only *converted* into the planner's representation —
no displacement caps, no smoothing, no repair, no MPC tracking — so an
implausible motion shows up honestly as a candidate the person rejects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.approaches.grounders.base import (
    LANDMARKS,
    ClusterSelector,
    Grounder,
)
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_feature_series,
    arm_q_from_features,
)
from uncertain_feedback.planners.mpc.costs import extract_json_object
from uncertain_feedback.planners.mpc.kinematics import (
    ELBOW_CHAIN_IDX,
    Q_CLAVICLE,
    WRIST_CHAIN_IDX,
    q_to_arm_aa,
)
from uncertain_feedback.planners.rig import PlanningRig
from uncertain_feedback.simulated_users import SimulatedUser
from uncertain_feedback.uncertainty.cluster_picker import scale_trajectory

_AXES = (
    "World frame: +X is the person's left, +Y is up, +Z is the person's front. "
    "Positions are in metres, angles in radians."
)

# Same wording the method's own grounding stage uses, so both read the anatomy
# the same way.
_FEATURE_GLOSSARY = "\n".join(
    [
        "- elbow_flexion: how bent the elbow is (0 = arm straight, pi/2 = bent to a"
        " right angle, pi = forearm folded back against the upper arm).",
        "- shoulder_flexion_extension: signed depth of the upper arm (0 = neither"
        " forward nor back, +pi/2 = straight forward, -pi/2 = straight back).",
        "- shoulder_abduction_adduction: signed lateral position of the upper arm"
        " (0 = neither out nor across, +pi/2 = straight out to the side away from"
        " the torso, -pi/2 = fully across the body).",
        "- shoulder_elevation: total upper-arm elevation from straight down (0 = arm"
        " at the side, pi/2 = horizontal, pi = overhead).",
        "- shoulder_internal_external_rotation: signed twist of the upper arm about"
        " its long axis (positive = internal, toward the body midline).",
    ]
)

_SYSTEM_PROMPT = (
    "You interpret a care recipient's spoken correction about how a robot is "
    "repositioning their left arm, and you write the corrected motion yourself. "
    "The utterance is ambiguous, so return the {n} most likely interpretations of "
    "what they are asking for. They must be BEHAVIOURALLY DISTINCT — {n} different "
    "motions, not {n} wordings of one motion — because the person keeps whichever "
    "one matches what they meant. Use the numeric scene context and the "
    "conversation history to resolve the wording, and keep every motion "
    "kinematically plausible for a seated person whose shoulder stays where it is. "
    "Return only a JSON object:\n"
    '{{"interpretations": [{{"interpretation": "<one sentence>", {payload}}}, ...], '
    '"reply": "<one sentence back to the person>"}}'
)

_OUTPUT_SPACES = ("positions", "anatomical")
_CONTEXT_LEVELS = ("pose", "plan", "full")
_ROW = {
    "positions": (
        6,
        "[elbow_x, elbow_y, elbow_z, wrist_x, wrist_y, wrist_z] in world metres",
    ),
    "anatomical": (len(FEATURE_NAMES), "[" + ", ".join(FEATURE_NAMES) + "] in radians"),
}


def _fmt(values: np.ndarray) -> str:
    return (
        "[" + ", ".join(f"{v:.3f}" for v in np.asarray(values, dtype=np.float64)) + "]"
    )


def feature_rows(trajectory: np.ndarray, context: Any) -> np.ndarray:
    """Anatomical features as a ``(T, 5)`` table in :data:`FEATURE_NAMES` order."""
    series = arm_feature_series(trajectory, context)
    return np.stack([np.atleast_1d(series[name]) for name in FEATURE_NAMES], axis=-1)


def anatomical_context_text(
    nominal_plan: np.ndarray,
    q_feedback: np.ndarray,
    rig: PlanningRig,
    rows: int = 15,
    context_level: str = "full",
) -> str:
    """The trajectory context this grounder interprets, in anatomical space.

    The nominal plan as a per-frame feature table (downsampled to about ``rows``
    rows), the pose the person is reacting to, and the arm and body landmark
    positions that anchor it in the world frame. The Cartesian goal is
    deliberately withheld so the LLM grounds the utterance, not the task; the
    nominal plan still shows where the robot was heading. ``context_level``
    trims the scene: ``"pose"`` is the current pose alone, ``"plan"`` adds the
    nominal-plan table, ``"full"`` adds the body landmarks.
    """
    arm_aa = q_to_arm_aa(nominal_plan, rig.fk.elbow_hinge_axis)
    plan_rows = feature_rows(nominal_plan, rig.context)
    plan_pos = rig.fk.fk_batch(arm_aa, rig.spine3_pos, rig.spine3_aa)
    current_pos = rig.fk.fk(
        q_to_arm_aa(q_feedback, rig.fk.elbow_hinge_axis), rig.spine3_pos, rig.spine3_aa
    )
    body = rig.body_pos if rig.body_pos is not None else rig.fk.tpose_all_joints
    stride = max(1, len(plan_rows) // rows)

    def joints(positions: np.ndarray) -> str:
        return (
            f"elbow {_fmt(positions[ELBOW_CHAIN_IDX])}, "
            f"wrist {_fmt(positions[WRIST_CHAIN_IDX])}"
        )

    lines = [
        _AXES,
        "",
        "Anatomical features, in the order every feature row lists them:",
        _FEATURE_GLOSSARY,
        "",
        "Current pose, the moment the person spoke. This is where the arm IS and "
        "where the correction starts; relative words like up, higher, a bit, more "
        "refer to this pose:",
        f"  features {_fmt(feature_rows(q_feedback, rig.context)[0])}",
        f"  {joints(current_pos)}",
    ]
    if context_level in ("plan", "full"):
        lines += [
            "",
            f"The robot's own plan from the current pose if nothing is corrected "
            f"({len(plan_rows)} frames; the frame column gives each row's index). "
            "It has NOT been executed: the arm is still at the current pose, and the "
            "person is objecting to this plan. Use it only to know where the robot "
            "was heading, not as the pose the correction starts from:",
            "frame," + ",".join(FEATURE_NAMES),
        ]
        lines += [
            f"{frame}," + ",".join(f"{value:.3f}" for value in plan_rows[frame])
            for frame in range(0, len(plan_rows), stride)
        ]
        lines += [
            f"plan start: {joints(plan_pos[0])}",
            f"plan end: {joints(plan_pos[-1])}",
        ]
    if context_level == "full":
        lines += [
            "",
            "Body landmarks: "
            + ", ".join(
                f"{name} {_fmt(np.asarray(body[index], dtype=np.float64))}"
                for name, index in LANDMARKS
            ),
        ]
    return "\n".join(lines)


def _rows(payload: dict[str, Any], key: str, width: int) -> np.ndarray | None:
    """The payload's ``(T, width)`` numeric table, or ``None`` if malformed."""
    try:
        rows = np.asarray(payload.get(key), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if rows.ndim != 2 or rows.shape[1] != width or not np.all(np.isfinite(rows)):
        return None
    return rows


def _interpolate(start: np.ndarray, waypoints: np.ndarray, n_frames: int) -> np.ndarray:
    """Sample ``n_frames`` points along straight lines through the waypoints."""
    knots = np.vstack([start[np.newaxis], waypoints])
    knot_t = np.linspace(0.0, 1.0, len(knots))
    frame_t = np.linspace(0.0, 1.0, n_frames)
    return np.stack(
        [np.interp(frame_t, knot_t, knots[:, i]) for i in range(knots.shape[1])],
        axis=1,
    )


class LlmTrajectoryGrounder(Grounder):
    """Candidates are trajectories the LLM wrote, converted but not repaired.

    ``n_waypoints=0`` asks for every frame of the motion; a positive value asks
    for that many waypoints, which are interpolated from the current pose in the
    output space. ``n_waypoints=1`` is the "goal of the correction" variant: the
    arm moves to the one waypoint and the episode loop's continuation rollout
    resumes toward the original goal from there. Interpretations that fail to
    translate are dropped, and a round that produces none falls back to the
    nominal plan.
    """

    def __init__(
        self,
        n_interpretations: int = 4,
        output_space: str = "positions",
        n_waypoints: int = 0,
        n_frames: int = 16,
        use_generator_rig: bool = False,
        model: str | None = None,
        context_level: str = "full",
    ) -> None:
        super().__init__()
        if output_space not in _OUTPUT_SPACES:
            raise ValueError(f"output_space must be one of {_OUTPUT_SPACES}.")
        if context_level not in _CONTEXT_LEVELS:
            raise ValueError(f"context_level must be one of {_CONTEXT_LEVELS}.")
        self.n_interpretations = n_interpretations
        self.output_space = output_space
        self.n_waypoints = n_waypoints
        self.n_frames = n_frames
        self.model = model
        self.context_level = context_level
        # Load the motion generator anyway (unused for grounding) so the rig
        # geometry (decoded pose spine3/body) matches the system arms exactly.
        if use_generator_rig:
            self.requires_generator = True

    def reset(
        self,
        rig: PlanningRig,
        user: SimulatedUser,
        seed: int,
        episode_dir: Path,
    ) -> None:
        super().reset(rig, user, seed, episode_dir)
        self._history: list[str] = []
        self._llm: Any = None
        self._round = 0

    @property
    def _key(self) -> str:
        return "waypoints" if self.n_waypoints else "frames"

    def _payload_contract(self) -> str:
        _, row = _ROW[self.output_space]
        if self.n_waypoints == 1:
            return (
                f'"waypoints": a list holding exactly one row, {row}, where the '
                "correction should end. The arm moves there in a straight line from "
                "its current pose, then the robot resumes toward its goal."
            )
        if self.n_waypoints:
            return (
                f'"waypoints": a list of at most {self.n_waypoints} rows, each {row}. '
                "The arm moves from its current pose through the waypoints in order, "
                "in a straight line between consecutive ones. Do not repeat the "
                "current pose as a waypoint: the first row is the first place the arm "
                "moves to."
            )
        return (
            f'"frames": a list of exactly {self.n_frames} rows, each {row}, giving the '
            "whole corrected motion frame by frame. Frame 0 must be the current pose."
        )

    def _interpret(self, text: str, scene: str) -> list[dict[str, Any]]:
        if self._llm is None:
            from uncertain_feedback.llm import (  # pylint: disable=import-outside-toplevel
                OpenAIModel,
            )

            self._llm = OpenAIModel(
                model=self.model or self.rig.cfg.llm_cost.model,
                system_prompt=_SYSTEM_PROMPT.format(
                    n=self.n_interpretations, payload=self._payload_contract()
                ),
                temperature=0.0,
                reasoning_effort="high",
            )
        prompt = (
            f"Scene:\n{scene}\n"
            "Conversation history:\n"
            + ("\n".join(self._history) if self._history else "(none)")
            + f'\nUser says: "{text}"'
        )
        for _ in range(2):
            data = extract_json_object(self._llm.get_full_output(prompt))
            if data is not None and isinstance(data.get("interpretations"), list):
                self._history.append(f"user: {text}")
                self._history.append(f"robot: {data.get('reply', '')}")
                return [
                    item
                    for item in data["interpretations"][: self.n_interpretations]
                    if isinstance(item, dict)
                ]
        self._history.append(f"user: {text}")
        return []

    def _save_interpretations(
        self, payloads: list[dict[str, Any]], candidates: dict[int, np.ndarray]
    ) -> None:
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        path = self._episode_dir / f"interpretations_{self._round:02d}.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {"interpretations": payloads, "n_candidates": len(candidates)},
                file,
                indent=2,
                default=str,
            )
        self._round += 1

    def _current_row(self, q_feedback: np.ndarray) -> np.ndarray:
        if self.output_space == "anatomical":
            return feature_rows(q_feedback, self.rig.context)[0]
        positions = self._chain(q_feedback)
        return np.concatenate([positions[ELBOW_CHAIN_IDX], positions[WRIST_CHAIN_IDX]])

    def _chain(self, q_feedback: np.ndarray) -> np.ndarray:
        rig = self.rig
        return rig.fk.fk(
            q_to_arm_aa(q_feedback, rig.fk.elbow_hinge_axis),
            rig.spine3_pos,
            rig.spine3_aa,
        )

    def _candidate(
        self, payload: dict[str, Any], q_feedback: np.ndarray
    ) -> np.ndarray | None:
        """One interpretation's payload as an arm trajectory, or ``None``."""
        rows = _rows(payload, self._key, _ROW[self.output_space][0])
        if rows is None:
            return None
        if self.n_waypoints:
            rows = _interpolate(
                self._current_row(q_feedback), rows[: self.n_waypoints], self.n_frames
            )
        rig = self.rig
        # Coincident elbow/wrist positions leave a zero-length limb vector, which
        # normalises to NaN and reaches scipy as a zero-norm quaternion. That is
        # a payload that fails to translate, not a run-ending error.
        try:
            if self.output_space == "anatomical":
                q = arm_q_from_features(
                    rows,
                    np.asarray(q_feedback, dtype=np.float64)[Q_CLAVICLE],
                    rig.context,
                )
                return q_to_arm_aa(q, rig.fk.elbow_hinge_axis)
            # Frame completion, not repair: the joints upstream of the elbow are
            # unactuated, so they hold the pose the correction starts from.
            chains = np.repeat(self._chain(q_feedback)[np.newaxis], len(rows), axis=0)
            chains[:, ELBOW_CHAIN_IDX] = rows[:, :3]
            chains[:, WRIST_CHAIN_IDX] = rows[:, 3:]
            return rig.fk.arm_aa_from_positions_batch(chains, rig.spine3_aa)
        except ValueError:
            return None

    def ground(
        self,
        text: str,
        q_feedback: np.ndarray,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        payloads = self._interpret(
            text,
            anatomical_context_text(
                nominal_plan, q_feedback, self.rig, context_level=self.context_level
            ),
        )
        candidates: dict[int, np.ndarray] = {}
        for payload in payloads:
            candidate = self._candidate(payload, q_feedback)
            if candidate is not None:
                candidates[len(candidates)] = candidate
        if not candidates:
            candidates = {0: q_to_arm_aa(nominal_plan, self.rig.fk.elbow_hinge_axis)}
        self._save_interpretations(payloads, candidates)
        chosen_label, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=chosen_label,
            magnitude=magnitude,
            correction_traj=scale_trajectory(candidates[chosen_label], magnitude),
        )
