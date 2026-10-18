"""Direct-trajectory pure-agent grounder: the LLM writes the correction itself.

This grounder produces the same feedback grounding as the system but with no
motion prior: one LLM call per round returns the most likely interpretations of
the utterance, each carrying a trajectory payload converted into one candidate
motion. One class covers the dense, sparse-waypoint, and single-waypoint
variants in either output space (elbow/wrist positions or the five anatomical
angles), and a free-form mode where the LLM picks, per interpretation, the
space, waypoints or frames, how many, and optionally the timing. The LLM output is only *converted* into the planner's representation —
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
    required_llm_model,
)
from evaluation.approaches.grounders.nominal import stop_pose
from evaluation.metrics.grounding.structs import GroundingResult
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_feature_series,
)
from uncertain_feedback.planners.mpc.config import MpcRunConfig
from uncertain_feedback.planners.mpc.costs import extract_json_object
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import ELBOW_CHAIN_IDX, WRIST_CHAIN_IDX
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

# The free-form mode's own prompt: the form of each motion is left to the LLM.
_FREE_FORM_SYSTEM_PROMPT = (
    "You interpret a care recipient's spoken correction about how a robot is "
    "repositioning their left arm, and you write the corrected motion yourself. "
    "The utterance is ambiguous, so return the {n} most likely interpretations of "
    "what they are asking for. Try to make them distinct motions, since the person "
    "keeps whichever one matches what they meant. Use the numeric scene context and "
    "the conversation history to resolve the wording, and keep every motion "
    "kinematically plausible for a seated person whose shoulder stays where it is. "
    "After the correction, the robot resumes toward its goal.\n"
    "\n"
    "Write each motion in whatever form suits it; the interpretations need not "
    "share a form.\n"
    '- space: "positions", each row {positions_row}; or "anatomical", each row '
    "{anatomical_row}.\n"
    '- kind: "waypoints", the places the arm passes through after its current '
    "pose, in order (at most {max_waypoints}; do not repeat the current pose), "
    'joined by straight lines in the chosen space; or "frames", every frame of the '
    "motion (at most {max_frames}), frame 0 being the current pose.\n"
    "- times (optional, waypoints only): when the arm reaches each waypoint, one "
    "strictly increasing fraction of the motion in (0, 1] per waypoint. Evenly "
    "spaced if omitted.\n"
    "- n_frames (optional, waypoints only): how many frames the motion lasts, 2 to "
    "{max_frames}; {n_frames} if omitted. A frame is one time step of the robot's "
    "plan shown in the scene.\n"
    "\n"
    "Return only a JSON object:\n"
    '{{"interpretations": [{{"interpretation": "<one sentence>", "space": ..., '
    '"kind": ..., "rows": [[...], ...], "times": [...], "n_frames": ...}}, ...], '
    '"reply": "<one sentence back to the person>"}}'
)

_OUTPUT_SPACES = ("positions", "anatomical", "free")
_KINDS = ("waypoints", "frames")
# Free-form caps: they bound the tokens a reply can take. Rows past a cap are
# dropped, as the fixed waypoint modes drop rows past ``n_waypoints``.
MAX_FREE_WAYPOINTS = 20
MAX_FREE_FRAMES = 100
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
    human: Human,
    rows: int = 15,
    context_level: str = "full",
) -> str:
    """The trajectory context this grounder interprets, in anatomical space.

    The nominal plan as a per-frame feature table (downsampled to about ``rows``
    rows), the pose the person is reacting to (``human.q``), and the arm and body landmark
    positions that anchor it in the world frame. The Cartesian goal is
    deliberately withheld so the LLM grounds the utterance, not the task; the
    nominal plan still shows where the robot was heading. ``context_level``
    trims the scene: ``"pose"`` is the current pose alone, ``"plan"`` adds the
    nominal-plan table, ``"full"`` adds the body landmarks.
    """
    arm_aa = human.arm_aa_from_q(nominal_plan)
    plan_rows = feature_rows(nominal_plan, human)
    plan_pos = human.fk.fk_batch(arm_aa, human.spine3_pos, human.spine3_aa)
    current_pos = human.fk.fk(
        human.arm_aa_from_q(human.q), human.spine3_pos, human.spine3_aa
    )
    body = human.posture
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
        f"  features {_fmt(feature_rows(human.q, human)[0])}",
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


def _interpolate(
    start: np.ndarray,
    waypoints: np.ndarray,
    n_frames: int,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """Sample ``n_frames`` points along straight lines through the waypoints.

    ``times`` places each waypoint at that fraction of the motion (the start
    is at 0); evenly spaced when ``None``. After the last waypoint the arm holds.
    """
    knots = np.vstack([start[np.newaxis], waypoints])
    knot_t = (
        np.linspace(0.0, 1.0, len(knots))
        if times is None
        else np.concatenate([[0.0], times])
    )
    frame_t = np.linspace(0.0, 1.0, n_frames)
    return np.stack(
        [np.interp(frame_t, knot_t, knots[:, i]) for i in range(knots.shape[1])],
        axis=1,
    )


def _times(value: Any, count: int) -> np.ndarray | None:
    """``count`` strictly increasing fractions in (0, 1], or ``None``."""
    try:
        times = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if times.shape != (count,) or not np.all(np.isfinite(times)):
        return None
    if times[0] <= 0.0 or times[-1] > 1.0 or np.any(np.diff(times) <= 0.0):
        return None
    return times


def _frame_count(value: Any, default: int) -> int:
    """A requested motion length, held to [2, MAX_FREE_FRAMES]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not np.isfinite(value):
        return default
    return int(np.clip(round(value), 2, MAX_FREE_FRAMES))


class LlmTrajectoryGrounder(Grounder):
    """Candidates are trajectories the LLM wrote, converted but not repaired.

    ``n_waypoints=0`` asks for every frame of the motion; a positive value asks
    for that many waypoints, which are interpolated from the current pose in the
    output space. ``n_waypoints=1`` is the "goal of the correction" variant: the
    arm moves to the one waypoint and the episode loop's continuation rollout
    resumes toward the original goal from there. ``output_space="free"`` lets
    each interpretation choose its own form: ``space`` (positions or
    anatomical), ``kind`` (waypoints or frames), the ``rows``, and for
    waypoints optional ``times`` and ``n_frames`` (``n_frames`` is then the
    default length); waypoints are interpolated from the current pose in the
    chosen space. Interpretations that fail to translate are dropped. A round
    that produces none offers the nominal plan for the persona to judge but
    does not replay it: the arm holds at the stop pose and replans. Each round's
    ``interpretations_NN.json`` records the form of every interpretation and
    whether each LLM call parsed (an unparseable reply is retried once).
    """

    def __init__(
        self,
        n_interpretations: int = 4,
        output_space: str = "positions",
        n_waypoints: int = 0,
        n_frames: int = 16,
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
        self._llm: Any = None
        self._round = 0
        self._calls: list[dict[str, Any]] = []

    def reset(
        self,
        cfg: MpcRunConfig,
        gen: MotionGenerator | None,
        user: SimulatedUser,
        seed: int,
        episode_dir: Path,
    ) -> None:
        super().reset(cfg, gen, user, seed, episode_dir)
        self._llm = None
        self._round = 0

    def _system_prompt(self) -> str:
        if self.output_space == "free":
            return _FREE_FORM_SYSTEM_PROMPT.format(
                n=self.n_interpretations,
                positions_row=_ROW["positions"][1],
                anatomical_row=_ROW["anatomical"][1],
                max_waypoints=MAX_FREE_WAYPOINTS,
                max_frames=MAX_FREE_FRAMES,
                n_frames=self.n_frames,
            )
        return _SYSTEM_PROMPT.format(
            n=self.n_interpretations, payload=self._payload_contract()
        )

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
                model=self.model or required_llm_model(self.cfg),
                system_prompt=self._system_prompt(),
                temperature=0.0,
                reasoning_effort="high",
            )
        prompt = (
            f"Scene:\n{scene}\n"
            "Conversation history:\n"
            + ("\n".join(self._history) if self._history else "(none)")
            + f'\nUser says: "{text}"'
        )
        self._calls = []
        for _ in range(2):
            data = extract_json_object(self._llm.get_full_output(prompt))
            parsed = data is not None and isinstance(data.get("interpretations"), list)
            self._calls.append({"parsed": parsed})
            if data is not None and parsed:
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
        self,
        payloads: list[dict[str, Any]],
        forms: list[dict[str, Any]],
        candidates: dict[int, np.ndarray],
    ) -> None:
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        path = self._episode_dir / f"interpretations_{self._round:02d}.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "interpretations": payloads,
                    "forms": forms,
                    "n_candidates": len(candidates),
                    "calls": self._calls,
                },
                file,
                indent=2,
                default=str,
            )
        self._round += 1

    def _current_row(self, human: Human, space: str) -> np.ndarray:
        if space == "anatomical":
            return feature_rows(human.q, human)[0]
        positions = self._chain(human)
        return np.concatenate([positions[ELBOW_CHAIN_IDX], positions[WRIST_CHAIN_IDX]])

    def _chain(self, human: Human) -> np.ndarray:
        return human.fk.fk(
            human.arm_aa_from_q(human.q),
            human.spine3_pos,
            human.spine3_aa,
        )

    def _form(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """The payload's space, kind, rows, times and length; ``None`` if malformed."""
        if self.output_space != "free":
            kind = "waypoints" if self.n_waypoints else "frames"
            rows = _rows(payload, kind, _ROW[self.output_space][0])
            if rows is None:
                return None
            if self.n_waypoints:
                rows = rows[: self.n_waypoints]
            return {
                "space": self.output_space,
                "kind": kind,
                "rows": rows,
                "times": None,
                "n_frames": self.n_frames if self.n_waypoints else len(rows),
            }
        space, free_kind = payload.get("space"), payload.get("kind")
        if space not in _ROW or free_kind not in _KINDS:
            return None
        kind = str(free_kind)
        rows = _rows(payload, "rows", _ROW[space][0])
        if rows is None or len(rows) == 0:
            return None
        if kind == "frames":
            rows = rows[:MAX_FREE_FRAMES]
            return {
                "space": space,
                "kind": kind,
                "rows": rows,
                "times": None,
                "n_frames": len(rows),
            }
        rows = rows[:MAX_FREE_WAYPOINTS]
        return {
            "space": space,
            "kind": kind,
            "rows": rows,
            "times": _times(payload.get("times"), len(rows)),
            "n_frames": _frame_count(payload.get("n_frames"), self.n_frames),
        }

    def _candidate(
        self, payload: dict[str, Any], human: Human
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """One interpretation's payload as an arm trajectory, and its form record."""
        form = self._form(payload)
        if form is None:
            return None, {"converted": False}
        rows = form["rows"]
        if form["kind"] == "waypoints":
            rows = _interpolate(
                self._current_row(human, form["space"]),
                rows,
                form["n_frames"],
                form["times"],
            )
        record = {
            "space": form["space"],
            "kind": form["kind"],
            "n_rows": len(form["rows"]),
            "times": form["times"] is not None,
            "times_given": "times" in payload,
            "n_frames": form["n_frames"],
        }
        # Coincident elbow/wrist positions leave a zero-length limb vector, which
        # normalises to NaN and reaches scipy as a zero-norm quaternion. That is
        # a payload that fails to translate, not a run-ending error.
        try:
            if form["space"] == "anatomical":
                trajectory = human.arm_aa_from_q(human.q_from_features(rows))
            else:
                # Frame completion, not repair: the joints upstream of the elbow
                # are unactuated, so they hold the pose the correction starts from.
                chains = np.repeat(self._chain(human)[np.newaxis], len(rows), axis=0)
                chains[:, ELBOW_CHAIN_IDX] = rows[:, :3]
                chains[:, WRIST_CHAIN_IDX] = rows[:, 3:]
                trajectory = human.arm_aa_from_q(human.ik_q_from_positions(chains))
        except ValueError:
            return None, {**record, "converted": False}
        return trajectory, {**record, "converted": True}

    def ground(
        self,
        text: str,
        human: Human,
        nominal_plan: np.ndarray,
        cluster_selector: ClusterSelector,
    ) -> GroundingResult:
        payloads = self._interpret(
            text,
            anatomical_context_text(
                nominal_plan, human, context_level=self.context_level
            ),
        )
        candidates: dict[int, np.ndarray] = {}
        forms: list[dict[str, Any]] = []
        for payload in payloads:
            candidate, form = self._candidate(payload, human)
            if candidate is not None:
                form["label"] = len(candidates)
                candidates[len(candidates)] = candidate
            forms.append(form)
        fallback = not candidates
        if fallback:
            candidates = {0: human.arm_aa_from_q(nominal_plan)}
        self._save_interpretations(payloads, forms, candidates)
        chosen_label, magnitude = cluster_selector(candidates)
        return GroundingResult(
            candidates=candidates,
            chosen_label=chosen_label,
            magnitude=magnitude,
            correction_traj=(
                stop_pose(human)
                if fallback
                else scale_trajectory(candidates[chosen_label], magnitude)
            ),
        )
