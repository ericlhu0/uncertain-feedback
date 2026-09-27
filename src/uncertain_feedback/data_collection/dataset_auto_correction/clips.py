"""Correction clips: oracle-corrected branches off randomly sampled reaches.

Stage (a) of the correction-clip finetune pipeline. Every run draws its own
scenario — a start arm configuration and a Cartesian goal, both sampled from the
anatomical feature box (:func:`sample_arm_q`) — and rolls a naive MPC reach
between them. It then samples a hidden comfort bound *the rollout crosses* — a
crossing frame is drawn and the bound placed in the gap the rollout opens there
(:func:`sample_violating_bound`), so the naive path is guaranteed to violate it
— replans from the induced trigger step under the oracle cost, and saves the
whole rollout.

A run's naive approach and oracle rollout join into one continuous motion
(:func:`motion_frames`), and a clip is a *cut* out of it: ``n_prefix`` frames of
history up to an anchor, then a window of what follows
(:func:`assemble_clip`). The sampled default anchors the cut at the trigger, but
the labeling UI can drag it anywhere along the motion — later stretches of the
same rollout are often the interesting ones — and :meth:`ClipSource.cut` rewrites
the clip with no replan.

The prefix is the same arm history inference pins as conditioning
(:mod:`uncertain_feedback.planners.run`), so a checkpoint fine-tuned on these
clips sees the distribution it is queried with. That holds at any anchor: pinning
recent *corrected* history is still the arm's real recent history.

Only trajectories are written, never rendered video — video dominated the output
size (3.5 MB of a 3.9 MB 32-clip set). ``label.py`` previews a run
in the browser from the run's ``naive.npy`` plus its ``continuation.npy``, on the
person the manifest's planner config describes, so preview needs neither the
MDM environment nor a GPU.

Stage (b) — this folder's :mod:`build_dataset` — turns the hand-labeled manifest
into a HumanML3D-format finetune dataset.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from uncertain_feedback.motion_generators.mdm.mdm_api import N_PREFIX_FRAMES
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_feature_series,
)
from uncertain_feedback.planners.mpc.config import (
    MpcRunConfig,
    cfg_with_goal,
    load_mpc_config,
)
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    base_extra_costs,
)
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.rollout import goal_reach, rollout_to_goal
from uncertain_feedback.simulated_users import (
    HiddenBound,
    HiddenCostTerm,
    SimulatedUser,
    first_violation_step,
    violation_metrics,
)
from uncertain_feedback.simulated_users.personas import (
    DEFAULT_ARM_JOINT_LIMITS,
    UNRESTRICTED,
)

_LOG = "[correction-clips]"
_MAX_SAMPLE_ATTEMPTS = 50
_MAX_SCENARIO_ATTEMPTS = 10

# Uniform ranges (radians, FEATURE_NAMES order) the per-run start arm and goal
# arm are drawn from. They bracket the span the low1 naive rollout swept, which
# is the region the anatomical joint box and the seated body both allow.
START_FEATURE_RANGES: tuple[tuple[float, float], ...] = (
    (0.5, 2.0),
    (-0.3, 0.9),
    (-0.1, 0.9),
    (0.3, 1.7),
    (-0.5, 1.0),
)

# Clip = n_prefix + window frames. The floor keeps it past the t2m loader's
# min-40 filter, the ceiling under MDM's 196-frame cap.
MIN_WINDOW = 42
MAX_WINDOW = 180


@dataclass(frozen=True)
class CorrectionClipConfig:
    """What to generate: ``n_runs`` corrected branches, one scenario each.

    ``max_angle_delta`` overrides the planner config's action-sampling spread and
    is the one knob controlling how big and how fast a clip's motion is. It is a
    std dev, not a per-step cap, so it sets distance travelled per frame; since a
    clip is a fixed frame budget, halving it halves both the speed and the ground
    covered. On the low1 start: 0.0025 gives a 4.2 s reach and 0.351 m of wrist
    path per clip at 0.0079 m/frame, 0.00125 (the default) gives 8.2 s and
    0.186 m at 0.0042 m/frame. Clip length and padding are unaffected.

    ``trigger_window`` counts naive frames, so it has to scale with
    ``max_angle_delta``: (12, 100) suits the 165-frame reach at 0.00125, (6, 50)
    the 85-frame reach at 0.0025. ``min_goal_distance`` scales with it too — it
    is the floor on start-wrist-to-goal separation, and at 0.25 m the shortest
    accepted reach is roughly 60 naive frames at the default spread, leaving the
    trigger window room to land inside every run.

    Clips are paced more slowly than the 0.01 demo the finetuned checkpoint is
    queried in — MDM output is tracked as a path (playback advances on
    proximity), so the pacing costs nothing downstream, but the pinned prefix a
    clip carries is slower than the one inference pins.
    """

    config_path: Path
    out_dir: Path
    n_runs: int = 0
    seed: int = 0
    features: tuple[str, ...] = FEATURE_NAMES
    bound_types: tuple[str, ...] = ("upper_bound", "lower_bound")
    trigger_window: tuple[int, int] = (12, 100)
    correction_frames: tuple[int, int] = (42, 56)
    max_angle_delta: float = 0.00125
    min_goal_distance: float = 0.25


@dataclass(frozen=True)
class SampledBound:
    """One hidden bound plus the trigger step it induces on the naive rollout."""

    user: SimulatedUser
    feature: str
    bound_type: str
    value: float
    peak_violation: float
    crossing_step: int
    trigger_step: int


def sample_arm_q(rng: np.random.Generator, human: Human) -> np.ndarray:
    """Draw one anatomically valid arm state from :data:`START_FEATURE_RANGES`.

    The arm keeps ``human``'s current clavicle. The features are drawn
    independently and inverted by :meth:`Human.q_from_features`, which resolves the over-determined
    flexion/abduction/elevation triple rather than honouring all three — so the
    realized features differ from the drawn ones. The draw is for coverage, not
    for hitting a target pose. Draws landing outside
    :data:`DEFAULT_ARM_JOINT_LIMITS` are redrawn, the same check a transplanted
    clip has to pass.
    """
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        features = np.array(
            [rng.uniform(low, high) for low, high in START_FEATURE_RANGES]
        )
        q = human.q_from_features(features[None])[0]
        arm_aa = human.arm_aa_from_q(q)
        if all(
            float(limit.violation(arm_aa).max()) <= 0.0
            for limit in DEFAULT_ARM_JOINT_LIMITS
        ):
            return q
    raise RuntimeError(
        f"No arm configuration stayed inside the joint box within "
        f"{_MAX_SAMPLE_ATTEMPTS} draws — narrow START_FEATURE_RANGES."
    )


def synthetic_user(feature: str, bound_type: str, value: float) -> SimulatedUser:
    """Synthesize a one-bound user; not drawn from the persona library.

    The returned ``SimulatedUser`` exists only to carry the sampled bound into
    ``HiddenCostTerm`` and ``first_violation_step``. Its ``joint_limits`` are the
    shared anatomical box, so the oracle replan stays in range; the single
    ``HiddenBound`` is the only comfort restriction it expresses.
    """
    bound = HiddenBound(
        feature=feature,
        bound_type=bound_type,
        high=value if bound_type == "upper_bound" else None,
        low=value if bound_type == "lower_bound" else None,
    )
    return SimulatedUser(
        name=f"synthetic_{feature}",
        description="",
        feedback_text="",
        bounds=(bound,),
        joint_limits=DEFAULT_ARM_JOINT_LIMITS,
    )


def sample_violating_bound(
    rng: np.random.Generator,
    naive_q: np.ndarray,
    human: Human,
    cfg: CorrectionClipConfig,
    threshold: float,
) -> SampledBound:
    """Sample a hidden bound the naive rollout is guaranteed to cross.

    The crossing step is drawn directly and the bound placed in the gap between
    the feature's running extremum over the frames before it and its value at
    it. Every earlier frame therefore has positive clearance, the crossing step
    is the first frame on the wrong side, and the trigger the bound induces
    lands at or shortly after it, as the violation builds through ``threshold``
    — the reverse of placing the bound past a value the rollout already passed,
    which dragged the trigger back to the start of the window. A step setting no
    running record opens no gap and is redrawn.

    :func:`first_violation_step` on the naive rollout stays the accept test: it
    also sees the joint-limit term the gap arithmetic ignores, and rejects
    bounds whose violation never reaches ``threshold``. Samples triggering
    outside ``cfg.trigger_window`` are rejected too.
    """
    feats = arm_feature_series(naive_q, human)
    low, high = cfg.trigger_window
    high = min(high, len(naive_q) - 1)
    if low >= high:
        raise ValueError(
            f"trigger_window {cfg.trigger_window} is empty for a "
            f"{len(naive_q)}-frame naive rollout."
        )
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        feature = str(rng.choice(cfg.features))
        bound_type = str(rng.choice(cfg.bound_types))
        step = int(rng.integers(low, high + 1))
        series = feats[feature]
        if bound_type == "upper_bound":
            floor = float(series[:step].max())
            if series[step] <= floor:
                continue
            value = float(rng.uniform(floor, series[step]))
            peak_violation = float(series.max()) - value
        else:
            ceiling = float(series[:step].min())
            if series[step] >= ceiling:
                continue
            value = float(rng.uniform(series[step], ceiling))
            peak_violation = value - float(series.min())
        user = synthetic_user(feature, bound_type, value)
        trigger = first_violation_step(user, human, naive_q, threshold)
        if trigger is not None and low <= trigger <= high:
            return SampledBound(
                user=user,
                feature=feature,
                bound_type=bound_type,
                value=value,
                peak_violation=peak_violation,
                crossing_step=step,
                trigger_step=trigger,
            )
    raise RuntimeError(
        f"No sampled bound triggered inside {cfg.trigger_window} within "
        f"{_MAX_SAMPLE_ATTEMPTS} attempts — widen trigger_window."
    )


def assemble_clip(
    motion: np.ndarray, anchor: int, window: int, n_prefix: int
) -> tuple[np.ndarray, int]:
    """Cut a clip out of one continuous motion at ``anchor``.

    ``anchor`` is the index of the frame the pinned prefix *ends* on — the state
    inference pins last — so the clip is the ``n_prefix`` frames up to and
    including it, then ``window`` frames of the motion that follows. The prefix is
    left-padded by repeating the oldest frame when the anchor is earlier than
    ``n_prefix - 1``, the rule ``planners/run.py`` uses for the inference prefix.

    Cutting from one array rather than splicing naive-plus-continuation is what
    lets the clip be moved: at ``anchor = trigger`` the prefix is naive history
    and the window is the start of the correction (the sampled default), while a
    later anchor pins recent *corrected* history and describes motion further
    along — still a legitimate clip, since inference pins whatever the arm just
    did. Returns the clip and how many final frames were held because the motion
    ran out before ``window`` was filled.
    """
    motion = np.asarray(motion, dtype=np.float64)
    prefix = list(motion[max(0, anchor - n_prefix + 1) : anchor + 1])
    prefix = [prefix[0]] * (n_prefix - len(prefix)) + prefix
    tail = list(motion[anchor + 1 : anchor + 1 + window])
    pad_frames = window - len(tail)
    hold = tail[-1] if tail else motion[anchor]
    tail = tail + [hold] * pad_frames
    return np.asarray(prefix + tail, dtype=np.float64), pad_frames


def motion_frames(
    naive_q: np.ndarray, continuation_q: np.ndarray, trigger: int
) -> tuple[np.ndarray, int]:
    """The run's whole motion — naive approach then the full oracle rollout.

    ``continuation_q`` restarts from the trigger state, so its duplicate first
    frame is dropped. Returns the frames and ``transition``, the index of the last
    naive frame (where the correction takes over). Every clip for this run is cut
    out of this one array, so UI indices, clip indices and manifest indices all
    live in the same coordinate system.
    """
    naive_q = np.asarray(naive_q, dtype=np.float64)
    approach = naive_q[: trigger + 1]
    return np.concatenate([approach, continuation_q[1:]]), len(approach) - 1


def write_feature_csv(path: Path, q: np.ndarray, human: Human) -> None:
    """Write the per-frame anatomical features of an arm trajectory."""
    feats = arm_feature_series(q, human)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", *FEATURE_NAMES])
        for frame in range(len(q)):
            writer.writerow([frame, *(feats[name][frame] for name in FEATURE_NAMES)])


def clip_bounds(anchor: int, window: int, n_frames: int) -> tuple[int, int]:
    """Clamp ``(anchor, window)`` so the clip fits inside an ``n_frames`` motion.

    The anchor is held back far enough to leave ``MIN_WINDOW`` real frames after
    it, then the window is trimmed to whatever room remains. Bounding the anchor
    this way is what keeps a clip dragged to the very end of the motion from
    becoming mostly held frames: the window shrinks toward real content instead of
    padding out past the last frame. ``MIN_WINDOW`` keeps the clip above the t2m
    loader's minimum length, ``MAX_WINDOW`` below MDM's frame cap.
    """
    anchor = int(np.clip(anchor, 0, max(0, n_frames - 1 - MIN_WINDOW)))
    room = n_frames - 1 - anchor
    return anchor, int(
        np.clip(window, MIN_WINDOW, max(MIN_WINDOW, min(MAX_WINDOW, room)))
    )


@dataclass(frozen=True)
class ClipSource:
    """Everything needed to sample and correct one more run.

    Holds only the immutable per-set context — the person, the cost stack and
    the planner config; the scenario itself is drawn per run — so runs can be
    produced one at a time and out of order, which is how the labeling UI
    generates the next one while the previous is being captioned. The person is
    the planner config's (its pose file and ``arm:``), and every sampled arm
    keeps its clavicle. Nothing here loads the motion generator, so on-demand
    generation needs neither MDM nor a GPU.
    """

    out_dir: Path
    cfg: CorrectionClipConfig
    run_cfg: MpcRunConfig
    human: Human
    base: CompositeTrajectoryCost
    n_prefix: int
    threshold: float

    def sample_scenario(
        self, rng: np.random.Generator, label: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a reach: a Cartesian goal and the naive rollout that chases it.

        Both ends of the reach are drawn with :func:`sample_arm_q`, so the goal is
        always a wrist position some arm configuration can hold. A draw is
        rejected when the two ends sit closer than ``cfg.min_goal_distance``
        (cheap, before any rollout), when the naive rollout never reaches the
        goal, or when it is too short for ``cfg.trigger_window`` to contain a
        trigger. Returns the goal and the naive rollout, whose first frame is the
        sampled start arm.
        """
        human = self.human
        for _ in range(_MAX_SCENARIO_ATTEMPTS):
            q0 = sample_arm_q(rng, human)
            goal = human.wrist_from_q(sample_arm_q(rng, human))
            if (
                float(np.linalg.norm(goal - human.wrist_from_q(q0)))
                < self.cfg.min_goal_distance
            ):
                continue
            naive = rollout_to_goal(
                cfg_with_goal(self.run_cfg, goal),
                human.reset_human_with_q(q0),
                goal,
                self.base,
                progress_label=f"{label} naive",
                log_prefix=_LOG,
            ).history
            reached = goal_reach(human, self.run_cfg, naive, goal)["reached"]
            if reached and len(naive) - 1 > self.cfg.trigger_window[0]:
                return goal, naive
        raise RuntimeError(
            f"No sampled scenario produced a usable reach within "
            f"{_MAX_SCENARIO_ATTEMPTS} attempts — lower min_goal_distance or "
            "widen START_FEATURE_RANGES."
        )

    def generate(self, index: int) -> dict[str, Any]:
        """Produce run ``index``: write its clip files, return its manifest row.

        Seeded on ``(seed, index)`` rather than a streaming generator, so a run is
        reproducible from its index alone whether it came from a batch or from a
        click on Next. The draw order — scenario first, then the hidden bound —
        is part of that contract.
        """
        rng = np.random.default_rng([self.cfg.seed, index])
        goal, naive = self.sample_scenario(rng, f"run {index}")
        sampled = sample_violating_bound(
            rng, naive, self.human, self.cfg, self.threshold
        )
        window = int(
            rng.integers(
                self.cfg.correction_frames[0], self.cfg.correction_frames[1] + 1
            )
        )
        oracle_costs = CompositeTrajectoryCost(
            [
                *self.base.terms(),
                HiddenCostTerm(user=sampled.user, human=self.human),
            ]
        )
        goal_cfg = cfg_with_goal(self.run_cfg, goal)
        continuation = rollout_to_goal(
            goal_cfg,
            self.human.reset_human_with_q(naive[sampled.trigger_step]),
            goal,
            oracle_costs,
            progress_label=f"run {index} continuation",
            log_prefix=_LOG,
        ).history
        run_id = f"run_{index:03d}"
        run_dir = self.out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "naive.npy", naive)
        # The whole continuation, not just the window a clip keeps: the labeling
        # UI shows where the correction was still heading, and re-cuts against it.
        np.save(run_dir / "continuation.npy", continuation)
        row = {
            "run_id": run_id,
            "captions": [],
            "goal": goal.tolist(),
            "feature": sampled.feature,
            "bound_type": sampled.bound_type,
            "bound_value": sampled.value,
            "peak_violation": sampled.peak_violation,
            "crossing_step": sampled.crossing_step,
            "trigger_step": sampled.trigger_step,
            "continuation_frames": int(len(continuation)),
            "continuation_reach": goal_reach(self.human, goal_cfg, continuation, goal),
            "clip_file": f"{run_id}/clip.npy",
            "naive_file": f"{run_id}/naive.npy",
            "continuation_file": f"{run_id}/continuation.npy",
            "features_file": f"{run_id}/clip_features.csv",
        }
        motion, _ = motion_frames(naive, continuation, sampled.trigger_step)
        row.update(self.cut(row, motion, anchor=sampled.trigger_step, window=window))
        print(
            f"{_LOG} {run_id}: {sampled.bound_type} on {sampled.feature} "
            f"@ {sampled.value:.3f} -> trigger {sampled.trigger_step}, "
            f"{row['clip_frames']} clip frames",
            flush=True,
        )
        return row

    def cut(
        self, row: dict[str, Any], motion: np.ndarray, anchor: int, window: int
    ) -> dict[str, Any]:
        """Re-cut ``row``'s clip at ``(anchor, window)``; return the changed fields.

        Writes ``clip.npy`` and ``clip_features.csv``, so the same call serves both
        the sampled default at generation time and a drag in the labeling UI. The
        violation summary is recomputed from the row's own recorded bound, since
        moving the window changes how much of it the bound is violated over.
        """
        anchor, window = clip_bounds(anchor, window, len(motion))
        clip, pad_frames = assemble_clip(motion, anchor, window, self.n_prefix)
        run_dir = self.out_dir / row["run_id"]
        np.save(run_dir / "clip.npy", clip)
        write_feature_csv(run_dir / "clip_features.csv", clip, self.human)
        user = synthetic_user(row["feature"], row["bound_type"], row["bound_value"])
        return {
            "clip_anchor": anchor,
            "correction_frames": window,
            "pad_frames": pad_frames,
            "clip_frames": int(len(clip)),
            # Measured on the described window only: the pinned prefix is
            # conditioning, and at the default anchor it is the naive frames that
            # violated the bound in the first place.
            "window_violation": violation_metrics(
                user, self.human, clip[self.n_prefix :]
            ),
        }


def new_session_dir(base_dir: Path) -> Path:
    """Fork a fresh labeling session off the clip set in ``base_dir``.

    Every labeling session gets its own directory, runs and manifest, so starting
    one can never overwrite an earlier session's captions. Stage (b) reads a
    session exactly like it reads a clip set.

    The seed is the session's own timestamp. Runs are seeded on ``(seed, index)``,
    so two sessions off the same base would otherwise sample the same scenarios
    and replan the same corrections from run 0 onward.
    """
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{stamp}"
    session_dir.mkdir(parents=True)
    seed = int(stamp.replace("_", ""))
    manifest["seed"] = seed
    manifest["base_dir"] = str(base_dir)
    manifest["clip_config"]["seed"] = seed
    manifest["clip_config"]["n_runs"] = 0
    manifest["runs"] = []
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return session_dir


def clip_source_from_dir(out_dir: Path) -> ClipSource:
    """Rebuild a :class:`ClipSource` from an existing clip set, without MDM.

    The person comes from the planner config the manifest records.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    stored = manifest["clip_config"]
    cfg = CorrectionClipConfig(
        config_path=Path(stored["config_path"]),
        out_dir=out_dir,
        n_runs=stored["n_runs"],
        seed=stored["seed"],
        features=tuple(stored["features"]),
        bound_types=tuple(stored["bound_types"]),
        trigger_window=(stored["trigger_window"][0], stored["trigger_window"][1]),
        correction_frames=(
            stored["correction_frames"][0],
            stored["correction_frames"][1],
        ),
        max_angle_delta=stored["max_angle_delta"],
        min_goal_distance=stored["min_goal_distance"],
    )
    run_cfg = load_mpc_config(cfg.config_path)
    human = Human(pose=run_cfg.pose, arm=run_cfg.arm)
    return ClipSource(
        out_dir=out_dir,
        cfg=cfg,
        run_cfg=replace(run_cfg, max_angle_delta=cfg.max_angle_delta, seed=cfg.seed),
        human=human,
        base=base_extra_costs(run_cfg.costs, human, UNRESTRICTED),
        n_prefix=manifest["n_prefix_frames"],
        threshold=manifest["trigger_threshold"],
    )


def generate_correction_clips(cfg: CorrectionClipConfig) -> Path:
    """Write the base artifacts plus ``cfg.n_runs`` corrected branches.

    ``n_runs=0`` writes only an empty manifest, which is all the labeling UI
    needs to generate runs on demand.

    Refuses to write into a directory that already holds a clip set: the manifest
    is rewritten wholesale, so doing so would blank its captions and orphan the
    labeling sessions underneath it.
    """
    if (cfg.out_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{cfg.out_dir} already holds a clip set. Generating into it would "
            "blank its captions and its labeling sessions — pass a new --out_dir."
        )
    loaded = load_mpc_config(cfg.config_path)
    assert loaded.cartesian is not None
    human = Human(pose=loaded.pose, arm=loaded.arm)
    n_prefix = N_PREFIX_FRAMES
    threshold = loaded.corrections.trigger_threshold
    run_cfg = replace(loaded, max_angle_delta=cfg.max_angle_delta, seed=cfg.seed)
    # UNRESTRICTED is `bounds=()`: it carries the anatomical joint box into the
    # cost stack and nothing else. No persona's comfort bounds enter this
    # pipeline — every bound is sampled per run off the naive rollout this base
    # cost produces.
    base = base_extra_costs(run_cfg.costs, human, UNRESTRICTED)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    source = ClipSource(
        out_dir=cfg.out_dir,
        cfg=cfg,
        run_cfg=run_cfg,
        human=human,
        base=base,
        n_prefix=n_prefix,
        threshold=threshold,
    )
    runs = [source.generate(index) for index in range(cfg.n_runs)]

    manifest = {
        "config_path": str(cfg.config_path),
        "seed": cfg.seed,
        "n_prefix_frames": n_prefix,
        "trigger_threshold": threshold,
        # Sampling knobs, so clip_source_from_dir can keep generating runs that
        # match the ones already in this set.
        "clip_config": {
            "config_path": str(cfg.config_path),
            "n_runs": cfg.n_runs,
            "seed": cfg.seed,
            "features": list(cfg.features),
            "bound_types": list(cfg.bound_types),
            "trigger_window": list(cfg.trigger_window),
            "correction_frames": list(cfg.correction_frames),
            "max_angle_delta": cfg.max_angle_delta,
            "min_goal_distance": cfg.min_goal_distance,
        },
        "runs": runs,
    }
    manifest_path = cfg.out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"{_LOG} wrote {manifest_path}", flush=True)
    return manifest_path
