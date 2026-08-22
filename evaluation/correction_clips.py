"""Correction clips: many oracle-corrected branches off one naive MPC rollout.

Stage (a) of the correction-clip finetune pipeline. One naive rollout to a fixed
Cartesian goal is the base trajectory every clip branches from. Each run samples
a hidden comfort bound *anchored on that rollout* (so the naive path is
guaranteed to violate it), replans from the induced trigger step under the
oracle cost, and saves the whole rollout.

A run's naive approach and oracle rollout join into one continuous motion
(:func:`motion_frames`), and a clip is a *cut* out of it: ``n_prefix`` frames of
history up to an anchor, then a window of what follows
(:func:`assemble_clip`). The sampled default anchors at the trigger, but the
labeling UI can drag the cut anywhere along the motion — later stretches of the
same rollout are often the interesting ones — and :meth:`ClipSource.cut` rewrites
the clip with no replan.

The prefix is the same arm history inference pins as conditioning
(:mod:`uncertain_feedback.planners.run`), so a checkpoint fine-tuned on these
clips sees the distribution it is queried with. That holds at any anchor: pinning
recent *corrected* history is still the arm's real recent history.

Only trajectories are written, never rendered video — video dominated the output
size (3.5 MB of a 3.9 MB 32-clip set). ``label_correction_clips.py`` previews a run
in the browser from ``naive.npy`` plus the run's ``continuation.npy``, and
``geometry.npz`` carries the generator-decoded body so that preview needs neither
the MDM environment nor a GPU.

Stage (b) — :mod:`uncertain_feedback.data_collection.build_correction_dataset` —
turns the hand-labeled manifest into a HumanML3D-format finetune dataset.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.rig import base_extra_costs, build_rig, cfg_with_goal
from uncertain_feedback.planners.mpc.arm_features import (
    FEATURE_NAMES,
    arm_feature_series,
)
from uncertain_feedback.planners.mpc.config import MpcRunConfig, load_mpc_config
from uncertain_feedback.planners.mpc.costs import (
    CompositeTrajectoryCost,
    MpcCostContext,
    build_extra_costs,
)
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa
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

# Clip = n_prefix + window frames. The floor keeps it past the t2m loader's
# min-40 filter, the ceiling under MDM's 196-frame cap.
MIN_WINDOW = 42
MAX_WINDOW = 180


@dataclass(frozen=True)
class CorrectionClipConfig:
    """What to generate: one base trajectory, ``n_runs`` corrected branches.

    ``max_angle_delta`` overrides the planner config's action-sampling spread and
    is the one knob controlling how big and how fast a clip's motion is. It is a
    std dev, not a per-step cap, so it sets distance travelled per frame; since a
    clip is a fixed frame budget, halving it halves both the speed and the ground
    covered. On the low1 start: 0.0025 gives a 4.2 s reach and 0.351 m of wrist
    path per clip at 0.0079 m/frame, 0.00125 (the default) gives 8.2 s and
    0.186 m at 0.0042 m/frame. Clip length and padding are unaffected.

    ``margin_range`` is *not* a size knob, despite looking like one — it sets how
    far past the naive value the bound sits, i.e. which way and how insistently
    the correction deviates, not how far the arm travels in the window. Halving
    it left wrist path within a centimetre while making corrections less distinct
    from the naive path, so it is left wide.

    ``trigger_window`` counts naive frames, so it has to scale with
    ``max_angle_delta``: (12, 100) suits the 165-frame reach at 0.00125, (6, 50)
    the 85-frame reach at 0.0025.

    Clips are paced more slowly than the 0.01 demo the finetuned checkpoint is
    queried in — MDM output is tracked as a path (playback advances on
    proximity), so the pacing costs nothing downstream, but the pinned prefix a
    clip carries is slower than the one inference pins.
    """

    config_path: Path
    out_dir: Path
    n_runs: int = 32
    seed: int = 0
    features: tuple[str, ...] = FEATURE_NAMES
    bound_types: tuple[str, ...] = ("upper_bound", "lower_bound")
    trigger_window: tuple[int, int] = (12, 100)
    margin_range: tuple[float, float] = (0.05, 0.20)
    correction_frames: tuple[int, int] = (42, 56)
    max_angle_delta: float = 0.00125


@dataclass(frozen=True)
class SampledBound:
    """One hidden bound plus the trigger step it induces on the naive rollout."""

    user: SimulatedUser
    feature: str
    bound_type: str
    value: float
    margin: float
    anchor_step: int
    trigger_step: int


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
    context: MpcCostContext,
    cfg: CorrectionClipConfig,
    threshold: float,
) -> SampledBound:
    """Sample a hidden bound the naive rollout is guaranteed to violate.

    The bound sits ``margin`` radians on the wrong side of the naive
    trajectory's own feature value at a random anchor step, so that step always
    violates it and :func:`first_violation_step` on the naive rollout *is* the
    trigger the bound induces — at or before the anchor, since the features are
    not monotonic. Samples whose induced trigger falls outside
    ``cfg.trigger_window`` are rejected, which also discards bounds pointing the
    way the naive path already moves.
    """
    feats = arm_feature_series(naive_q, context)
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
        anchor = int(rng.integers(low, high + 1))
        margin = float(rng.uniform(*cfg.margin_range))
        offset = -margin if bound_type == "upper_bound" else margin
        value = float(feats[feature][anchor]) + offset
        user = synthetic_user(feature, bound_type, value)
        trigger = first_violation_step(user, context, naive_q, threshold)
        if trigger is not None and low <= trigger <= high:
            return SampledBound(
                user=user,
                feature=feature,
                bound_type=bound_type,
                value=value,
                margin=margin,
                anchor_step=anchor,
                trigger_step=trigger,
            )
    raise RuntimeError(
        f"No sampled bound triggered inside {cfg.trigger_window} within "
        f"{_MAX_SAMPLE_ATTEMPTS} attempts — widen trigger_window or margin_range."
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


def write_feature_csv(path: Path, q: np.ndarray, context: MpcCostContext) -> None:
    """Write the per-frame anatomical features of an arm trajectory."""
    feats = arm_feature_series(q, context)
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
    return anchor, int(np.clip(window, MIN_WINDOW, max(MIN_WINDOW, min(MAX_WINDOW, room))))


def arm_positions(
    q: np.ndarray, fk: SmplLeftArmFK, spine3_pos: np.ndarray, spine3_aa: np.ndarray
) -> np.ndarray:
    """``(T, 5, 3)`` world positions of the arm chain over a q trajectory."""
    return fk.fk_batch(q_to_arm_aa(q, fk.elbow_hinge_axis), spine3_pos, spine3_aa)


@dataclass(frozen=True)
class ClipSource:
    """Everything needed to branch one more corrected run off a fixed naive rollout.

    Holds only the immutable per-set context, so runs can be produced one at a
    time and out of order — the labeling UI generates the next one while the
    previous is being captioned. :func:`clip_source_from_dir` rebuilds this from
    the artifacts on disk without touching MDM, so on-demand generation needs
    neither the generator nor a GPU.
    """

    out_dir: Path
    cfg: CorrectionClipConfig
    goal: np.ndarray
    goal_cfg: MpcRunConfig
    context: MpcCostContext
    base: CompositeTrajectoryCost
    naive: np.ndarray
    body_pos: np.ndarray
    n_prefix: int
    threshold: float

    def generate(self, index: int) -> dict[str, Any]:
        """Produce run ``index``: write its clip files, return its manifest row.

        Seeded on ``(seed, index)`` rather than a streaming generator, so a run is
        reproducible from its index alone whether it came from a batch or from a
        click on Next.
        """
        rng = np.random.default_rng([self.cfg.seed, index])
        sampled = sample_violating_bound(
            rng, self.naive, self.context, self.cfg, self.threshold
        )
        window = int(
            rng.integers(
                self.cfg.correction_frames[0], self.cfg.correction_frames[1] + 1
            )
        )
        oracle_costs = CompositeTrajectoryCost(
            [
                *self.base.terms(),
                HiddenCostTerm(user=sampled.user, context=self.context),
            ]
        )
        continuation = rollout_to_goal(
            self.goal_cfg,
            self.naive[sampled.trigger_step],
            self.goal,
            self.context,
            oracle_costs,
            self.body_pos,
            self.context.spine3_pos,
            self.context.spine3_aa,
            progress_label=f"run {index} continuation",
            log_prefix=_LOG,
        )
        run_id = f"run_{index:03d}"
        run_dir = self.out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # The whole continuation, not just the window a clip keeps: the labeling
        # UI shows where the correction was still heading, and re-cuts against it.
        np.save(run_dir / "continuation.npy", continuation)
        row = {
            "run_id": run_id,
            "caption": "",
            "feature": sampled.feature,
            "bound_type": sampled.bound_type,
            "bound_value": sampled.value,
            "margin": sampled.margin,
            "anchor_step": sampled.anchor_step,
            "trigger_step": sampled.trigger_step,
            "continuation_frames": int(len(continuation)),
            "continuation_reach": goal_reach(
                self.context, self.goal_cfg, continuation, self.goal
            ),
            "clip_file": f"{run_id}/clip.npy",
            "continuation_file": f"{run_id}/continuation.npy",
            "features_file": f"{run_id}/clip_features.csv",
        }
        motion, _ = motion_frames(self.naive, continuation, sampled.trigger_step)
        row.update(
            self.cut(row, motion, anchor=sampled.trigger_step, window=window)
        )
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
        write_feature_csv(run_dir / "clip_features.csv", clip, self.context)
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
                user, self.context, clip[self.n_prefix :]
            ),
        }


def new_session_dir(base_dir: Path) -> Path:
    """Fork a fresh labeling session off the clip set in ``base_dir``.

    Every labeling session gets its own directory, runs and manifest, so starting
    one can never overwrite an earlier session's captions. The base artifacts are
    copied rather than referenced, leaving each session self-contained: stage (b)
    reads a session exactly like it reads a clip set.

    The seed is the session's own timestamp. Runs are seeded on ``(seed, index)``,
    so two sessions off the same base would otherwise sample the same bounds and
    replan the same corrections from run 0 onward.
    """
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{stamp}"
    session_dir.mkdir(parents=True)
    for key in ("naive_file", "base_pose_file", "geometry_file"):
        shutil.copy(base_dir / manifest[key], session_dir / manifest[key])
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

    The base artifacts stage (a) writes — ``naive.npy`` for the trajectory every
    run branches from and ``geometry.npz`` for the generator-decoded body — are
    exactly what a rollout needs, so nothing here loads the motion generator.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    geo = np.load(out_dir / manifest["geometry_file"])
    fk = SmplLeftArmFK()
    fk.collar_aa = geo["collar_aa"]
    stored = manifest["clip_config"]
    cfg = CorrectionClipConfig(
        config_path=Path(stored["config_path"]),
        out_dir=out_dir,
        n_runs=stored["n_runs"],
        seed=stored["seed"],
        features=tuple(stored["features"]),
        bound_types=tuple(stored["bound_types"]),
        trigger_window=(stored["trigger_window"][0], stored["trigger_window"][1]),
        margin_range=(stored["margin_range"][0], stored["margin_range"][1]),
        correction_frames=(
            stored["correction_frames"][0],
            stored["correction_frames"][1],
        ),
        max_angle_delta=stored["max_angle_delta"],
    )
    run_cfg = load_mpc_config(cfg.config_path)
    context = MpcCostContext(
        fk=fk,
        spine3_pos=geo["spine3_pos"],
        spine3_aa=geo["spine3_aa"],
        time_of_day=run_cfg.simulated_user.time_of_day,
    )
    goal = np.asarray(manifest["goal"], dtype=np.float64)
    return ClipSource(
        out_dir=out_dir,
        cfg=cfg,
        goal=goal,
        goal_cfg=cfg_with_goal(
            replace(run_cfg, max_angle_delta=cfg.max_angle_delta, seed=cfg.seed), goal
        ),
        context=context,
        base=CompositeTrajectoryCost(
            [
                *build_extra_costs(run_cfg.costs, context).terms(),
                UNRESTRICTED.limit_cost(),
            ]
        ),
        naive=np.load(out_dir / manifest["naive_file"]),
        body_pos=geo["body_pos"],
        n_prefix=manifest["n_prefix_frames"],
        threshold=manifest["trigger_threshold"],
    )


def generate_correction_clips(cfg: CorrectionClipConfig) -> Path:
    """Write the base artifacts plus ``cfg.n_runs`` corrected branches.

    ``n_runs=0`` writes only the base artifacts — the naive rollout, body
    geometry and an empty manifest — which is all the labeling UI needs to
    generate runs on demand.

    Refuses to write into a directory that already holds a clip set: the manifest
    is rewritten wholesale, so doing so would blank its captions and orphan the
    labeling sessions underneath it.
    """
    if (cfg.out_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{cfg.out_dir} already holds a clip set. Generating into it would "
            "blank its captions and its labeling sessions — pass a new --out_dir."
        )
    rig = build_rig(cfg.config_path, seed=cfg.seed, load_generator=True)
    assert rig.gen is not None
    assert rig.initial_hml_pose is not None
    assert rig.body_pos is not None
    assert rig.cfg.cartesian is not None
    n_prefix = rig.gen.prefix_frames
    threshold = rig.cfg.corrections.trigger_threshold
    goal = np.asarray(rig.cfg.cartesian.goals[0], dtype=np.float64)
    goal_cfg = cfg_with_goal(
        replace(rig.cfg, max_angle_delta=cfg.max_angle_delta), goal
    )
    # UNRESTRICTED is `bounds=()`: it carries the anatomical joint box into the
    # cost stack and nothing else. No persona's comfort bounds enter this
    # pipeline — every bound is sampled per run, anchored on the naive rollout
    # this base cost produces.
    base = base_extra_costs(rig, UNRESTRICTED)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.out_dir / "base_pose.npy", rig.initial_hml_pose)
    # The body geometry the generator decoded, so previewing a clip set needs
    # neither the MDM environment nor a GPU.
    np.savez(
        cfg.out_dir / "geometry.npz",
        body_pos=rig.body_pos,
        spine3_pos=rig.spine3_pos,
        spine3_aa=rig.spine3_aa,
        collar_aa=rig.fk.collar_aa,
    )

    naive = rollout_to_goal(
        goal_cfg,
        rig.q0,
        goal,
        rig.context,
        base,
        rig.body_pos,
        rig.spine3_pos,
        rig.spine3_aa,
        progress_label="naive",
        log_prefix=_LOG,
    )
    np.save(cfg.out_dir / "naive.npy", naive)
    write_feature_csv(cfg.out_dir / "naive_features.csv", naive, rig.context)
    print(f"{_LOG} naive rollout: {len(naive)} frames", flush=True)

    source = ClipSource(
        out_dir=cfg.out_dir,
        cfg=cfg,
        goal=goal,
        goal_cfg=goal_cfg,
        context=rig.context,
        base=base,
        naive=naive,
        body_pos=rig.body_pos,
        n_prefix=n_prefix,
        threshold=threshold,
    )
    runs = [source.generate(index) for index in range(cfg.n_runs)]

    manifest = {
        "config_path": str(cfg.config_path),
        "seed": cfg.seed,
        "goal": goal.tolist(),
        "n_prefix_frames": n_prefix,
        "trigger_threshold": threshold,
        "naive_file": "naive.npy",
        "base_pose_file": "base_pose.npy",
        "geometry_file": "geometry.npz",
        # Sampling knobs, so clip_source_from_dir can keep generating runs that
        # match the ones already in this set.
        "clip_config": {
            "config_path": str(cfg.config_path),
            "n_runs": cfg.n_runs,
            "seed": cfg.seed,
            "features": list(cfg.features),
            "bound_types": list(cfg.bound_types),
            "trigger_window": list(cfg.trigger_window),
            "margin_range": list(cfg.margin_range),
            "correction_frames": list(cfg.correction_frames),
            "max_angle_delta": cfg.max_angle_delta,
        },
        "runs": runs,
    }
    manifest_path = cfg.out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"{_LOG} wrote {manifest_path}", flush=True)
    return manifest_path
