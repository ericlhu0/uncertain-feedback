"""Turn hand-labeled correction clips into a HumanML3D-format finetune dataset.

Stage (b) of the correction-clip pipeline. Reads the clip sets written by
``generate.py`` and the labeling sessions forked off them — planner-space
``(K, 7)`` clips plus a manifest whose ``captions`` lists have been filled in by
hand — and encodes each captioned clip into HML263. A clip's captions become the
lines of its text file, which the humanml loader samples one of per epoch, so
several phrasings of one correction train as one motion rather than as copies.

``--clips_dir`` takes several directories, which is how separate labeling
sessions become one training set::

    uv run python \\
        src/uncertain_feedback/data_collection/dataset_auto_correction/build_dataset.py \\
        --clips_dir <clip set> <clip set>/session_* \\
        --output_dir .../motion-diffusion-model/dataset/correction_demo1

``--consistent_captions`` reconciles those lines with the clip before encoding:
a caption whose direction words the measured motion does not follow is dropped —
on the wrist's three axes, the elbow's three, and the elbow angle, all through
:mod:`motion_facts` — and down lines are duplicated until the up:down line ratio
matches the up:down motion ratio. Each transplanted copy is re-filtered against
its own measured motion, since replaying the joint deltas from another pose can
flip a world-frame axis. Off by default, so the original build is unchanged.

``--templated_captions`` goes further and does not read the labeled lines at all:
every motion gets the short single-axis imperatives
:mod:`templated_captions` writes from its own measured facts, which are true by
construction and land inside the ten words a bag-of-words text encoder can weigh.

``--retimes`` adds copies of each clip whose described window replays the run's
own motion at a different rate (:func:`retime_clip`), so the planner's single
pace stops being the only one the model sees. Every motion's pace — its wrist
speed over the set's median clip's, retimed or not — is written to
``paces.json`` for :mod:`tempo_captions`, which rewrites its text to say how
fast the change goes wherever that is noticeably off the usual pace.

Clips are encoded with :func:`smpl_arm_aa_seq_to_hml263_frames` — the *same*
function inference uses to build its pinned prefix — so training clips and
query-time prefixes share a body and an encoding by construction. That function
normalizes, so its output is un-normalized here to match the raw
``new_joint_vecs`` convention of :mod:`dataset_video.build_dataset`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spacy
from spacy.language import Language

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.data_collection.common.dataset import (
    copy_stats,
    write_splits,
    write_text_file,
)
from uncertain_feedback.data_collection.common.hml263 import load_hml_stats
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    assemble_clip,
    clip_source_from_dir,
    motion_frames,
)
from uncertain_feedback.data_collection.dataset_auto_correction.motion_facts import (
    MotionFacts,
    asserted,
    caption_conflicts,
    motion_facts,
)
from uncertain_feedback.data_collection.dataset_auto_correction.templated_captions import (
    templated_captions,
    templated_report,
)
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    smpl_arm_aa_seq_to_hml263_frames,
)
from uncertain_feedback.planners.mpc.arm_features import arm_feature_series
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import SmplLeftArmFK, q_to_arm_aa
from uncertain_feedback.simulated_users.personas import DEFAULT_ARM_JOINT_LIMITS

_DATASET_ROOT = MDM_ROOT / "motion-diffusion-model" / "dataset"
_DEFAULT_STATS_DIR = _DATASET_ROOT / "custom1_seatedcanon"

# A transplant may not inflate the bounded feature's excursion past this factor:
# the same joint deltas applied from a different base pose land differently,
# because the anatomical features are arccos/arcsin of rotated axes. Excursions
# under the floor are too small to contradict a caption either way (a bound's
# signature is the feature going *still*, so a clip's own excursion is ~0.01 rad).
_MAX_AMPLIFICATION = 3.0
_AMPLIFICATION_FLOOR = 0.05
_MAX_TRANSPLANT_ATTEMPTS = 20

# Retimed paces are log-uniform within this factor of the set's median clip
# speed. A fast copy can outrun the recorded motion, which then holds its last
# frame; the rate is capped so held frames stay a minority, since hold-dominated
# clips taught an earlier speed fine-tune to jump to the end pose and freeze.
_MAX_RETIME = 3.0
_MAX_HELD_FRACTION = 0.4
_NO_USUAL_BAND = (1.0, 1.0)


def retime_pace(
    rng: np.random.Generator,
    faster: bool,
    clear_of: tuple[float, float] = _NO_USUAL_BAND,
) -> float:
    """A target pace, log-uniform between the usual band and ``_MAX_RETIME``.

    ``clear_of = (low, high)`` is the band of paces counted as usual: the slower
    draw lies in ``[1 / _MAX_RETIME, low]`` and the faster one in ``[high,
    _MAX_RETIME]``, so every copy is noticeably off the usual pace and carries
    tempo text once captioned. The default band is empty, which is the original
    draw, log-uniform within ``_MAX_RETIME`` of usual on either side. One uniform
    draw either way, so the stream of an existing build is unchanged.
    """
    low, high = clear_of
    if faster:
        return float(np.exp(rng.uniform(np.log(high), np.log(_MAX_RETIME))))
    return float(np.exp(-rng.uniform(np.log(1.0 / low), np.log(_MAX_RETIME))))


def run_captions(run: dict[str, Any]) -> list[str]:
    """A run's captions, blanks dropped.

    Sets labeled before captions became a list carry a single ``caption`` string.
    """
    raw: list[str] = run.get("captions", [run.get("caption", "")])
    return [c.strip() for c in raw if c.strip()]


def consistent_captions(captions: list[str], facts: MotionFacts) -> list[str]:
    """The lines of *captions* whose direction words agree with the measured motion.

    A motion with no text is not trainable, so a run every line of which
    contradicts it keeps its least-contradictory line rather than dropping out.
    """
    kept = [c for c in captions if not caption_conflicts(c, facts)]
    if kept:
        return kept
    return [min(captions, key=lambda c: len(caption_conflicts(c, facts)))]


def rebalance_vertical(
    captions: dict[str, list[str]], vertical: dict[str, int]
) -> dict[str, int]:
    """Duplicate down lines until the up:down line ratio matches the motion ratio.

    The humanml loader draws one line per motion per epoch, so the line ratio is
    the text prior the model learns, and the captioner's is far more up-heavy than
    the clips are. Duplicating rather than capping keeps every phrasing: a
    repeated line is simply drawn more often.
    """
    up_motions = sum(1 for sign in vertical.values() if sign > 0)
    down_motions = sum(1 for sign in vertical.values() if sign < 0)
    down_by_run = {
        label: [line for line in lines if asserted(line).get("wrist_dy", 0) < 0]
        for label, lines in captions.items()
    }
    # One line from every run before any run's second, so the added weight spreads
    # over all the descending clips instead of piling onto the first few.
    down_lines = [
        (label, lines[i])
        for i in range(max(len(lines) for lines in down_by_run.values()))
        for label, lines in down_by_run.items()
        if i < len(lines)
    ]
    up_total = sum(
        1
        for lines in captions.values()
        for line in lines
        if asserted(line).get("wrist_dy", 0) > 0
    )
    deficit = round(up_total * down_motions / up_motions) - len(down_lines)
    added: dict[str, int] = {}
    for i in range(max(deficit, 0)):
        label, line = down_lines[i % len(down_lines)]
        captions[label].append(line)
        added[label] = added.get(label, 0) + 1
    return added


@dataclass(frozen=True)
class _Clip:
    """One captioned clip, with everything encoding and filtering it need."""

    label: str
    clip: np.ndarray
    naive: np.ndarray
    motion: np.ndarray
    anchor: int
    captions: list[str]
    feature: str
    base_pose: np.ndarray
    human: Human
    n_prefix: int


def clean_captions(clips: list[_Clip]) -> dict[str, list[str]]:
    """Reconcile every clip's caption lines with the motion it actually makes.

    Lines contradicting the wrist's, the elbow's or the elbow angle's measured
    travel are dropped, then down lines are duplicated until the up:down line
    ratio matches the up:down motion ratio. Reports what each run kept, dropped
    and gained.
    """
    kept: dict[str, list[str]] = {}
    vertical: dict[str, int] = {}
    fallbacks: list[str] = []
    for entry in clips:
        facts = motion_facts(entry.clip, entry.human, entry.n_prefix)
        vertical[entry.label] = facts.signs()["wrist_dy"]
        kept[entry.label] = consistent_captions(entry.captions, facts)
        if all(caption_conflicts(c, facts) for c in entry.captions):
            fallbacks.append(entry.label)
    added = rebalance_vertical(kept, vertical)

    print("\n=== caption consistency ===")
    for entry in clips:
        n_added = added.get(entry.label, 0)
        n_kept = len(kept[entry.label]) - n_added
        note = (
            " (all lines contradict — kept the least)"
            if entry.label in fallbacks
            else ""
        )
        print(
            f"{entry.label}: {len(entry.captions)} lines -> {n_kept} kept, "
            f"{len(entry.captions) - n_kept} dropped, {n_added} added{note}"
        )
    lines = [line for entry in clips for line in kept[entry.label]]
    up_lines = sum(1 for line in lines if asserted(line).get("wrist_dy", 0) > 0)
    down_lines = sum(1 for line in lines if asserted(line).get("wrist_dy", 0) < 0)
    up_motions = sum(1 for sign in vertical.values() if sign > 0)
    down_motions = sum(1 for sign in vertical.values() if sign < 0)
    print(
        f"total: {sum(len(e.captions) for e in clips)} lines -> {len(lines)} "
        f"({sum(added.values())} duplicated), {len(fallbacks)} runs kept only "
        f"their least-contradictory line\n"
        f"up:down = {up_lines}:{down_lines} lines over "
        f"{up_motions}:{down_motions} motions\n"
    )
    return kept


def transplant_clip(
    clip: np.ndarray, naive: np.ndarray, start: int, n_prefix: int
) -> np.ndarray:
    """Replay ``clip``'s correction from the naive rollout's ``start`` frame.

    The prefix becomes the arm's real history up to ``start`` — left-padded by
    repeating frame 0 the way ``planners/run.py`` pads an early trigger — and the
    described window is the clip's joint-space deltas applied from there, so the
    seam stays continuous and the whole clip sits at a different point of the
    reach. The correction's *shape* is preserved exactly; the absolute feature
    values it was generated under are not, which is why captions have to describe
    a behaviour rather than a limit.
    """
    prefix = naive[max(0, start - n_prefix + 1) : start + 1]
    if len(prefix) < n_prefix:
        pad = np.repeat(prefix[:1], n_prefix - len(prefix), axis=0)
        prefix = np.concatenate([pad, prefix])
    window = naive[start] + (clip[n_prefix:] - clip[n_prefix - 1])
    return np.concatenate([prefix, window])


def retime_clip(
    motion: np.ndarray, anchor: int, window: int, n_prefix: int, rate: float
) -> np.ndarray:
    """Replay the ``window`` frames after ``anchor`` at ``rate`` times their pace.

    The prefix is untouched — it is the arm's real history, at the pace it really
    had — and window frame ``i`` is the motion at time ``anchor + i * rate``,
    interpolated between frames and held at the motion's last frame once it runs
    out. The path is the run's own; only how fast it is travelled changes, so a
    slow copy covers less of it and a fast one more.
    """
    prefix, _ = assemble_clip(motion, anchor, 0, n_prefix)
    t = np.minimum(anchor + rate * np.arange(1, window + 1), len(motion) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(motion) - 1)
    frac = (t - lo)[:, None]
    return np.concatenate([prefix, motion[lo] * (1 - frac) + motion[hi] * frac])


def window_speed(clip: np.ndarray, human: Human, n_prefix: int) -> float:
    """Mean wrist travel per frame over the described window, in metres."""
    wrist = human.wrist_from_q(clip[n_prefix - 1 :])
    return float(np.linalg.norm(np.diff(wrist, axis=0), axis=1).mean())


def transplant_is_valid(
    clip: np.ndarray,
    moved: np.ndarray,
    feature: str,
    human: Human,
    n_prefix: int,
) -> bool:
    """Whether a transplanted clip is anatomically and behaviourally usable."""
    arm_aa = q_to_arm_aa(moved, human.fk.elbow_hinge_axis)
    if any(
        float(limit.violation(arm_aa).max()) > 0.0 for limit in DEFAULT_ARM_JOINT_LIMITS
    ):
        return False
    before = arm_feature_series(clip, human)[feature]
    after = arm_feature_series(moved, human)[feature]
    excursion = abs(float(before[-1] - before[n_prefix - 1]))
    moved_excursion = abs(float(after[-1] - after[n_prefix - 1]))
    return moved_excursion <= max(_AMPLIFICATION_FLOOR, _MAX_AMPLIFICATION * excursion)


def transplant_captions(
    captions: list[str], moved: np.ndarray, human: Human, n_prefix: int
) -> tuple[list[str], bool]:
    """*captions* re-filtered against a transplanted copy's own measured motion.

    A transplant replays joint deltas from a different base pose, so a
    world-frame direction the base clip made need not survive the move: the sign
    of a non-zero axis flips on 6-20% of copies (flexion, being a joint angle,
    never does). Filtering the base clip alone therefore leaves ~10% of a copy's
    surviving lines false of the copy. Returns the kept lines and whether every
    line contradicted, so the fallback's rate can be reported.
    """
    facts = motion_facts(moved, human, n_prefix)
    return consistent_captions(captions, facts), all(
        caption_conflicts(caption, facts) for caption in captions
    )


def build_correction_dataset(  # pylint: disable=too-many-arguments,too-many-locals
    clips_dirs: list[Path],
    output_dir: Path,
    hml_stats_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
    transplants: int = 0,
    consistent: bool = False,
    templated: bool = False,
    retimes: int = 0,
    retimes_only: bool = False,
    retime_clear_of: tuple[float, float] = _NO_USUAL_BAND,
) -> None:
    """Encode every captioned clip across *clips_dirs* into one MDM dataset.

    Each directory carries its own manifest and base pose, so a clip set and the
    labeling sessions forked off it combine into a single dataset by listing them
    all — ids are handed out across the whole run, never per directory.

    ``transplants`` adds that many augmented copies of each captioned clip,
    each replaying its correction from a *randomly drawn* frame of that run's own
    naive rollout (:func:`transplant_clip`), so a behaviour labeled once high up
    the reach also appears with the arm low — what the pinned prefix conditions on
    at inference. Draws that leave the anatomical box or distort the bounded
    feature are rejected and redrawn.

    ``consistent`` runs the captions through :func:`clean_captions` first, so a
    clip trains only on lines whose direction words its wrist actually follows,
    and every transplant through :func:`transplant_captions`, so a copy trains
    only on the lines still true of where it landed.

    ``templated`` discards the labeled captions entirely and writes each motion —
    every transplant included — the short single-axis lines
    :func:`~...templated_captions.templated_captions` generates from its own
    measured facts, still gated by :func:`consistent_captions` so a template that
    stops agreeing with the word table shows up as a dropped line. It supersedes
    ``consistent``, whose work the generated lines already do.

    ``retimes`` adds that many copies of each captioned clip replayed at a pace
    drawn log-uniformly within ``_MAX_RETIME`` of the set's median clip speed,
    alternating faster and slower (:func:`retime_clip`), captioned like a
    transplant. Clips move at different speeds to begin with, so the pace is
    measured against that median rather than the clip's own, which keeps one
    label meaning one speed. Every motion's pace, originals and transplants at
    their clip's own, goes to ``paces.json`` keyed by id; the lines still
    describe only the path until :mod:`tempo_captions` rewrites them.

    ``retimes_only`` writes the retimed copies and nothing else: the originals
    still set the pace reference and the gated lines the copies are re-filtered
    from, but neither they nor any transplant is encoded, so the set extends a
    model already trained on the originals. ``retime_clear_of`` is the usual
    band the paces are drawn clear of (:func:`retime_pace`).
    """
    hml_mean, hml_std = load_hml_stats(hml_stats_dir)
    fk = SmplLeftArmFK()
    nlp: Language = spacy.load("en_core_web_sm")

    (output_dir / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (output_dir / "texts").mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    rng = np.random.default_rng(seed)
    # Separate stream, so adding retimes leaves every transplant draw unchanged.
    retime_rng = np.random.default_rng([seed, 1])
    paces: dict[str, float] = {}

    def encode(
        clip: np.ndarray, captions: list[str], base_pose: np.ndarray, label: str
    ) -> None:
        """Write one clip and its captions out as the next dataset id."""
        arm_aa = q_to_arm_aa(clip, fk.elbow_hinge_axis)  # (K, 3, 3)
        norm = smpl_arm_aa_seq_to_hml263_frames(
            base_pose, arm_aa, hml_mean, hml_std, fk
        )  # (K, 263) normalized
        raw = (norm * (hml_std + 1e-8) + hml_mean).astype(np.float32)
        id_str = f"{len(ids) + 1:06d}"
        np.save(output_dir / "new_joint_vecs" / f"{id_str}.npy", raw)
        write_text_file(output_dir / "texts" / f"{id_str}.txt", captions, nlp)
        ids.append(id_str)
        print(f"{label} -> {id_str}: {raw.shape} {captions}")

    clips: list[_Clip] = []
    for clips_dir in clips_dirs:
        manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))
        human = clip_source_from_dir(clips_dir).human
        base_pose = human.hml_pose  # (263,)
        assert base_pose is not None, "clip sets need a planner config with a pose"
        for run in manifest["runs"]:
            captions = run_captions(run)
            if not captions and not templated:
                print(f"{clips_dir.name}/{run['run_id']}: no caption — skipping")
                continue
            clip = np.load(clips_dir / run["clip_file"])  # (K, 7)
            # The run's own naive rollout, since every run has its own scenario;
            # the manifest-level fallback keeps sets captioned under the old
            # single-scenario layout buildable.
            naive = np.load(
                clips_dir / run.get("naive_file", manifest.get("naive_file"))
            )
            # Sets saved before continuations were kept only have the clip, which
            # is itself a stretch of the motion ending on its last frame.
            motion, anchor = clip, manifest["n_prefix_frames"] - 1
            if "continuation_file" in run:
                motion, _ = motion_frames(
                    naive,
                    np.load(clips_dir / run["continuation_file"]),
                    run["trigger_step"],
                )
                anchor = run["clip_anchor"]
            clips.append(
                _Clip(
                    label=f"{clips_dir.name}/{run['run_id']}",
                    clip=clip,
                    naive=naive,
                    motion=motion,
                    anchor=anchor,
                    captions=captions,
                    feature=run["feature"],
                    base_pose=base_pose,
                    human=human,
                    n_prefix=manifest["n_prefix_frames"],
                )
            )

    templated_records: list[tuple[list[str], MotionFacts]] = []
    n_templated_dropped = 0

    def generated(clip: np.ndarray, entry: _Clip) -> list[str]:
        """*clip*'s own templated lines, gated and recorded for the report."""
        nonlocal n_templated_dropped
        facts = motion_facts(clip, entry.human, entry.n_prefix)
        lines = templated_captions(facts)
        kept = consistent_captions(lines, facts)
        n_templated_dropped += len(lines) - len(kept)
        templated_records.append((kept, facts))
        return kept

    cleaned = clean_captions(clips) if consistent and not templated else {}
    n_transplants = 0
    n_transplant_dropped = 0
    n_transplant_fallbacks = 0

    def copy_captions(
        moved: np.ndarray, entry: _Clip, captions: list[str]
    ) -> list[str]:
        """The lines an augmented copy of *entry* trains on."""
        nonlocal n_transplants, n_transplant_dropped, n_transplant_fallbacks
        if templated:
            return generated(moved, entry)
        if not consistent:
            return captions
        kept, fell_back = transplant_captions(
            captions, moved, entry.human, entry.n_prefix
        )
        n_transplants += 1
        n_transplant_dropped += len(captions) - len(kept)
        n_transplant_fallbacks += int(fell_back)
        return kept

    speeds = {e.label: window_speed(e.clip, e.human, e.n_prefix) for e in clips}
    usual = float(np.median(list(speeds.values())))
    for entry in clips:
        captions = (
            generated(entry.clip, entry)
            if templated
            else cleaned.get(entry.label, entry.captions)
        )
        if not retimes_only:
            encode(entry.clip, captions, entry.base_pose, entry.label)
            if retimes:
                paces[ids[-1]] = speeds[entry.label] / usual

        for _ in range(0 if retimes_only else transplants):
            for _attempt in range(_MAX_TRANSPLANT_ATTEMPTS):
                start = int(rng.integers(0, len(entry.naive)))
                moved = transplant_clip(entry.clip, entry.naive, start, entry.n_prefix)
                if transplant_is_valid(
                    entry.clip, moved, entry.feature, entry.human, entry.n_prefix
                ):
                    encode(
                        moved,
                        copy_captions(moved, entry, captions),
                        entry.base_pose,
                        f"{entry.label}@{start}",
                    )
                    if retimes:
                        paces[ids[-1]] = speeds[entry.label] / usual
                    break
            else:
                print(
                    f"{entry.label}: no valid transplant in "
                    f"{_MAX_TRANSPLANT_ATTEMPTS} draws — skipping one copy"
                )

        window = len(entry.clip) - entry.n_prefix
        room = len(entry.motion) - 1 - entry.anchor
        for j in range(retimes):
            pace = retime_pace(retime_rng, faster=j % 2 == 0, clear_of=retime_clear_of)
            rate = min(
                pace * usual / speeds[entry.label],
                room / ((1 - _MAX_HELD_FRACTION) * window),
            )
            moved = retime_clip(
                entry.motion, entry.anchor, window, entry.n_prefix, rate
            )
            pace = rate * speeds[entry.label] / usual
            encode(
                moved,
                copy_captions(moved, entry, captions),
                entry.base_pose,
                f"{entry.label}x{pace:.2f}",
            )
            paces[ids[-1]] = pace

    if templated_records:
        print(templated_report(templated_records))
        print(f"{n_templated_dropped} generated lines dropped by the fact check\n")

    if n_transplants:
        print(
            f"\n=== transplant consistency ===\n"
            f"{n_transplants} augmented copies re-filtered against their own motion: "
            f"{n_transplant_dropped} lines dropped, {n_transplant_fallbacks} "
            f"({n_transplant_fallbacks / n_transplants:.1%}) kept only their "
            f"least-contradictory line\n"
        )

    if not ids:
        raise RuntimeError(
            "No captioned runs in " + ", ".join(str(d) for d in clips_dirs) + "."
        )

    write_splits(output_dir, ids, val_fraction, test_fraction, seed)
    if paces:
        (output_dir / "paces.json").write_text(
            json.dumps(paces, indent=1), encoding="utf-8"
        )

    copy_stats(hml_stats_dir, output_dir)

    print(f"\nDataset written to {output_dir}")


def main() -> None:
    """Parse arguments and build the correction dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips_dir",
        required=True,
        nargs="+",
        help=(
            "One or more clip sets or labeling sessions to encode together, e.g. "
            "<clip set> <clip set>/session_*."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Directory to write the dataset into; put it under "
            f"{_DATASET_ROOT} so finetune_standing.sh can swap it in."
        ),
    )
    parser.add_argument(
        "--hml_stats_dir",
        default=str(_DEFAULT_STATS_DIR),
        help=(
            "Directory containing HumanML3D Mean.npy and Std.npy. Defaults to "
            "custom1_seatedcanon, not dataset/HumanML3D — that path is the "
            "fine-tune swap slot and holds whichever dataset trained last."
        ),
    )
    parser.add_argument(
        "--transplants",
        type=int,
        default=0,
        help=(
            "Augmented copies per captioned clip, each replaying its correction "
            "from a randomly drawn frame of the naive rollout so the behaviour "
            "appears at other points of the reach (default: 0, no augmentation)."
        ),
    )
    parser.add_argument(
        "--consistent_captions",
        action="store_true",
        help=(
            "Drop caption lines whose direction words contradict the clip's own "
            "measured motion — the wrist's and the elbow's three axes and the "
            "elbow angle — then duplicate down lines until the up:down line "
            "ratio matches the up:down motion ratio (default: off, every "
            "caption is trained on as written)."
        ),
    )
    parser.add_argument(
        "--templated_captions",
        action="store_true",
        help=(
            "Ignore the labeled captions and write every motion — transplants "
            "included — short single-axis imperatives generated from its own "
            "measured facts, at most 10 words each (default: off, the labeled "
            "captions are used). Supersedes --consistent_captions."
        ),
    )
    parser.add_argument(
        "--retimes",
        type=int,
        default=0,
        help=(
            "Copies per captioned clip whose window replays the run's motion at "
            f"a pace within {_MAX_RETIME:g}x of the set's median clip speed, "
            "alternating faster and slower; every motion's pace goes to "
            "paces.json for tempo_captions.py (default: 0, every clip at the "
            "planner's pace)."
        ),
    )
    parser.add_argument(
        "--retimes_only",
        action="store_true",
        help=(
            "With --retimes: write only the retimed copies (no originals, no "
            "transplants); the originals still set the pace reference and the "
            "gated lines, so the set extends a model trained on them."
        ),
    )
    parser.add_argument(
        "--retime_clear_of",
        type=float,
        nargs=2,
        default=_NO_USUAL_BAND,
        metavar=("LOW", "HIGH"),
        help=(
            "Draw retime paces clear of this usual band: the slower copy in "
            f"[1/{_MAX_RETIME:g}, LOW], the faster in [HIGH, {_MAX_RETIME:g}] "
            "(default 1 1: no band, as the first retimed sets were built)."
        ),
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.retimes_only and not args.retimes:
        parser.error("--retimes_only needs --retimes")

    build_correction_dataset(
        clips_dirs=[Path(d) for d in args.clips_dir],
        output_dir=Path(args.output_dir),
        hml_stats_dir=Path(args.hml_stats_dir),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        transplants=args.transplants,
        consistent=args.consistent_captions,
        templated=args.templated_captions,
        retimes=args.retimes,
        retimes_only=args.retimes_only,
        retime_clear_of=(args.retime_clear_of[0], args.retime_clear_of[1]),
    )


if __name__ == "__main__":
    main()
