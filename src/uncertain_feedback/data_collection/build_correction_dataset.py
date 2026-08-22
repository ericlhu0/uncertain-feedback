"""Turn hand-labeled correction clips into a HumanML3D-format finetune dataset.

Stage (b) of the correction-clip pipeline. Reads the clip sets written by
``evaluation/generate_correction_clips.py`` and the labeling sessions forked off
them — planner-space ``(K, 7)`` clips plus a manifest whose ``caption`` fields
have been filled in by hand — and encodes each captioned clip into HML263.

``--clips_dir`` takes several directories, which is how separate labeling
sessions become one training set::

    uv run python src/uncertain_feedback/data_collection/build_correction_dataset.py \\
        --clips_dir outputs/correction_clips outputs/correction_clips/session_* \\
        --output_dir .../motion-diffusion-model/dataset/correction_demo1

Clips are encoded with :func:`smpl_arm_aa_seq_to_hml263_frames` — the *same*
function inference uses to build its pinned prefix — so training clips and
query-time prefixes share a body and an encoding by construction. That function
normalizes, so its output is un-normalized here to match the raw
``new_joint_vecs`` convention of :mod:`build_mdm_dataset`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spacy
from spacy.language import Language

from uncertain_feedback.consts import MDM_ROOT
from uncertain_feedback.data_collection.build_mdm_dataset import (
    _write_text_file,
    write_splits,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import load_hml_stats
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    smpl_arm_aa_seq_to_hml263_frames,
)
from uncertain_feedback.planners.mpc.arm_features import arm_feature_series
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


@dataclass(frozen=True)
class _GeometryContext:
    """The FK state :func:`arm_feature_series` needs, read off ``geometry.npz``."""

    fk: SmplLeftArmFK
    spine3_pos: np.ndarray
    spine3_aa: np.ndarray


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


def transplant_is_valid(
    clip: np.ndarray,
    moved: np.ndarray,
    feature: str,
    context: _GeometryContext,
    n_prefix: int,
) -> bool:
    """Whether a transplanted clip is anatomically and behaviourally usable."""
    arm_aa = q_to_arm_aa(moved, context.fk.elbow_hinge_axis)
    if any(
        float(limit.violation(arm_aa).max()) > 0.0
        for limit in DEFAULT_ARM_JOINT_LIMITS
    ):
        return False
    before = arm_feature_series(clip, context)[feature]
    after = arm_feature_series(moved, context)[feature]
    excursion = abs(float(before[-1] - before[n_prefix - 1]))
    moved_excursion = abs(float(after[-1] - after[n_prefix - 1]))
    return moved_excursion <= max(
        _AMPLIFICATION_FLOOR, _MAX_AMPLIFICATION * excursion
    )


def build_correction_dataset(  # pylint: disable=too-many-arguments,too-many-locals
    clips_dirs: list[Path],
    output_dir: Path,
    hml_stats_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
    transplants: int = 0,
) -> None:
    """Encode every captioned clip across *clips_dirs* into one MDM dataset.

    Each directory carries its own manifest and base pose, so a clip set and the
    labeling sessions forked off it combine into a single dataset by listing them
    all — ids are handed out across the whole run, never per directory.

    ``transplants`` adds that many augmented copies of each captioned clip,
    each replaying its correction from a *randomly drawn* frame of the naive
    rollout (:func:`transplant_clip`), so a behaviour labeled once high up the
    reach also appears with the arm low — what the pinned prefix conditions on at
    inference. Draws that leave the anatomical box or distort the bounded feature
    are rejected and redrawn.
    """
    hml_mean, hml_std = load_hml_stats(hml_stats_dir)
    fk = SmplLeftArmFK()
    nlp: Language = spacy.load("en_core_web_sm")

    (output_dir / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (output_dir / "texts").mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    rng = np.random.default_rng(seed)

    def encode(clip: np.ndarray, caption: str, base_pose: np.ndarray, label: str) -> None:
        """Write one clip and its caption out as the next dataset id."""
        arm_aa = q_to_arm_aa(clip, fk.elbow_hinge_axis)  # (K, 3, 3)
        norm = smpl_arm_aa_seq_to_hml263_frames(
            base_pose, arm_aa, hml_mean, hml_std, fk
        )  # (K, 263) normalized
        raw = (norm * (hml_std + 1e-8) + hml_mean).astype(np.float32)
        id_str = f"{len(ids) + 1:06d}"
        np.save(output_dir / "new_joint_vecs" / f"{id_str}.npy", raw)
        _write_text_file(output_dir / "texts" / f"{id_str}.txt", [caption], nlp)
        ids.append(id_str)
        print(f"{label} -> {id_str}: {raw.shape} {caption!r}")

    for clips_dir in clips_dirs:
        manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))
        base_pose = np.load(clips_dir / manifest["base_pose_file"])  # (263,)
        naive = np.load(clips_dir / manifest["naive_file"])
        geo = np.load(clips_dir / manifest["geometry_file"])
        geo_fk = SmplLeftArmFK()
        geo_fk.collar_aa = geo["collar_aa"]
        context = _GeometryContext(
            fk=geo_fk, spine3_pos=geo["spine3_pos"], spine3_aa=geo["spine3_aa"]
        )
        n_prefix = manifest["n_prefix_frames"]
        for run in manifest["runs"]:
            caption = run["caption"].strip()
            if not caption:
                print(f"{clips_dir.name}/{run['run_id']}: no caption — skipping")
                continue
            clip = np.load(clips_dir / run["clip_file"])  # (K, 7)
            encode(clip, caption, base_pose, f"{clips_dir.name}/{run['run_id']}")

            for _ in range(transplants):
                for _attempt in range(_MAX_TRANSPLANT_ATTEMPTS):
                    start = int(rng.integers(0, len(naive)))
                    moved = transplant_clip(clip, naive, start, n_prefix)
                    if transplant_is_valid(
                        clip, moved, run["feature"], context, n_prefix
                    ):
                        encode(
                            moved,
                            caption,
                            base_pose,
                            f"{clips_dir.name}/{run['run_id']}@{start}",
                        )
                        break
                else:
                    print(
                        f"{clips_dir.name}/{run['run_id']}: no valid transplant in "
                        f"{_MAX_TRANSPLANT_ATTEMPTS} draws — skipping one copy"
                    )

    if not ids:
        raise RuntimeError(
            "No captioned runs in " + ", ".join(str(d) for d in clips_dirs) + "."
        )

    write_splits(output_dir, ids, val_fraction, test_fraction, seed)

    for stat_file in ("Mean.npy", "Std.npy"):
        shutil.copy(hml_stats_dir / stat_file, output_dir / stat_file)

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
            "outputs/correction_clips outputs/correction_clips/session_*."
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
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_correction_dataset(
        clips_dirs=[Path(d) for d in args.clips_dir],
        output_dir=Path(args.output_dir),
        hml_stats_dir=Path(args.hml_stats_dir),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        transplants=args.transplants,
    )


if __name__ == "__main__":
    main()
