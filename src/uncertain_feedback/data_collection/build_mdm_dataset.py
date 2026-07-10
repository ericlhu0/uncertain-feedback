"""Convert labeled motion frame segments into an MDM-compatible HumanML3D dataset.

Each labeled segment (frame range + caption) from the labeler becomes one independent
training trajectory in the dataset.

Usage::

    uv run python src/uncertain_feedback/data_collection/build_mdm_dataset.py \\
        --frames_dir ./frames/ \\
        --labels_json ./frames/labels.json \\
        --output_dir ./my_mdm_dataset/ \\
        [--val_fraction 0.1] \\
        [--test_fraction 0.1]

Run ``extract_all_frames.py`` and the labeler first to produce *frames_dir* and
*labels_json*.

Output structure mirrors HumanML3D so it can be consumed directly by the MDM
data loader (``--dataset humanml``)::

    output_dir/
    ├── new_joint_vecs/000001.npy   # (N, 263) float32 raw (unnormalized) HML263
    ├── texts/000001.txt            # caption#w/POS w/POS ...#0.0#0.0
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── Mean.npy                    # copied from hml_stats_dir
    └── Std.npy                     # copied from hml_stats_dir
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import spacy
from spacy.language import Language

from uncertain_feedback.data_collection.mhr_pose_estimator import MhrEstimatorConfig
from uncertain_feedback.data_collection.mhr_to_hml263_pipeline import (
    MhrToHml263Config,
    MhrToHml263Pipeline,
    resample_positions,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)

# Bump whenever the pose-estimation / camera-to-world conversion changes so
# stale cache entries (e.g. pre-chirality-fix mirrored data) are never reused.
_CACHE_VERSION = 2

# ---------------------------------------------------------------------------
# Text annotation helpers
# ---------------------------------------------------------------------------


def _pos_tag(nlp: Language, caption: str) -> str:
    """Return space-separated ``word/POS`` tokens for *caption* using spaCy."""
    doc = nlp(caption)
    return " ".join(f"{token.text}/{token.pos_}" for token in doc)


def _write_text_file(path: Path, captions: list[str], nlp: Language) -> None:
    """Write one MDM-format text annotation file.

    Each line has the format ``caption#tokens#0.0#0.0`` where ``tokens`` is a
    space-separated sequence of ``word/POSTAG`` pairs and both timestamps are
    ``0.0`` to indicate the label covers the full motion clip.
    """
    lines = [f"{cap}#{_pos_tag(nlp, cap)}#0.0#0.0" for cap in captions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Motion resampling helper
# ---------------------------------------------------------------------------

_MDM_MIN_FRAMES = 50
_MDM_MAX_FRAMES = 190


def _clamp_positions_length(positions: np.ndarray) -> tuple[np.ndarray, bool]:
    """Resample positions so the resulting feature length is in MDM's range.

    ``positions_to_hml263`` yields ``N-1`` feature frames for ``N`` position
    frames, so positions are clamped to ``[_MDM_MIN_FRAMES + 1,
    _MDM_MAX_FRAMES + 1]``.  Resampling positions (rather than features) keeps
    velocity and foot-contact features consistent with the final frame timing.

    Args:
        positions: ``(N, 22, 3)`` joint positions.

    Returns:
        Tuple of the (possibly resampled) array and a boolean that is ``True``
        when resampling was applied.
    """
    n = len(positions)
    if _MDM_MIN_FRAMES + 1 <= n <= _MDM_MAX_FRAMES + 1:
        return positions, False
    target = _MDM_MIN_FRAMES + 1 if n < _MDM_MIN_FRAMES + 1 else _MDM_MAX_FRAMES + 1
    return resample_positions(positions, target), True


# ---------------------------------------------------------------------------
# Body-locking helpers
# ---------------------------------------------------------------------------

# SMPL-22 left arm joint indices (0-indexed): left_shoulder=16, left_elbow=18, left_wrist=20
_L_ARM_JOINTS = {16, 18, 20}


def _arm_feature_mask() -> np.ndarray:
    """Return a (263,) bool mask that is True for left-arm HML263 features."""
    mask = np.zeros(263, dtype=bool)
    for j in _L_ARM_JOINTS:
        mask[4 + (j - 1) * 3 : 4 + (j - 1) * 3 + 3] = True   # positions
        mask[67 + (j - 1) * 6 : 67 + (j - 1) * 6 + 6] = True  # rotations
        mask[193 + j * 3 : 193 + j * 3 + 3] = True             # velocities
    return mask


def _lock_body_to_frame0(hml263: np.ndarray) -> np.ndarray:
    """Copy frame-0 values of all non-arm features to every frame.

    Makes the training data fully consistent with fixed-body inference: the
    model only ever sees a static body with varying left-arm motion.
    """
    arm_mask = _arm_feature_mask()
    result = hml263.copy()
    result[:, ~arm_mask] = result[0:1, ~arm_mask]
    return result


# ---------------------------------------------------------------------------
# Frame-copy helper
# ---------------------------------------------------------------------------


def _copy_frame_segment(
    clip_dir: Path,
    tmp_dir: Path,
    start_frame: int,
    end_frame: int,
) -> int:
    """Copy frames *start_frame*–*end_frame* (0-based inclusive) into *tmp_dir*.

    On-disk frame files are 1-based (``frame_000001.jpg``); the *start_frame*
    and *end_frame* values from the labeler are 0-based scrubber indices.

    Returns:
        Number of frames copied.
    """
    copied = 0
    for scrubber_idx in range(start_frame, end_frame + 1):
        disk_num = scrubber_idx + 1  # 1-based filename
        src = clip_dir / f"frame_{disk_num:06d}.jpg"
        if not src.exists():
            continue
        dst = tmp_dir / f"frame_{copied + 1:06d}.jpg"
        shutil.copy2(str(src), str(dst))
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_dataset(  # pylint: disable=too-many-locals,too-many-statements
    frames_dir: Path,
    labels: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    pipeline: MhrToHml263Pipeline,
    hml_mean: np.ndarray,
    hml_std: np.ndarray,
    nlp: Language,
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
    fix_body: bool = False,
    n_augment: int = 0,
    noise_std: float = 0.05,
    cache_dir: Path | None = None,
) -> None:
    """Process labeled frame segments and write an MDM-compatible dataset directory.

    Each segment ``{start_frame, end_frame, caption}`` in *labels* becomes one
    independent motion trajectory in the dataset.

    Args:
        frames_dir: Directory containing per-clip frame subdirectories.
        labels: Mapping from clip name to list of segment dicts, each with keys
            ``start_frame`` (int), ``end_frame`` (int), and ``caption`` (str).
        output_dir: Root directory to write the dataset into.
        pipeline: Configured :class:`MhrToHml263Pipeline` instance.
        hml_mean: HumanML3D mean vector ``(263,)``.
        hml_std: HumanML3D std vector ``(263,)`` for converting augmentation
            noise from normalized space to raw HML space.
        nlp: Loaded spaCy ``Language`` model for POS tagging.
        val_fraction: Fraction of motions to assign to the validation split.
        test_fraction: Fraction of motions to assign to the test split.
        seed: Random seed for reproducible split shuffling.
        fix_body: If True, lock all non-arm HML263 features to their frame-0
            values so the training data is consistent with fixed-body inference.
        n_augment: Number of additional noisy copies to save per trajectory.
            Each copy adds Gaussian noise (std=*noise_std*) to the arm features.
        noise_std: Standard deviation of the noise added to the arm HML263
            features during augmentation, in normalized space.
        cache_dir: Directory to cache pose-estimation outputs. When provided,
            the ``(N, 22, 3)`` joint positions for each ``(clip, start, end,
            fps)`` are saved as a version-tagged ``.npy`` file so future runs
            skip re-running pose estimation. Defaults to
            ``frames_dir.parent / "mdm_cache"``.
    """
    (output_dir / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (output_dir / "texts").mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = frames_dir.parent / "mdm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    arm_mask = _arm_feature_mask()
    hml_std = np.asarray(hml_std, dtype=np.float32)
    successful_ids: list[str] = []
    motion_id = 0

    for clip_name, segments in sorted(labels.items()):
        if not segments:
            print(f"[skip] {clip_name} — no segments")
            continue
        clip_dir = frames_dir / clip_name
        if not clip_dir.is_dir():
            print(f"[skip] {clip_name} — directory not found in {frames_dir}")
            continue

        meta_path = clip_dir / "meta.json"
        clip_fps = 20.0
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                clip_fps = float(json.load(f).get("fps", 20.0))

        # Group by (start_frame, end_frame) so pose estimation runs once per
        # unique frame range regardless of how many captions share that range.
        range_to_captions: dict[tuple[int, int], list[str]] = {}
        for seg in segments:
            sf = int(seg.get("start_frame", 0))
            ef = int(seg.get("end_frame", 0))
            cap = str(seg.get("caption", "")).strip()
            if not cap or ef <= sf:
                print(f"  [skip] invalid segment {seg!r}")
                continue
            range_to_captions.setdefault((sf, ef), []).append(cap)

        for (start_frame, end_frame), captions in sorted(range_to_captions.items()):
            n_captions = len(captions)
            motion_id += 1
            id_str = f"{motion_id:06d}"
            caption_note = f"  ({n_captions} caption{'s' if n_captions > 1 else ''})"
            print(f"[{id_str}] {clip_name}  f{start_frame}-f{end_frame}{caption_note} ...")

            fps_tag = f"{clip_fps:.4g}".replace(".", "p")
            cache_file = (
                cache_dir
                / clip_name
                / f"{start_frame:06d}_{end_frame:06d}_fps{fps_tag}_v{_CACHE_VERSION}.npy"
            )

            if cache_file.exists():
                positions = np.load(cache_file)  # (N, 22, 3)
                n_frames = end_frame - start_frame + 1
                print(f"  (cache hit: {cache_file.relative_to(cache_dir)})")
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        n_frames = _copy_frame_segment(
                            clip_dir,
                            Path(tmp_dir),
                            start_frame=start_frame,
                            end_frame=end_frame,
                        )
                        if n_frames == 0:
                            print("  ✗ no frames found in range — skipping")
                            motion_id -= 1
                            continue
                        positions = pipeline.run_to_smpl_positions(
                            Path(tmp_dir), source_fps=clip_fps
                        )  # (N, 22, 3) at 20 FPS
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"  ✗ pipeline failed ({exc}) — skipping")
                        motion_id -= 1
                        continue
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, positions)

            try:
                positions, resampled = _clamp_positions_length(positions)
                hml263 = positions_to_hml263(
                    positions, hml_mean, hml_std, normalize=False
                )  # (N-1, 263) raw HML263
                if fix_body:
                    hml263 = _lock_body_to_frame0(hml263)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  ✗ post-processing failed ({exc}) — skipping")
                motion_id -= 1
                continue

            np.save(output_dir / "new_joint_vecs" / f"{id_str}.npy", hml263)
            _write_text_file(output_dir / "texts" / f"{id_str}.txt", captions, nlp)
            resample_note = " (resampled)" if resampled else ""
            print(f"  ✓ frames={n_frames}, hml263={hml263.shape}{resample_note}")
            successful_ids.append(id_str)

            # Noisy augmentations — perturb arm features around the base trajectory
            aug_rng = np.random.default_rng(seed + motion_id)
            for _ in range(n_augment):
                motion_id += 1
                aug_id_str = f"{motion_id:06d}"
                hml263_aug = hml263.copy()
                # Keep --noise_std semantics in normalized space while storing
                # raw vectors: scale Gaussian noise by per-feature Std.
                noise_norm = aug_rng.standard_normal(
                    (len(hml263), int(arm_mask.sum()))
                ).astype(np.float32)
                hml263_aug[:, arm_mask] += noise_norm * noise_std * hml_std[arm_mask]
                np.save(
                    output_dir / "new_joint_vecs" / f"{aug_id_str}.npy", hml263_aug
                )
                _write_text_file(
                    output_dir / "texts" / f"{aug_id_str}.txt", captions, nlp
                )
                successful_ids.append(aug_id_str)
            if n_augment:
                print(f"  + {n_augment} noisy augmentation(s) (noise_std={noise_std})")

    if not successful_ids:
        raise RuntimeError("No segments were processed successfully.")

    # --- splits ---
    rng = random.Random(seed)
    shuffled = list(successful_ids)
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction)) if val_fraction > 0 else 0
    n_test = max(1, round(len(shuffled) * test_fraction)) if test_fraction > 0 else 0
    # Ensure train set is never empty
    n_val = min(n_val, len(shuffled) - 1)
    n_test = min(n_test, len(shuffled) - n_val - 1)

    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val : n_val + n_test]
    train_ids = shuffled[n_val + n_test :]

    # MDM asserts len(dataset) > 1, so every split needs at least 2 IDs.
    if len(val_ids) < 2:
        val_ids = (val_ids + train_ids * 2)[:2]
    if len(test_ids) < 2:
        test_ids = (test_ids + train_ids * 2)[:2]

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (output_dir / f"{split_name}.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )
        print(f"  {split_name}: {len(ids)} motion(s)")

    print(f"\nDataset written to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pylint: disable=too-many-locals
    """Parse arguments and build the MDM dataset."""
    parser = argparse.ArgumentParser(
        description="Build an MDM-compatible dataset from labeled motion frame sequences."
    )
    _here = Path(__file__).parent
    parser.add_argument(
        "--frames_dir",
        default=str(_here / "data" / "frames"),
        help="Directory containing per-clip frame subdirectories (default: data_collection/frames/).",
    )
    parser.add_argument(
        "--labels_json",
        default=str(_here / "data" / "frames" / "labels.json"),
        help="Path to labels.json produced by the labeler (default: data_collection/frames/labels.json).",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write the dataset into."
    )
    parser.add_argument(
        "--hml_stats_dir",
        default=str(
            Path(__file__).parent.parent
            / "motion_generators"
            / "mdm"
            / "motion-diffusion-model"
            / "dataset"
            / "HumanML3D"
        ),
        help="Directory containing HumanML3D Mean.npy and Std.npy.",
    )
    parser.add_argument(
        "--sam_checkpoint_path",
        default=str(
            Path(__file__).parent
            / "sam-3d-body"
            / "checkpoints"
            / "sam-3d-body-dinov3"
            / "model.ckpt"
        ),
        help="Path to the SAM 3D Body model checkpoint.",
    )
    parser.add_argument(
        "--mhr_path",
        default=str(
            Path(__file__).parent
            / "sam-3d-body"
            / "checkpoints"
            / "sam-3d-body-dinov3"
            / "assets"
            / "mhr_model.pt"
        ),
        help="Path to the MHR model (mhr_model.pt) passed to the SAM 3D Body loader.",
    )
    parser.add_argument(
        "--fix_body",
        action="store_true",
        default=False,
        help=(
            "Lock all non-arm HML263 features to their frame-0 values in every "
            "training sequence, making fixed-body inference in-distribution."
        ),
    )
    parser.add_argument(
        "--n_augment",
        type=int,
        default=0,
        help=(
            "Number of noisy augmentation copies to generate per trajectory "
            "(default: 0). Each copy perturbs arm features with Gaussian noise."
        ),
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.05,
        help=(
            "Std-dev of noise added to the arm HML263 features per augmentation "
            "copy, in normalized space (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of motions for the validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.1,
        help="Fraction of motions for the test split (default: 0.1).",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Directory to cache raw HML263 pipeline outputs and skip re-running "
            "pose estimation on repeated calls (default: <frames_dir>/../mdm_cache)."
        ),
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    labels_json = Path(args.labels_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    hml_stats_dir = Path(args.hml_stats_dir).expanduser().resolve()
    sam_checkpoint_path = Path(args.sam_checkpoint_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    # Load labels
    with open(labels_json, encoding="utf-8") as f:
        labels: dict[str, list[dict[str, Any]]] = json.load(f)

    labeled_count = sum(1 for v in labels.values() if v)
    print(f"Found {labeled_count} labeled clip(s) in {labels_json}")

    # Build pipeline
    config = MhrToHml263Config(
        mhr_estimator_config=MhrEstimatorConfig(
            sam_checkpoint_path=str(sam_checkpoint_path),
            mhr_path=args.mhr_path,
        ),
        hml_stats_dir=hml_stats_dir,
        output_normalized=False,
    )
    pipeline = MhrToHml263Pipeline(config)
    hml_mean, hml_std = load_hml_stats(hml_stats_dir)

    # Load spaCy
    print("Loading spaCy model ...")
    nlp: Language = spacy.load("en_core_web_sm")

    # Process segments
    build_dataset(
        frames_dir=frames_dir,
        labels=labels,
        output_dir=output_dir,
        pipeline=pipeline,
        hml_mean=hml_mean,
        hml_std=hml_std,
        nlp=nlp,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        fix_body=args.fix_body,
        n_augment=args.n_augment,
        noise_std=args.noise_std,
        cache_dir=cache_dir,
    )

    # Copy normalization stats
    for stat_file in ("Mean.npy", "Std.npy"):
        src = hml_stats_dir / stat_file
        if src.exists():
            shutil.copy(src, output_dir / stat_file)
            print(f"Copied {stat_file}")
        else:
            print(f"Warning: {src} not found — {stat_file} not copied")


if __name__ == "__main__":
    main()
