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
    ├── new_joint_vecs/000001.npy   # (N, 263) float32 normalized HML263
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
)

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
    nlp: Language,
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
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
        nlp: Loaded spaCy ``Language`` model for POS tagging.
        val_fraction: Fraction of motions to assign to the validation split.
        test_fraction: Fraction of motions to assign to the test split.
        seed: Random seed for reproducible split shuffling.
    """
    (output_dir / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (output_dir / "texts").mkdir(parents=True, exist_ok=True)

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

        for seg in segments:
            start_frame: int = int(seg.get("start_frame", 0))
            end_frame: int = int(seg.get("end_frame", 0))
            caption: str = str(seg.get("caption", "")).strip()
            if not caption or end_frame <= start_frame:
                print(f"  [skip] invalid segment {seg!r}")
                continue

            motion_id += 1
            id_str = f"{motion_id:06d}"
            print(f"[{id_str}] {clip_name}  f{start_frame}-f{end_frame} ...")

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
                    hml263 = pipeline.run(Path(tmp_dir))  # (N, 263)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"  ✗ pipeline failed ({exc}) — skipping")
                    motion_id -= 1
                    continue

            np.save(output_dir / "new_joint_vecs" / f"{id_str}.npy", hml263)
            _write_text_file(output_dir / "texts" / f"{id_str}.txt", [caption], nlp)
            print(f"  ✓ frames={n_frames}, hml263={hml263.shape}")
            successful_ids.append(id_str)

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

    # When there are too few samples to fill every split, duplicate from train so
    # the data loader never receives an empty file.
    fallback = train_ids[:1]
    if not val_ids:
        val_ids = fallback
    if not test_ids:
        test_ids = fallback

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
    parser.add_argument(
        "--frames_dir",
        required=True,
        help="Directory containing per-clip frame subdirectories (from extract_all_frames.py).",
    )
    parser.add_argument(
        "--labels_json",
        required=True,
        help="Path to labels.json produced by the labeler.",
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
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    labels_json = Path(args.labels_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    hml_stats_dir = Path(args.hml_stats_dir).expanduser().resolve()
    sam_checkpoint_path = Path(args.sam_checkpoint_path).expanduser()

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
    )
    pipeline = MhrToHml263Pipeline(config)

    # Load spaCy
    print("Loading spaCy model ...")
    nlp: Language = spacy.load("en_core_web_sm")

    # Process segments
    build_dataset(
        frames_dir=frames_dir,
        labels=labels,
        output_dir=output_dir,
        pipeline=pipeline,
        nlp=nlp,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
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
