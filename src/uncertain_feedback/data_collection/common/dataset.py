"""HumanML3D dataset-writing helpers shared by every data-collection pipeline.

Each pipeline in this package (recorded video, sampled corrections, the
trajectory editor) ends by writing the same on-disk layout MDM's ``humanml``
loader reads::

    output_dir/
    ├── new_joint_vecs/000001.npy   # (N, 263) float32 raw (unnormalized) HML263
    ├── texts/000001.txt            # caption#w/POS w/POS ...#0.0#0.0
    ├── train.txt / val.txt / test.txt
    └── Mean.npy / Std.npy          # copied from the stats directory
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
from spacy.language import Language

# SMPL-22 left arm joint indices (0-indexed): left_shoulder=16, left_elbow=18, left_wrist=20
_L_ARM_JOINTS = {16, 18, 20}


def _pos_tag(nlp: Language, caption: str) -> str:
    """Return space-separated ``word/POS`` tokens for *caption* using spaCy."""
    doc = nlp(caption)
    return " ".join(f"{token.text}/{token.pos_}" for token in doc)


def write_text_file(path: Path, captions: list[str], nlp: Language) -> None:
    """Write one MDM-format text annotation file.

    Each line has the format ``caption#tokens#0.0#0.0`` where ``tokens`` is a
    space-separated sequence of ``word/POSTAG`` pairs and both timestamps are
    ``0.0`` to indicate the label covers the full motion clip.
    """
    lines = [f"{cap}#{_pos_tag(nlp, cap)}#0.0#0.0" for cap in captions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def arm_feature_mask() -> np.ndarray:
    """Return a (263,) bool mask that is True for left-arm HML263 features."""
    mask = np.zeros(263, dtype=bool)
    for j in _L_ARM_JOINTS:
        mask[4 + (j - 1) * 3 : 4 + (j - 1) * 3 + 3] = True  # positions
        mask[67 + (j - 1) * 6 : 67 + (j - 1) * 6 + 6] = True  # rotations
        mask[193 + j * 3 : 193 + j * 3 + 3] = True  # velocities
    return mask


def lock_body_to_frame0(hml263: np.ndarray) -> np.ndarray:
    """Copy frame-0 values of all non-arm features to every frame.

    Makes the training data fully consistent with fixed-body inference: the
    model only ever sees a static body with varying left-arm motion.
    """
    arm_mask = arm_feature_mask()
    result = hml263.copy()
    result[:, ~arm_mask] = result[0:1, ~arm_mask]
    return result


def write_splits(
    output_dir: Path,
    ids: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> None:
    """Shuffle *ids* into ``train.txt`` / ``val.txt`` / ``test.txt``.

    MDM asserts ``len(dataset) > 1``, so val and test are topped up from train
    whenever the requested fractions leave them with fewer than 2 IDs.
    """
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction)) if val_fraction > 0 else 0
    n_test = max(1, round(len(shuffled) * test_fraction)) if test_fraction > 0 else 0
    # Ensure train set is never empty
    n_val = min(n_val, len(shuffled) - 1)
    n_test = min(n_test, len(shuffled) - n_val - 1)

    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val : n_val + n_test]
    train_ids = shuffled[n_val + n_test :]

    if len(val_ids) < 2:
        val_ids = (val_ids + train_ids * 2)[:2]
    if len(test_ids) < 2:
        test_ids = (test_ids + train_ids * 2)[:2]

    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:
        (output_dir / f"{split_name}.txt").write_text(
            "\n".join(split_ids) + "\n", encoding="utf-8"
        )
        print(f"  {split_name}: {len(split_ids)} motion(s)")


def copy_stats(stats_dir: Path, output_dir: Path) -> None:
    """Copy the HumanML3D normalization stats into a built dataset."""
    for stat_file in ("Mean.npy", "Std.npy"):
        src = stats_dir / stat_file
        if src.exists():
            shutil.copy2(src, output_dir / stat_file)
        else:
            print(f"Warning: {src} not found — {stat_file} not copied")
