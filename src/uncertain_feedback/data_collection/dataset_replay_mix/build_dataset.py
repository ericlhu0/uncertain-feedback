"""Mix HumanML3D back into a correction fine-tune set, so fine-tuning keeps the prior.

A fine-tune on correction clips alone forgets MDM's prior within a few hundred steps
(every prompt becomes the seated body moving its left arm), because nothing in the data
says the seated left-arm behaviour belongs to correction captions only. This writes a
dataset whose train split holds every HumanML3D train motion next to the correction
clips, so each batch rehearses the original distribution.

Every file is a symlink, nothing is copied. Corrections are repeated under distinct ids
(``C<id>_r<k>``) instead of listing an id twice: the humanml loader sizes an epoch by
unique clips but indexes a length-sorted name list, so a repeated id would silently push
the longest motions out of reach. HumanML3D ids become ``H<id>``. The id's first letter
is each clip's source, which ``train_leftarm.py`` reads (``--distill_weight``,
``--correction_fraction``, ``--clean_prefix``); the captions are the sources' own.

    uv run python src/uncertain_feedback/data_collection/dataset_replay_mix/build_dataset.py \\
        --corrections .../motion-diffusion-model/dataset/correction_auto1100_vlm \\
        --output_dir .../motion-diffusion-model/dataset/correction_auto1100_vlm_hml50 \\
        --repeats 9

``--retimed DIR --retimed_repeats N`` adds a third source, the retimed copies of a
``build_dataset.py --retimes_only`` set (captioned by ``tempo_captions.py``), as
``R<id>_r<k>``: trained on like HumanML3D (no distillation), but carrying an
executed-history prefix like the corrections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uncertain_feedback.consts import HUMANML3D_DIR
from uncertain_feedback.data_collection.common.dataset import copy_stats

SPLITS = ("train", "val", "test")


def _link(src_dir: Path, src_id: str, out_dir: Path, out_id: str) -> bool:
    motion = src_dir / "new_joint_vecs" / f"{src_id}.npy"
    if not motion.exists():
        return False
    (out_dir / "new_joint_vecs" / f"{out_id}.npy").symlink_to(motion)
    (out_dir / "texts" / f"{out_id}.txt").symlink_to(
        src_dir / "texts" / f"{src_id}.txt"
    )
    return True


def _check_stats(source: Path, humanml: Path) -> None:
    for stat in ("Mean.npy", "Std.npy"):
        if not np.allclose(np.load(source / stat), np.load(humanml / stat)):
            raise ValueError(f"{source}/{stat} differs from HumanML3D's")


def _link_split(
    source: Path, letter: str, split: str, repeats: int, output_dir: Path
) -> list[str]:
    """Link one source's ``split`` into the mix, ``repeats`` copies of each train id."""
    ids = []
    for i in (source / f"{split}.txt").read_text().split():
        for k in range(repeats if split == "train" else 1):
            if _link(source, i, output_dir, f"{letter}{i}_r{k}"):
                ids.append(f"{letter}{i}_r{k}")
    return ids


def build_replay_mix(  # pylint: disable=too-many-arguments
    corrections: Path,
    humanml: Path,
    output_dir: Path,
    repeats: int,
    retimed: Path | None = None,
    retimed_repeats: int = 0,
) -> None:
    """Write the mixed dataset; corrections are repeated ``repeats`` times in train."""
    _check_stats(corrections, humanml)
    corrections, humanml = corrections.resolve(), humanml.resolve()
    if retimed is not None:
        _check_stats(retimed, humanml)
        retimed = retimed.resolve()
    for sub in ("new_joint_vecs", "texts"):
        (output_dir / sub).mkdir(parents=True)

    for split in SPLITS:
        ids = [
            f"H{i}"
            for i in (humanml / f"{split}.txt").read_text().split()
            if _link(humanml, i, output_dir, f"H{i}")
        ]
        n_humanml = len(ids)
        ids += _link_split(corrections, "C", split, repeats, output_dir)
        n_corrections = len(ids) - n_humanml
        if retimed is not None:
            ids += _link_split(retimed, "R", split, retimed_repeats, output_dir)
        (output_dir / f"{split}.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )
        print(
            f"  {split}: {n_humanml} HumanML3D + {n_corrections} correction + "
            f"{len(ids) - n_humanml - n_corrections} retimed entries"
        )
    copy_stats(corrections, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--humanml", type=Path, default=HUMANML3D_DIR)
    parser.add_argument(
        "--repeats",
        type=int,
        default=9,
        help="copies of each correction train clip; 9 makes 2640 clips ~1:1 with "
        "HumanML3D's 23384 train motions",
    )
    parser.add_argument(
        "--retimed",
        type=Path,
        default=None,
        help="a retimed-copies set (build_dataset.py --retimes_only, tempo-captioned) "
        "to add as R... ids",
    )
    parser.add_argument(
        "--retimed_repeats",
        type=int,
        default=0,
        help="copies of each retimed train clip (with --retimed)",
    )
    args = parser.parse_args()
    if (args.retimed is None) != (args.retimed_repeats == 0):
        parser.error("--retimed and --retimed_repeats go together")
    build_replay_mix(
        args.corrections,
        args.humanml,
        args.output_dir,
        args.repeats,
        retimed=args.retimed,
        retimed_repeats=args.retimed_repeats,
    )


if __name__ == "__main__":
    main()
