"""The replay mix links its sources under ids that name them, repeated in train only."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uncertain_feedback.data_collection.dataset_replay_mix.build_dataset import (
    build_replay_mix,
)


def _dataset(root: Path, ids: dict[str, list[str]], std: float = 1.0) -> Path:
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    np.save(root / "Mean.npy", np.zeros(263))
    np.save(root / "Std.npy", np.full(263, std))
    for split, names in ids.items():
        (root / f"{split}.txt").write_text("\n".join(names) + "\n")
        for name in names:
            np.save(root / "new_joint_vecs" / f"{name}.npy", np.zeros((50, 263)))
            (root / "texts" / f"{name}.txt").write_text(f"{name}#x/VERB#0.0#0.0\n")
    return root


def _ids(path: Path) -> list[str]:
    return path.read_text().split()


def test_sources_are_linked_under_their_letters(tmp_path: Path) -> None:
    humanml = _dataset(
        tmp_path / "h", {"train": ["h1", "h2"], "val": ["h3"], "test": ["h4"]}
    )
    corrections = _dataset(
        tmp_path / "c", {"train": ["c1"], "val": ["c2"], "test": ["c3"]}
    )
    retimed = _dataset(
        tmp_path / "r", {"train": ["r1", "r2"], "val": ["r3"], "test": ["r4"]}
    )
    out = tmp_path / "mix"

    build_replay_mix(
        corrections, humanml, out, repeats=2, retimed=retimed, retimed_repeats=3
    )

    assert _ids(out / "train.txt") == ["Hh1", "Hh2", "Cc1_r0", "Cc1_r1"] + [
        f"R{r}_r{k}" for r in ("r1", "r2") for k in range(3)
    ]
    assert _ids(out / "val.txt") == ["Hh3", "Cc2_r0", "Rr3_r0"]
    assert _ids(out / "test.txt") == ["Hh4", "Cc3_r0", "Rr4_r0"]
    motion = out / "new_joint_vecs" / "Rr1_r2.npy"
    assert (
        motion.is_symlink()
        and motion.resolve() == (retimed / "new_joint_vecs" / "r1.npy").resolve()
    )
    assert (out / "texts" / "Cc1_r1.txt").read_text() == "c1#x/VERB#0.0#0.0\n"


def test_without_retimed_the_mix_is_as_before(tmp_path: Path) -> None:
    humanml = _dataset(tmp_path / "h", {"train": ["h1"], "val": ["h2"], "test": ["h3"]})
    corrections = _dataset(
        tmp_path / "c", {"train": ["c1"], "val": ["c2"], "test": ["c3"]}
    )

    build_replay_mix(corrections, humanml, tmp_path / "mix", repeats=1)

    assert _ids(tmp_path / "mix" / "train.txt") == ["Hh1", "Cc1_r0"]


def test_a_retimed_set_with_other_stats_is_refused(tmp_path: Path) -> None:
    humanml = _dataset(tmp_path / "h", {"train": ["h1"], "val": ["h2"], "test": ["h3"]})
    corrections = _dataset(
        tmp_path / "c", {"train": ["c1"], "val": ["c2"], "test": ["c3"]}
    )
    retimed = _dataset(
        tmp_path / "r", {"train": ["r1"], "val": ["r2"], "test": ["r3"]}, std=2.0
    )

    with pytest.raises(ValueError, match="Std.npy differs"):
        build_replay_mix(corrections, humanml, tmp_path / "mix", 1, retimed, 1)
