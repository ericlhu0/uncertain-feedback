"""Re-encode the seated custom1 clips on the canonical skeleton → ``custom1_seatedcanon``.

Skeleton-only control for :mod:`graft_standing_dataset`.  That dataset changes
**two** things at once: the posture (seated → standing) and the skeleton —
``custom1`` was never encoded with ``process_file``, so its bones are neither
rigid nor the canonical t2m ones, while ``process_file``'s ``uniform_skeleton``
retargets everything onto the t2m skeleton.  This script changes only the
skeleton: each clip is decoded to global positions and re-encoded with the same
``process_file`` call, leaving posture and arm motion untouched.

``texts/``, the split files and ``Mean.npy``/``Std.npy`` are copied from
``custom1`` unchanged; ``process_file`` drops the last frame, so clips come out
``(49, 263)`` like ``custom1_standing``.

The output doubles as the drop-in **official-encoding** training set: query-time
pinned frames (``smpl_arm_aa_to_hml263_frame``) are always ``process_file``, so a
checkpoint fine-tuned here sees them on-manifold while one fine-tuned on
``custom1`` does not.  See ``CODEBASE_MAP.md`` §9 "Dataset encoding provenance".

Run from the repo root::

    uv run python src/uncertain_feedback/motion_generators/mdm/canonicalize_seated_dataset.py \\
        --verify-dir /tmp/graft_verify
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — must be set before importing pyplot
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
import matplotlib.pyplot as plt
import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from uncertain_feedback.data_collection.common.hml263 import (
    load_hml_stats,
    positions_to_hml263,
)
from uncertain_feedback.motion_generators.mdm.graft_standing_dataset import (
    _ARM_JOINT_SET,
    SRC_DATASET,
    _derotate,
    decode_raw_hml263,
)
from uncertain_feedback.planners.mpc.kinematics import SMPL_PARENTS_22
from uncertain_feedback.utils.plot import ArmVisualizer

OUT_DATASET = SRC_DATASET.parent / "custom1_seatedcanon"

PELVIS, SPINE3, L_SHOULDER, L_WRIST = 0, 9, 16, 20


def _collar_to_shoulder(frame: np.ndarray) -> float:
    """Left collar → shoulder bone length of a ``(22, 3)`` pose, in metres."""
    return float(np.linalg.norm(frame[L_SHOULDER] - frame[SMPL_PARENTS_22[L_SHOULDER]]))


def _skeleton_strip(
    name: str,
    original: np.ndarray,
    reencoded: np.ndarray,
    out_path: Path,
) -> None:
    """Original seated frames above the re-encoded ones, 4 shared timestamps.

    ``draw_smpl_skeleton`` autoscales every subplot on its own, which hides a
    uniform rescale, so the titles carry the pelvis height and the collar →
    shoulder bone length — the two numbers the retarget actually moves.
    """
    n = len(reencoded)
    frames = [0, n // 3, 2 * n // 3, n - 1]
    fig = plt.figure(figsize=(4.5 * len(frames), 9))
    for i, t in enumerate(frames):
        for row, (positions, label) in enumerate(
            ((original, "custom1"), (reencoded, "seatedcanon"))
        ):
            ax = fig.add_subplot(
                2, len(frames), row * len(frames) + i + 1, projection="3d"
            )
            ArmVisualizer.draw_smpl_skeleton(
                ax,
                positions[t],
                f"{label} f{t}  pelvis={positions[t, PELVIS, 1]:.3f}  "
                f"collar→sdr={_collar_to_shoulder(positions[t]):.3f}",
                _ARM_JOINT_SET,
            )
    fig.suptitle(f"{name}: original custom1 (top) vs canonical re-encode (bottom)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build ``custom1_seatedcanon`` and report the re-encoding deviations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-dir",
        type=Path,
        default=None,
        help="Directory for verification PNGs. Skipped when omitted.",
    )
    args = parser.parse_args()
    verify_dir = args.verify_dir.resolve() if args.verify_dir else None
    if verify_dir is not None:
        verify_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_hml_stats(SRC_DATASET)
    (OUT_DATASET / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    clip_paths = sorted((SRC_DATASET / "new_joint_vecs").glob("*.npy"))

    original_by_name: dict[str, np.ndarray] = {}
    reencoded_by_name: dict[str, np.ndarray] = {}
    wrist_dev: list[np.ndarray] = []
    src_pelvis: list[np.ndarray] = []
    out_pelvis: list[np.ndarray] = []

    for path in clip_paths:
        raw = np.load(path)
        positions = decode_raw_hml263(raw)
        features = positions_to_hml263(positions, mean, std, normalize=False)
        assert np.isfinite(features).all(), path.name
        np.save(OUT_DATASET / "new_joint_vecs" / path.name, features)

        # process_file applies one global rotation (frame 0 faces Z+), so a
        # frame-0 fit derotates the whole clip back into custom1's frame and the
        # spine3-relative wrist offsets become comparable.
        n = len(features)
        round_trip = _derotate(decode_raw_hml263(features), positions)
        wrist_dev.append(
            np.linalg.norm(
                (round_trip[:, L_WRIST] - round_trip[:, SPINE3])
                - (positions[:n, L_WRIST] - positions[:n, SPINE3]),
                axis=-1,
            )
        )
        src_pelvis.append(raw[:, 3])
        out_pelvis.append(features[:, 3])

        original_by_name[path.stem] = positions
        reencoded_by_name[path.stem] = round_trip
        print(f"  {path.stem}: {raw.shape} → {features.shape} {features.dtype}")

    shutil.copytree(SRC_DATASET / "texts", OUT_DATASET / "texts", dirs_exist_ok=True)
    for name in ("train.txt", "val.txt", "test.txt", "Mean.npy", "Std.npy"):
        shutil.copy2(SRC_DATASET / name, OUT_DATASET / name)

    dev = np.concatenate(wrist_dev)
    per_clip_dev = np.array([d.mean() for d in wrist_dev])
    src_h = np.concatenate(src_pelvis)
    out_h = np.concatenate(out_pelvis)
    print(f"\n{len(clip_paths)} clips written to {OUT_DATASET}")
    print(
        "spine3-relative left-wrist deviation (cm): "
        f"mean={dev.mean() * 100:.3f} max={dev.max() * 100:.3f} "
        f"worst-clip mean={per_clip_dev.max() * 100:.3f}"
    )
    print(
        f"pelvis height custom1:      mean={src_h.mean():.4f} "
        f"min={src_h.min():.4f} max={src_h.max():.4f}"
    )
    print(
        f"pelvis height seatedcanon:  mean={out_h.mean():.4f} "
        f"min={out_h.min():.4f} max={out_h.max():.4f}"
    )

    if verify_dir is None:
        return
    for name in sorted(reencoded_by_name)[:2]:
        _skeleton_strip(
            name,
            original_by_name[name],
            reencoded_by_name[name],
            verify_dir / f"04_seatedcanon_{name}.png",
        )
    print(f"Verification images in {verify_dir}")


if __name__ == "__main__":
    main()
