"""Build a speed-variant MDM dataset from cached pose-estimation positions.

Feasibility probe for speed-language fine-tuning: each cached ``(N, 22, 3)``
position segment becomes several retimed variants (fast / normal / slow, plus a
configuration-dependent variant that slows while the wrist is above the
shoulder), captioned with matching speed language.

All variants are forced to the same clip length (``_CLIP_FRAMES``), and every
frame is in transit: the fast variant covers the whole path, slower variants
keep a moving window of their retimed path. Two failure modes drove this: with
variant-specific lengths, sequence length perfectly predicts speed and MDM
learns length→speed from the frame mask, ignoring the caption (verified: the
same prompt sampled at 51 vs 180 frames moved 3× faster at 51); with
hold-padding to a uniform length, "parked at the target pose" dominates the
frame distribution and the fine-tuned model teleports to the end state and
freezes instead of moving.

Usage::

    uv run python src/uncertain_feedback/data_collection/build_speed_dataset.py \\
        --output_dir ./speed_mdm_dataset/

Reads cached positions directly (``<frames_dir>/../mdm_cache``) — no pose
estimation is run, so segments without a ``_v2`` cache entry are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import spacy

from uncertain_feedback.data_collection.build_mdm_dataset import (
    _lock_body_to_frame0,
    _write_text_file,
)
from uncertain_feedback.data_collection.mhr_to_hml263_pipeline import (
    resample_positions,
)
from uncertain_feedback.data_collection.smpl_to_hml263 import (
    load_hml_stats,
    positions_to_hml263,
)

_CACHE_VERSION = 2

_L_SHOULDER, _L_WRIST = 16, 20

_FAST_FRAMES = 51
_NORMAL_FRAMES = 96
_SLOW_FRAMES = 180
_CLIP_FRAMES = 51
_COND_SLOW_FACTOR = 3.0
_COND_MIN_REGION = 0.15

_SPEED_WORDS = ("slow", "gently", "quick", "fast", "rapid", "swift", "relax")

_TEMPLATE_STYLES: dict[str, dict[str, list[str]]] = {
    # Plain adverbs: nearly invisible to CLIP ("quickly" vs "very slowly"
    # cosine 0.9936 — closer than "raise" vs "lower").
    "adverb": {
        "fast": ["{c} quickly", "quickly {c}", "{c} at a fast speed"],
        "normal": ["{c}"],
        "slow": ["{c} very slowly", "slowly {c}", "{c} at a very slow speed"],
    },
    # Lexically distinctive speed phrases: ~6× the CLIP separation
    # ("sprint" vs "glacial crawl" cosine 0.9137).
    "vivid": {
        "fast": [
            "{c}, sprint",
            "{c} at lightning speed",
            "{c}, darting like a startled cat",
        ],
        "normal": ["{c}"],
        "slow": [
            "{c}, glacial crawl",
            "{c} at a snail's pace",
            "{c}, creeping like a sloth",
        ],
    },
}
_COND_TEMPLATES: list[str] = [
    "{c}, moving very slowly while my hand is above my shoulder",
    "{c}, and slow down when my hand is above my shoulder",
    "{c}, going slower whenever my hand is higher than my shoulder",
]

# Whole base motions held out of training (all their variants).
_VAL_MOTIONS = {("IMG_3503", 448, 468), ("IMG_3537", 1404, 1463)}
_TEST_MOTIONS = {("IMG_3503", 1287, 1307), ("IMG_3537", 2253, 2294)}


def _above_shoulder(positions: np.ndarray) -> np.ndarray:
    """Per-frame bool: wrist Y above shoulder Y."""
    return positions[:, _L_WRIST, 1] > positions[:, _L_SHOULDER, 1]


def _hold_pad(positions: np.ndarray, target: int) -> np.ndarray:
    """Extend to *target* frames by holding the final pose."""
    pad = np.repeat(positions[-1:], target - len(positions), axis=0)
    return np.concatenate([positions, pad], axis=0)


def _fit_clip(positions: np.ndarray) -> np.ndarray:
    """Force exactly ``_CLIP_FRAMES`` frames: hold-pad short, best-mix-crop long.

    Every variant carries the same sequence length so length cannot stand in
    for speed — the caption is the only speed cue. Long clips keep the window
    with the best balance of above/below-shoulder frames so the conditional
    variant retains both regimes.
    """
    n = len(positions)
    if n == _CLIP_FRAMES:
        return positions
    if n < _CLIP_FRAMES:
        return _hold_pad(positions, _CLIP_FRAMES)
    above = _above_shoulder(positions)
    starts = range(0, n - _CLIP_FRAMES + 1, 4)
    best = max(
        starts,
        key=lambda s: min(
            above[s : s + _CLIP_FRAMES].mean(), 1.0 - above[s : s + _CLIP_FRAMES].mean()
        ),
    )
    return positions[best : best + _CLIP_FRAMES]


def retime_conditional(positions: np.ndarray) -> np.ndarray | None:
    """Retime so above-shoulder frames run ``_COND_SLOW_FACTOR`` slower.

    Below-shoulder frames run at the normal-variant rate. Returns None when the
    motion lacks both regions or the result would exceed MDM's length limit.
    """
    above = _above_shoulder(positions)
    if not _COND_MIN_REGION <= above.mean() <= 1.0 - _COND_MIN_REGION:
        return None
    n = len(positions)
    base_rate = _NORMAL_FRAMES / n
    weights = np.where(above, _COND_SLOW_FACTOR, 1.0) * base_rate
    cum = np.concatenate([[0.0], np.cumsum((weights[:-1] + weights[1:]) / 2)])
    n_out = int(round(cum[-1])) + 1
    new_t = np.linspace(0.0, cum[-1], n_out)
    src_idx = np.interp(new_t, cum, np.arange(n, dtype=np.float64))
    flat = positions.reshape(n, -1)
    out = np.stack(
        [np.interp(src_idx, np.arange(n), flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    ).astype(np.float32)
    return out.reshape(n_out, 22, 3)


def _base_captions(captions: list[str]) -> list[str]:
    """Drop captions that already carry speed/manner language."""
    kept = [c for c in captions if not any(w in c.lower() for w in _SPEED_WORDS)]
    return [c.rstrip(".").strip() for c in kept]


def _variant_captions(variant: str, captions: list[str], style: str) -> list[str]:
    templates = (
        _COND_TEMPLATES if variant == "cond" else _TEMPLATE_STYLES[style][variant]
    )
    return [
        templates[i % len(templates)].format(c=c[0].lower() + c[1:])
        for i, c in enumerate(captions[:3])
    ]


def build(
    cache_dir: Path,
    labels_path: Path,
    output_dir: Path,
    hml_stats_dir: Path,
    caption_style: str = "adverb",
) -> None:
    (output_dir / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (output_dir / "texts").mkdir(parents=True, exist_ok=True)
    hml_mean, hml_std = load_hml_stats(hml_stats_dir)
    nlp = spacy.load("en_core_web_sm")
    labels: dict[str, list[dict]] = json.loads(labels_path.read_text(encoding="utf-8"))

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    motion_id = 0
    n_cond = 0

    for clip_name, segments in sorted(labels.items()):
        range_to_captions: dict[tuple[int, int], list[str]] = {}
        for seg in segments:
            key = (int(seg["start_frame"]), int(seg["end_frame"]))
            range_to_captions.setdefault(key, []).append(str(seg["caption"]).strip())

        for (sf, ef), captions in sorted(range_to_captions.items()):
            matches = list(
                (cache_dir / clip_name).glob(
                    f"{sf:06d}_{ef:06d}_fps*_v{_CACHE_VERSION}.npy"
                )
            )
            if not matches:
                print(f"[skip] {clip_name} f{sf}-f{ef} — no v{_CACHE_VERSION} cache")
                continue
            positions = np.load(matches[0])
            base = _base_captions(captions)
            if not base:
                print(f"[skip] {clip_name} f{sf}-f{ef} — no speed-free captions")
                continue

            if (clip_name, sf, ef) in _VAL_MOTIONS:
                split = "val"
            elif (clip_name, sf, ef) in _TEST_MOTIONS:
                split = "test"
            else:
                split = "train"

            variants: dict[str, np.ndarray] = {
                "fast": _fit_clip(resample_positions(positions, _FAST_FRAMES)),
                "normal": _fit_clip(resample_positions(positions, _NORMAL_FRAMES)),
                "slow": _fit_clip(resample_positions(positions, _SLOW_FRAMES)),
            }
            cond = retime_conditional(positions)
            if cond is not None:
                variants["cond"] = _fit_clip(cond)
                n_cond += 1

            for variant, pos in variants.items():
                motion_id += 1
                id_str = f"{motion_id:06d}"
                hml263 = positions_to_hml263(pos, hml_mean, hml_std, normalize=False)
                hml263 = _lock_body_to_frame0(hml263)
                np.save(output_dir / "new_joint_vecs" / f"{id_str}.npy", hml263)
                _write_text_file(
                    output_dir / "texts" / f"{id_str}.txt",
                    _variant_captions(variant, base, caption_style),
                    nlp,
                )
                splits[split].append(id_str)
            print(
                f"[{clip_name} f{sf}-f{ef}] {len(variants)} variants "
                f"({'with' if cond is not None else 'no'} cond) → {split}"
            )

    # MDM asserts len(dataset) > 1 per split.
    for name in ("val", "test"):
        if len(splits[name]) < 2:
            splits[name] = (splits[name] + splits["train"] * 2)[:2]
    for name, ids in splits.items():
        (output_dir / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"{name}: {len(ids)} motions")
    for stat in ("Mean.npy", "Std.npy"):
        shutil.copy(hml_stats_dir / stat, output_dir / stat)
    print(f"{motion_id} motions total ({n_cond} conditional) → {output_dir}")


def main() -> None:
    _here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default=str(_here / "data" / "mdm_cache"))
    parser.add_argument(
        "--labels_json", default=str(_here / "data" / "frames" / "labels.json")
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--caption_style", choices=sorted(_TEMPLATE_STYLES), default="adverb"
    )
    parser.add_argument(
        "--hml_stats_dir",
        default=str(
            _here.parent
            / "motion_generators"
            / "mdm"
            / "motion-diffusion-model"
            / "dataset"
            / "HumanML3D"
        ),
    )
    args = parser.parse_args()
    build(
        Path(args.cache_dir).expanduser().resolve(),
        Path(args.labels_json).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        Path(args.hml_stats_dir).expanduser().resolve(),
        caption_style=args.caption_style,
    )


if __name__ == "__main__":
    main()
