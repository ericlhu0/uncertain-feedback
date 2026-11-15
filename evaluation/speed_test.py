"""The pre-registered speed test: does a checkpoint follow plain everyday speed words?

    uv run python evaluation/speed_test.py --model-path <checkpoint.pt> \\
        --clips-dir <templated clip set> --out-dir <dir> [--n-cases 40] \\
        [--phrases development|heldout] [--case-cache <npz>]

Bed (``outputs/speed_replay/PROTOCOL.md``, "Speed test"): sampled cases of a
templated-caption clip set as :mod:`run_grounding` builds them, the executed
history up to the feedback pose (``OracleCase.history``) as the pinned prefix and
the run's first caption as the request. Prompts per case: unmarked = the
utterance; slow / fast = the utterance with each of four phrases, adverbs
appended ("Lift my hand up, slowly.") and idioms prepended ("Slow down, lift my
hand up."); the development phrases select checkpoints, the held-out ones are for
confirmation only. Sampling: ``--num-samples`` per prompt, seed-locked across a
case's prompts so prompts compare sample for sample, ``--frames`` frames, the
deployed inference settings (guidance from ``mdm_config.yaml``, no steering),
each sample re-anchored on the feedback pose (``anchor_q_trajectory``), wrist
spine3-relative. Measures and criteria S1-S4: :mod:`evaluation.metrics.tempo`.

Writes ``<out-dir>/case_NNN/samples.npz`` (the wrists per prompt), ``cases.csv``
and ``summary.json`` (the verdict). A case whose ``samples.npz`` exists is read
back, not re-sampled, so an interrupted run resumes to the same numbers: every
prompt is sampled from the same locked seed, whatever the order.

``--case-cache`` stores each case's history (the expensive oracle replan of
``build_sampled_case`` is not needed here) so the screens of many checkpoints
build the cases once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.benchmarks.oracle_viz import build_sampled_case
from evaluation.metrics import tempo
from uncertain_feedback.data_collection.dataset_auto_correction.clips import (
    ClipSource,
    clip_source_from_dir,
)
from uncertain_feedback.motion_generators import make_motion_generator
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.planners.mpc.human import Human
from uncertain_feedback.planners.mpc.kinematics import (
    LEFT_ARM_CHAIN_INDICES,
    anchor_q_trajectory,
)

_LOG = "[speed]"

# (phrase, placement): adverbs are appended, idioms prepended.
PHRASES: dict[str, dict[str, list[tuple[str, str]]]] = {
    "development": {
        "slow": [
            ("slowly", "append"),
            ("slower", "append"),
            ("nice and slow", "prepend"),
            ("take your time", "prepend"),
        ],
        "fast": [
            ("quickly", "append"),
            ("faster", "append"),
            ("fast", "append"),
            ("hurry up", "prepend"),
        ],
    },
    "heldout": {
        "slow": [
            ("slow down", "prepend"),
            ("real slow", "append"),
            ("as slow as you can", "append"),
            ("no rush", "prepend"),
        ],
        "fast": [
            ("speed up", "prepend"),
            ("quick", "append"),
            ("as fast as you can", "append"),
            ("in a hurry", "append"),
        ],
    },
}


def phrased(utterance: str, phrase: str, placement: str) -> str:
    """The utterance with a tempo phrase appended after a comma or prepended before one."""
    base = utterance.strip().rstrip(".!")
    if placement == "append":
        return f"{base}, {phrase}."
    return f"{phrase[0].upper()}{phrase[1:]}, {base[0].lower()}{base[1:]}."


def prompts_for(
    utterance: str, phrases: dict[str, list[tuple[str, str]]]
) -> dict[str, list[str]]:
    """The three conditions' prompts for one case."""
    return {
        "unmarked": [utterance],
        "slow": [phrased(utterance, p, where) for p, where in phrases["slow"]],
        "fast": [phrased(utterance, p, where) for p, where in phrases["fast"]],
    }


def case_histories(
    source: ClipSource, indices: list[int], cache: Path | None
) -> dict[int, np.ndarray]:
    """Each case's executed history up to the feedback pose, cached across checkpoints."""
    stored: dict[str, np.ndarray] = {}
    if cache is not None and cache.exists():
        stored = dict(np.load(cache))
    histories = {}
    for index in indices:
        key = f"case_{index:03d}"
        if key not in stored:
            stored[key] = build_sampled_case(source, index).history
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, **stored)
        histories[index] = stored[key]
    return histories


def wrists(
    gen: MotionGenerator,
    human: Human,
    text: str,
    prefix: bool,
    num_samples: int,
    frames: int,
) -> np.ndarray:
    """``(num_samples, frames - 1, 3)`` executed spine3-relative wrist paths for ``text``."""
    positions = gen.generate_positions(
        text, human, prefix=prefix, num_samples=num_samples, num_frames=frames
    )
    q = human.ik_q_from_positions(positions[:, :, LEFT_ARM_CHAIN_INDICES])
    executed = np.stack([anchor_q_trajectory(sample, human.q) for sample in q])
    return human.wrist_from_q(executed)


def main() -> None:  # pylint: disable=too-many-locals
    """Sample every case's prompts, measure, and print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--first-case", type=int, default=0)
    parser.add_argument("--phrases", choices=sorted(PHRASES), default="development")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--case-cache", type=Path, default=None)
    parser.add_argument(
        "--static-prefix",
        action="store_true",
        help="pin the feedback pose repeated instead of the executed history",
    )
    args = parser.parse_args()

    clips_dir, out_dir = args.clips_dir.resolve(), args.out_dir.resolve()
    cache = None if args.case_cache is None else args.case_cache.resolve()
    model_path = None if args.model_path is None else args.model_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = clip_source_from_dir(clips_dir)
    runs = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))["runs"]
    indices = list(range(args.first_case, args.first_case + args.n_cases))
    histories = case_histories(source, indices, cache)
    gen = make_motion_generator(
        source.run_cfg.motion_generator,
        model_path,
        seed=args.seed,
        lock_seed=True,
        guidance_param=args.guidance,
    )
    phrases = PHRASES[args.phrases]

    rows = []
    for index in indices:
        case_dir = out_dir / f"case_{index:03d}"
        case_dir.mkdir(exist_ok=True)
        utterance = runs[index]["captions"][0]
        prompts = prompts_for(utterance, phrases)
        samples_path = case_dir / "samples.npz"
        if samples_path.exists():
            saved = dict(np.load(samples_path))
        else:
            history = histories[index]
            human = source.human.reset_human_with_q(history[0]).step(history[1:])
            saved = {
                f"{condition}_{k}": wrists(
                    gen,
                    human,
                    text,
                    not args.static_prefix,
                    args.num_samples,
                    args.frames,
                )
                for condition, texts in prompts.items()
                for k, text in enumerate(texts)
            }
            np.savez(samples_path, **saved)
            (case_dir / "prompts.json").write_text(json.dumps(prompts, indent=1))
        stats = {
            condition: tempo.condition_stats(
                [
                    tempo.sample_stats(saved[f"{condition}_{k}"])
                    for k in range(len(texts))
                ]
            )
            for condition, texts in prompts.items()
        }
        rows.append(
            {
                "case": index,
                "utterance": utterance,
                **tempo.case_row(stats["unmarked"], stats["slow"], stats["fast"]),
            }
        )
        print(
            f"{_LOG} case {index:03d} {utterance!r}: t90 slow {stats['slow']['t90']:.1f} / "
            f"unmarked {stats['unmarked']['t90']:.1f} / fast {stats['fast']['t90']:.1f}, "
            f"rush {stats['unmarked']['rush'] * 1000:.1f} mm/frame",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_dir / "cases.csv", index=False)

    verdict = tempo.speed_verdict(rows)
    verdict["model_path"] = str(model_path)
    verdict["phrases"] = args.phrases
    verdict["cases"] = f"{clips_dir.name} {indices[0]}-{indices[-1]}"
    (out_dir / "summary.json").write_text(json.dumps(verdict, indent=1))
    print(
        f"{_LOG} {verdict['cases']} {args.phrases} phrases\n{tempo.verdict_table(verdict)}"
    )


if __name__ == "__main__":
    main()
