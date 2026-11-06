"""The prior table under both T2M evaluators: MDM's 2025 re-upload and the 2022 original.

    env -u LD_LIBRARY_PATH uv run python evaluation/paper_tables/prior_evaluators.py \\
        --original <dir holding text_mot_match/model/finest.tar> --cache <dir> \\
        --model base=<checkpoint> --model replay=<checkpoint> ... \\
        [--pair base:replay] [--reference <a run_prior metrics.json>]

MDM's evaluator download was re-served in 2025-01 with a differently trained
text-motion matcher that reads R-precision about 0.05 low and makes small FID
gaps appear (2026-10-21 audit, §14); the 2022 original, kept in T2M-GPT's
extractor archive, reproduces the published real-motion values on our rebuilt
data (R@1 0.511, R@3 0.797, matching 2.974). The original's weights are laid out
next to the re-upload's normalization folder under ``<cache>/evaluator_2022``.

Real motions are scored first with both evaluators, which is the check on the
weights. Each model's samples are generated as ``run_prior.py`` makes them (seed
1, the configured guidance, nothing pinned) and cached under ``<cache>`` with
their body statistics, so a rerun only embeds. ``--reference`` compares the
re-upload's numbers to a stored ``metrics.json``, which confirms the samples are
the ones that table used. ``--pair a:b`` adds a paired case-bootstrap 95% CI of
b − a on R@3 (per-item top-3 over 200 random batchings of 32) and FID, under
the original evaluator. Needs a GPU for the sampling; embedding runs anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import prior
from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator

# pylint: disable=wrong-import-order,import-error
from data_loaders.humanml.utils.metrics import (  # noqa: E402  # isort: skip
    calculate_activation_statistics,
    calculate_frechet_distance,
    euclidean_distance_matrix,
)

_SAMPLE_SEED = 1
_N_BOOT = 300
_N_BATCHINGS = 200


def original_evaluator_dir(original: Path, cache: Path) -> Path:
    """Lay the 2022 weights out beside the re-upload's normalization folder."""
    directory = cache / "evaluator_2022"
    (directory / "text_mot_match" / "model").mkdir(parents=True, exist_ok=True)
    links = (
        (
            directory / "text_mot_match" / "model" / "finest.tar",
            original / "text_mot_match" / "model" / "finest.tar",
        ),
        (directory / "Comp_v6_KLD01", prior.T2M_EVALUATOR_DIR / "Comp_v6_KLD01"),
    )
    for link, target in links:
        if not link.exists():
            link.symlink_to(target)
    return directory


def load_evaluator(directory: Path, device: torch.device) -> prior.T2MEvaluator:
    """A ``T2MEvaluator`` reading its weights from ``directory``."""
    prior.T2M_EVALUATOR_DIR = directory
    return prior.T2MEvaluator(device)


def sample_or_load(
    name: str,
    checkpoint: Path,
    test: prior.TestSet,
    evaluators: dict[str, prior.T2MEvaluator],
    cache: Path,
) -> dict[str, np.ndarray]:
    """A model's embeddings under every evaluator plus body statistics, cached."""
    path = cache / f"{name}.npz"
    if not path.exists():
        gen = MdmMotionGenerator(model_path=checkpoint)
        raw = gen.sample_text_to_motion(
            list(test.captions), test.lengths, seed=_SAMPLE_SEED
        )
        pelvis_y, travel = prior.body_usage(raw, test.lengths)
        np.savez(
            path,
            pelvis_y=pelvis_y,
            travel=travel,
            **{
                tag: e.motion_embeddings(raw, test.lengths)
                for tag, e in evaluators.items()
            },
        )
        print(f"{name}: sampled", flush=True)
        del gen
        torch.cuda.empty_cache()
    saved = np.load(path)
    return {key: saved[key] for key in saved.files}


def per_item_top3(
    text: np.ndarray, emb: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Each item's top-3 hit rate over random batchings of 32, for a paired bootstrap."""
    hits, count = np.zeros(len(emb)), np.zeros(len(emb))
    for _ in range(_N_BATCHINGS):
        order = rng.permutation(len(emb))[: len(emb) // 32 * 32].reshape(-1, 32)
        for batch in order:
            ranks = np.argsort(
                np.argsort(euclidean_distance_matrix(text[batch], emb[batch]), axis=1),
                axis=1,
            )
            hits[batch] += ranks[np.arange(32), np.arange(32)] < 3
            count[batch] += 1
    return hits / count


def fid(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between two embedding sets."""
    return float(
        calculate_frechet_distance(
            *calculate_activation_statistics(a), *calculate_activation_statistics(b)
        )
    )


def paired_bootstrap(
    text: np.ndarray, real: np.ndarray, first: np.ndarray, second: np.ndarray
) -> str:
    """``second − first`` on R@3 and FID with case-bootstrap 95% CIs."""
    rng = np.random.default_rng(0)
    top3 = {
        "first": per_item_top3(text, first, rng),
        "second": per_item_top3(text, second, rng),
    }
    draws = [rng.integers(0, len(real), len(real)) for _ in range(_N_BOOT)]
    d_r3 = np.array([top3["second"][i].mean() - top3["first"][i].mean() for i in draws])
    d_fid = np.array([fid(real[i], second[i]) - fid(real[i], first[i]) for i in draws])
    return (
        f"ΔR@3 {top3['second'].mean() - top3['first'].mean():+.3f} "
        f"[{np.percentile(d_r3, 2.5):+.3f}, {np.percentile(d_r3, 97.5):+.3f}]; "
        f"ΔFID {fid(real, second) - fid(real, first):+.3f} "
        f"[{np.percentile(d_fid, 2.5):+.3f}, {np.percentile(d_fid, 97.5):+.3f}]"
    )


def _metric_line(name: str, metrics: dict[str, float]) -> str:
    return (
        f"  {name:13s} R@1 {metrics['R@1']:.3f}  R@3 {metrics['R@3']:.3f}  "
        f"FID {metrics['FID']:.3f}  matching {metrics['matching']:.3f}  "
        f"diversity {metrics['diversity']:.3f}"
    )


def main() -> None:  # pylint: disable=too-many-locals
    """Score real motions and every model under both evaluators; write and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", default=[], metavar="NAME=CHECKPOINT"
    )
    parser.add_argument("--pair", action="append", default=[], metavar="FIRST:SECOND")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    test = prior.load_test_set()
    directories = {
        "reupload": prior.T2M_EVALUATOR_DIR,
        "original": original_evaluator_dir(args.original, args.cache),
    }
    evaluators = {tag: load_evaluator(d, device) for tag, d in directories.items()}
    text = {tag: e.text_embeddings(test) for tag, e in evaluators.items()}
    real = {
        tag: e.motion_embeddings(test.motions, test.lengths)
        for tag, e in evaluators.items()
    }

    models = dict(spec.split("=", 1) for spec in args.model)
    samples = {
        name: sample_or_load(name, Path(path), test, evaluators, args.cache)
        for name, path in models.items()
    }
    zeros = np.zeros(len(test.lengths))
    table: dict[str, dict[str, dict[str, float]]] = {}
    for tag in directories:
        rows = {
            "real motions": prior.prior_metrics(
                test, text[tag], real[tag], real[tag], zeros, zeros
            )
        }
        for name, saved in samples.items():
            rows[name] = prior.prior_metrics(
                test,
                text[tag],
                real[tag],
                saved[tag],
                saved["pelvis_y"],
                saved["travel"],
            )
        table[tag] = rows
    (args.cache / "prior_evaluators.json").write_text(json.dumps(table, indent=2))

    reference = (
        json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else {}
    )
    for tag, rows in table.items():
        print(f"\nevaluator: {tag}")
        for name, metrics in rows.items():
            line = _metric_line(name, metrics)
            if tag == "reupload" and name in reference:
                line += (
                    f"   (reference: R@3 {reference[name]['R@3']:.3f}, "
                    f"FID {reference[name]['FID']:.3f})"
                )
            print(line)
    for pair in args.pair:
        first, _, second = pair.partition(":")
        print(
            f"\noriginal evaluator, {second} − {first}: "
            + paired_bootstrap(
                text["original"],
                real["original"],
                samples[first]["original"],
                samples[second]["original"],
            )
        )


if __name__ == "__main__":
    main()
