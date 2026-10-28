"""Score MDM checkpoints on HumanML3D, MDM's pretraining distribution.

    uv run python evaluation/run_prior.py --out-dir outputs/prior_eval \\
        --model base=<pt> --model deployed=<pt> [--model name=<pt> ...] [--showcase]

One sample per test caption at its motion's own length (MDM's protocol), plus 20 samples
for each of 100 captions for multimodality; see :mod:`evaluation.metrics.prior`. Each
model's embeddings are cached in ``<out-dir>/cache/<name>.npz``, so a re-run, or a run
adding a checkpoint, only samples what is new. Writes ``metrics.json`` and, with
``--showcase``, ``showcase.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics.prior import (
    SHOWCASE,
    T2MEvaluator,
    body_usage,
    load_test_set,
    prior_metrics,
    render_showcase,
)
from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator

_MM_PROMPTS, _MM_REPEATS = 100, 20
_SHOWCASE_FRAMES = 120


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", required=True, help="name=checkpoint.pt"
    )
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N test ids (smoke test)"
    )
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="prepended to every generation prompt (a model whose replay data carried "
        "a caption tag); the evaluator still reads the original captions",
    )
    parser.add_argument(
        "--null-prompt",
        default=None,
        help="guide against this prompt instead of the empty condition (the bare tag of a "
        "train_leftarm.py --tag_null fine-tune)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="sampling seed (multimodality draws use 100 * seed + r); another seed gives an "
        "independent replicate of a checkpoint's metrics",
    )
    args = parser.parse_args()
    # MdmMotionGenerator chdirs into the MDM submodule, so resolve paths first.
    out_dir = args.out_dir.resolve()
    models = {
        name: Path(path).resolve()
        for name, path in (m.split("=", 1) for m in args.model)
    }
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    test = load_test_set(args.limit)
    evaluator = T2MEvaluator(torch.device("cuda"))
    text_emb = evaluator.text_embeddings(test)
    gt_emb = evaluator.motion_embeddings(test.motions, test.lengths)
    mm_idx = np.random.default_rng(1).permutation(len(test.captions))[:_MM_PROMPTS]
    rows = {
        "ground truth": prior_metrics(
            test, text_emb, gt_emb, gt_emb, *body_usage(test.motions, test.lengths)
        )
    }
    showcase = {}
    for name, path in models.items():
        cache = out_dir / "cache" / f"{name}.npz"
        if not cache.exists():
            gen = MdmMotionGenerator(model_path=path)
            prompts = [args.prompt_prefix + c for c in test.captions]
            raw = gen.sample_text_to_motion(
                prompts, test.lengths, seed=args.seed, null_text=args.null_prompt
            )
            mm_captions = [prompts[i] for i in mm_idx]
            mm_emb = np.stack(
                [
                    evaluator.motion_embeddings(
                        gen.sample_text_to_motion(
                            mm_captions,
                            test.lengths[mm_idx],
                            seed=100 * args.seed + r,
                            null_text=args.null_prompt,
                        ),
                        test.lengths[mm_idx],
                    )
                    for r in range(_MM_REPEATS)
                ],
                axis=1,
            )
            pelvis_y, travel = body_usage(raw, test.lengths)
            np.savez(
                cache,
                emb=evaluator.motion_embeddings(raw, test.lengths),
                mm_emb=mm_emb,
                pelvis_y=pelvis_y,
                travel=travel,
                showcase=gen.sample_text_to_motion(
                    [args.prompt_prefix + c for c in SHOWCASE],
                    np.full(len(SHOWCASE), _SHOWCASE_FRAMES),
                    seed=7,
                    null_text=args.null_prompt,
                ),
            )
            del gen
            torch.cuda.empty_cache()
        cached = np.load(cache)
        rows[name] = prior_metrics(
            test,
            text_emb,
            gt_emb,
            cached["emb"],
            cached["pelvis_y"],
            cached["travel"],
            cached["mm_emb"],
        )
        showcase[name] = cached["showcase"]

    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    columns = list(rows[next(iter(models))])
    print(f"{len(test.captions)} test motions\n| model | " + " | ".join(columns) + " |")
    for name, row in rows.items():
        print(
            f"| {name} | "
            + " | ".join(f"{row[c]:.3f}" if c in row else "—" for c in columns)
            + " |"
        )
    if args.showcase:
        render_showcase(showcase, _SHOWCASE_FRAMES, out_dir / "showcase.png")


if __name__ == "__main__":
    main()
