"""LLM tempo captions for a correction dataset built with ``--retimes``.

Clips move at different speeds — the planner's pace varies by clip set, and
``build_dataset.py --retimes`` replays clips at other rates — but their lines
describe only the path, which would teach the model that the same words mean
any speed. The builder records every motion's pace against the set's median clip
speed in ``paces.json``; this step gives a text-only LLM each motion's lines and
that pace and asks it to rewrite every line so it also says how fast, worded
however a person would say it mid-motion, with no speed vocabulary supplied.
Paces close to usual come back unchanged, so a request that says nothing about
speed keeps meaning the usual pace. Every line is gated with the direction check
:mod:`paraphrase_captions` uses.

Writes a new dataset directory whose motions, stats and splits are the source's
(motions symlinked) and whose texts are replaced. Completions are cached in
``<out>/tempo.json`` per line-set and pace — a transplant shares its clip's — so a
rerun is free.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import spacy

from uncertain_feedback.data_collection.common.dataset import write_text_file
from uncertain_feedback.data_collection.dataset_auto_correction.captioning import (
    draft_lines,
)
from uncertain_feedback.data_collection.dataset_auto_correction.paraphrase_captions import (
    consistent_paraphrases,
    source_lines,
)
from uncertain_feedback.llm.openai_model import OpenAIModel

_LOG = "[tempo]"

# Each line is rewritten in place rather than drafted afresh, so retimed text
# keeps the length, register and person of the text it sits beside and only the
# tempo tells the two apart.
TEMPO_PROMPT = (
    "You are role-playing a care recipient whose left arm a caregiver robot is "
    "moving. The robot is moving your arm one way and you want it moved "
    "differently. Each line below is something you might say to ask for that "
    "change (they all describe the same single motion):\n\n{lines}\n\n"
    "You want the change made at {pace:.2f} times the speed the caregiver "
    "usually moves your arm (1.00 is its usual speed). Rewrite each line, in "
    "order, so it also says how fast you want it - briefly, the way you would "
    "actually say it in the moment while your arm is being moved, wherever in "
    "the line it comes naturally or as its own short phrase before or after the "
    "request, in the everyday words of a person who can only feel the motion, "
    "not measure it. "
    "Vary how you say the speed from line to line. Keep the rest of each line's "
    "wording, register and person (my/your). Say the speed you want, not the "
    "one you don't. Never quantify it: no numbers, fractions or multiples such "
    "as 'twice', 'half' or 'three times'. If that speed is close to usual, "
    "repeat the line unchanged. Write one line per input line and nothing else "
    "- no numbering, bullets or quotes."
)


def tempo_line_set(model: OpenAIModel, sources: list[str], pace: float) -> list[str]:
    """One completion's rewrite of each line of *sources* at *pace*, unfiltered."""
    prompt = TEMPO_PROMPT.format(lines="\n".join(sources), pace=pace)
    for attempt in range(3):
        try:
            return draft_lines(model.get_full_output(prompt), len(sources))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"{_LOG} attempt {attempt + 1} failed: {exc}", flush=True)
    return []


def tempo_dataset(src: Path, out: Path, model_name: str, workers: int) -> None:
    """Build *out* from *src* with every motion's text rewritten at its pace."""
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "tempo.json"
    cache: dict[str, list[str]] = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    paces: dict[str, float] = json.loads(
        (src / "paces.json").read_text(encoding="utf-8")
    )
    sources = {i: source_lines(src / "texts" / f"{i}.txt") for i in paces}
    keys = {i: f"{paces[i]:.2f}|" + "\n".join(sorted(sources[i])) for i in paces}
    todo = {keys[i]: i for i in paces if keys[i] not in cache}
    print(f"{_LOG} {len(paces)} motions, {len(todo)} line-sets to draft")
    model = OpenAIModel(
        model=model_name,
        system_prompt="You answer with short spoken sentences, one per line, and nothing else.",
    )

    def draft(key: str) -> tuple[str, list[str]]:
        motion_id = todo[key]
        return key, tempo_line_set(model, sources[motion_id], paces[motion_id])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (key, raw) in enumerate(pool.map(draft, todo), 1):
            if raw:
                cache[key] = raw
            if n % 25 == 0 or n == len(todo):
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                print(f"{_LOG} drafted {n}/{len(todo)}", flush=True)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    nlp = spacy.load("en_core_web_sm")
    (out / "texts").mkdir(exist_ok=True)
    link = out / "new_joint_vecs"
    if not link.exists():
        link.symlink_to((src / "new_joint_vecs").resolve())
    for name in (
        "Mean.npy",
        "Std.npy",
        "train.txt",
        "val.txt",
        "test.txt",
        "paces.json",
    ):
        shutil.copy(src / name, out / name)
    n_rewritten = 0
    for i in paces:
        # Near-usual paces come back unchanged and are dropped as copies of the
        # sources, leaving the path-only text, which is still true of the motion.
        kept = consistent_paraphrases(cache.get(keys[i], []), sources[i])
        n_rewritten += int(bool(kept))
        if kept:
            write_text_file(out / "texts" / f"{i}.txt", kept, nlp)
        else:
            # Copied, not rewritten: keeps any lines the builder duplicated.
            shutil.copy(src / "texts" / f"{i}.txt", out / "texts" / f"{i}.txt")
    print(
        f"{_LOG} {n_rewritten} of {len(paces)} motions carry tempo text; "
        f"the rest kept their path-only lines; wrote {out}"
    )


def main() -> None:
    """Write tempo captions for a built dataset from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--src", type=Path, required=True, help="dataset built with --retimes"
    )
    parser.add_argument("--out", type=Path, required=True, help="new dataset dir")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    tempo_dataset(args.src, args.out, args.model, args.workers)


if __name__ == "__main__":
    main()
