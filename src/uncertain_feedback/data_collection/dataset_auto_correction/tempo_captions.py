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

``--keep_plain`` writes each motion's path-only lines beside its rewrites
instead of replacing them, so a request that says nothing about speed is also
seen at every pace. ``--usual_band LOW HIGH`` keeps the lines of motions whose
pace lies inside the band without a call, so what counts as usual is a rule of
ours rather than the LLM's judgement. ``--budget`` meters the API spend from the usage the API
reports (:mod:`uncertain_feedback.llm.spend`, ledger ``<out>/api_cost.jsonl``)
and stops drafting once it is spent; the cache keeps what was drafted and the
run fails rather than write a dataset with motions still undrafted.
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
from uncertain_feedback.llm.spend import SpendLedger

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


def raw_lines(text_file: Path) -> list[str]:
    """The captions of one MDM text file as written, duplicates included."""
    return [
        line.split("#")[0].strip()
        for line in text_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tempo_lines(plain: list[str], rewrites: list[str], keep_plain: bool) -> list[str]:
    """The lines a motion trains on: its rewrites, after its plain lines with ``keep_plain``."""
    return plain + rewrites if keep_plain else rewrites


def near_usual(paces: dict[str, float], band: tuple[float, float] | None) -> set[str]:
    """The motions whose pace lies strictly inside ``band``: they keep their path-only lines
    without a call, so what counts as usual is this rule's, not the LLM's."""
    if band is None:
        return set()
    return {i for i, pace in paces.items() if band[0] < pace < band[1]}


def tempo_dataset(  # pylint: disable=too-many-locals,too-many-arguments
    src: Path,
    out: Path,
    model_name: str,
    workers: int,
    keep_plain: bool = False,
    budget: float | None = None,
    usual_band: tuple[float, float] | None = None,
) -> None:
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
    usual = near_usual(paces, usual_band)
    sources = {i: source_lines(src / "texts" / f"{i}.txt") for i in paces}
    keys = {i: f"{paces[i]:.2f}|" + "\n".join(sorted(sources[i])) for i in paces}
    todo = {keys[i]: i for i in paces if i not in usual and keys[i] not in cache}
    print(
        f"{_LOG} {len(paces)} motions, {len(usual)} at a usual pace {usual_band} kept as "
        f"written, {len(todo)} line-sets to draft"
    )
    model = OpenAIModel(
        model=model_name,
        system_prompt="You answer with short spoken sentences, one per line, and nothing else.",
    )
    ledger = None if budget is None else SpendLedger(out / "api_cost.jsonl", budget)
    if ledger is not None:
        ledger.meter(model)
        print(f"{_LOG} metered: ${ledger.total:.4f} of ${budget:.2f} already spent")

    def draft(key: str) -> tuple[str, list[str]]:
        if ledger is not None and ledger.exhausted:
            return key, []
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
    if ledger is not None:
        print(f"{_LOG} ${ledger.total:.4f} spent over {len(ledger.costs)} calls")
    undrafted = [key for key in todo if key not in cache]
    if undrafted:
        raise RuntimeError(
            f"{len(undrafted)} line-sets not drafted"
            + (" (budget spent)" if ledger is not None and ledger.exhausted else "")
            + "; the cache keeps the rest, rerun to continue"
        )

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
        kept = (
            []
            if i in usual
            else consistent_paraphrases(cache.get(keys[i], []), sources[i])
        )
        n_rewritten += int(bool(kept))
        if kept:
            write_text_file(
                out / "texts" / f"{i}.txt",
                tempo_lines(raw_lines(src / "texts" / f"{i}.txt"), kept, keep_plain),
                nlp,
            )
        else:
            # Copied, not rewritten: keeps any lines the builder duplicated.
            shutil.copy(src / "texts" / f"{i}.txt", out / "texts" / f"{i}.txt")
    print(
        f"{_LOG} {n_rewritten} of {len(paces)} motions carry tempo text"
        + (" beside their plain lines" if keep_plain else "")
        + f"; the rest kept their path-only lines; wrote {out}"
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
    parser.add_argument(
        "--keep_plain",
        action="store_true",
        help="write each motion's path-only lines beside its rewrites",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="dollars; drafting stops once the metered spend reaches it",
    )
    parser.add_argument(
        "--usual_band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="motions whose pace lies strictly inside this band keep their lines "
        "without a call (default: the LLM decides what is close to usual)",
    )
    args = parser.parse_args()
    tempo_dataset(
        args.src,
        args.out,
        args.model,
        args.workers,
        args.keep_plain,
        args.budget,
        None if args.usual_band is None else (args.usual_band[0], args.usual_band[1]),
    )


if __name__ == "__main__":
    main()
