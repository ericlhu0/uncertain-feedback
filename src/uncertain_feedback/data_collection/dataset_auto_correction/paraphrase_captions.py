"""LLM paraphrase augmentation of a built correction dataset's captions.

Templated captions are true of their motion by construction but come from a
~170-line vocabulary, so a model trained on them meets few of the ways people
actually ask for a change. This step keeps the templated lines as the ground
truth and asks a text-only LLM, per distinct line-set, for many more things a
care recipient might say to request the same change — varied register, verbs,
abstraction, complaints as well as requests. Every paraphrase is gated with the
same direction-word check the builder uses (:func:`motion_facts.asserted`):
a line may not claim a direction the source lines do not state.

Writes a new dataset directory whose ``new_joint_vecs``, stats and splits are
the source's (motions symlinked) and whose texts are source + paraphrases, so a
fine-tune on it isolates the effect of language coverage alone. Completions are
cached in ``<out>/paraphrases.json`` per line-set, so a rerun is free.
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
from uncertain_feedback.data_collection.dataset_auto_correction.motion_facts import (
    asserted,
)
from uncertain_feedback.llm.openai_model import OpenAIModel

_LOG = "[paraphrase]"

PARAPHRASE_PROMPT = (
    "You are role-playing a care recipient whose left arm a caregiver robot is "
    "moving. The robot is moving your arm one way and you want it moved "
    "differently. Every line below is a true, literal description of the change "
    "you want (they all describe the same single motion):\n\n{lines}\n\n"
    "Write exactly {n} different things you might actually say to the caregiver "
    "to ask for that change, one per line and nothing else - no numbering, "
    "bullets or quotes. Vary them widely: some name the body part and direction "
    "plainly, some describe the outcome or feeling loosely (\"that's too high\", "
    "\"not so far out\"), some are complaints, some polite requests, some terse, "
    "some combine two of the changes, some mention only the biggest change; use "
    "both 'my' and 'your' phrasings. Never ask for a direction the descriptions "
    "do not state, and never mention the robot's arm or the word 'trajectory'."
)


def source_lines(text_file: Path) -> list[str]:
    """The distinct captions of one MDM text file, in file order."""
    seen: dict[str, None] = {}
    for line in text_file.read_text(encoding="utf-8").splitlines():
        caption = line.split("#")[0].strip()
        if caption:
            seen.setdefault(caption, None)
    return list(seen)


def consistent_paraphrases(paraphrases: list[str], sources: list[str]) -> list[str]:
    """The paraphrases whose every direction claim is one the sources make."""
    claims: dict[str, int] = {}
    for line in sources:
        claims.update(asserted(line))
    return [
        p
        for p in paraphrases
        if all(claims.get(q) == sign for q, sign in asserted(p).items())
        and p not in sources
    ]


def paraphrase_line_set(model: OpenAIModel, sources: list[str], n: int) -> list[str]:
    """One completion's ``n`` paraphrases of *sources*, unfiltered."""
    prompt = PARAPHRASE_PROMPT.format(lines="\n".join(sources), n=n)
    for attempt in range(3):
        try:
            return draft_lines(model.get_full_output(prompt), n)
        except Exception as exc:  # noqa: BLE001 - one empty completion must not sink the run
            print(f"{_LOG} attempt {attempt + 1} failed: {exc}", flush=True)
    return []


def paraphrase_dataset(
    src: Path, out: Path, model_name: str, n: int, workers: int
) -> None:
    """Build *out* from *src* with every motion's text extended by LLM paraphrases."""
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "paraphrases.json"
    cache: dict[str, list[str]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    files = sorted((src / "texts").glob("*.txt"))
    per_file = {f: source_lines(f) for f in files}
    line_sets = {"\n".join(sorted(v)): v for v in per_file.values()}
    todo = [k for k in line_sets if k not in cache]
    print(f"{_LOG} {len(files)} motions, {len(line_sets)} line-sets, {len(todo)} to draft")
    model = OpenAIModel(
        model=model_name,
        system_prompt="You answer with short spoken sentences, one per line, and nothing else.",
    )

    def draft(key: str) -> tuple[str, list[str]]:
        return key, paraphrase_line_set(model, line_sets[key], n)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (key, raw) in enumerate(pool.map(draft, todo), 1):
            if raw:
                cache[key] = raw
            if i % 25 == 0 or i == len(todo):
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                print(f"{_LOG} drafted {i}/{len(todo)}", flush=True)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    nlp = spacy.load("en_core_web_sm")
    (out / "texts").mkdir(exist_ok=True)
    link = out / "new_joint_vecs"
    if not link.exists():
        link.symlink_to((src / "new_joint_vecs").resolve())
    for name in ("Mean.npy", "Std.npy", "train.txt", "val.txt", "test.txt"):
        shutil.copy(src / name, out / name)
    n_raw = n_kept = 0
    unique: set[str] = set()
    for f, sources in per_file.items():
        raw = cache.get("\n".join(sorted(sources)), [])
        kept = consistent_paraphrases(raw, sources)
        n_raw += len(raw)
        n_kept += len(kept)
        unique.update(sources)
        unique.update(kept)
        write_text_file(out / "texts" / f.name, sources + kept, nlp)
    print(
        f"{_LOG} kept {n_kept}/{n_raw} paraphrase lines "
        f"({n_kept / max(n_raw, 1):.0%}); {len(unique)} unique lines in {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="built dataset dir")
    parser.add_argument("--out", type=Path, required=True, help="new dataset dir")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--n", type=int, default=12, help="paraphrases per line-set")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    paraphrase_dataset(args.src, args.out, args.model, args.n, args.workers)


if __name__ == "__main__":
    main()
