"""Caption a whole correction-clip set with the VLM, with no browser and no human.

    uv run python \
        src/uncertain_feedback/data_collection/dataset_auto_correction/autolabel.py \
        [--clips_dir <clip set>] [--n_captions 5] [--prompt_from <dir>]

The headless twin of ``label.py``'s *Draft caption* button: for every run whose
``captions`` list is still empty it renders the same window image and asks
``llm_cost.model`` for ``--n_captions`` phrasings in one completion, then writes
them into the run's ``captions`` in ``manifest.json``. Runs that already have
captions are skipped and the manifest is saved after each run, so the job is
resumable and can be interrupted freely — and a hand-labeled set can be topped up
without touching its existing captions.

The prompt is the clip set's own saved ``caption_prompt`` (as written by the
labeling UI's *Draft caption prompt* panel), falling back to the stock
:data:`~...captioning.DRAFT_PROMPT`. Pass ``--prompt_from`` a clip set or session
directory to reuse the wording settled on there instead; whichever is used is
recorded in this set's manifest.

Needs ``OPENAI_API_KEY``. Writes into the directory it is given, so point it at a
session directory (or a copy) if the base set's captions matter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uncertain_feedback.data_collection.common.paths import DEFAULT_CLIP_SET
from uncertain_feedback.data_collection.dataset_auto_correction.captioning import (
    MAX_DRAFTS,
    autolabel_clip_set,
    caption_prompt_from,
)


def main() -> None:
    """Parse arguments and caption every uncaptioned run in the clip set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips_dir",
        default=str(DEFAULT_CLIP_SET),
        help=f"Clip set to caption in place (default: {DEFAULT_CLIP_SET}).",
    )
    parser.add_argument(
        "--n_captions",
        type=int,
        default=MAX_DRAFTS,
        help=f"Captions per run, from one completion (default: {MAX_DRAFTS}).",
    )
    parser.add_argument(
        "--prompt_from",
        default=None,
        help=(
            "Clip set or session directory whose saved caption_prompt to use, "
            "instead of --clips_dir's own."
        ),
    )
    args = parser.parse_args()

    # Absolute before anything loads, matching generate.py: the MDM loader
    # chdir()s into its submodule and never restores.
    clips_dir = Path(args.clips_dir).expanduser().resolve()
    prompt_dir = (
        Path(args.prompt_from).expanduser().resolve() if args.prompt_from else clips_dir
    )
    autolabel_clip_set(clips_dir, args.n_captions, caption_prompt_from(prompt_dir))


if __name__ == "__main__":
    main()
