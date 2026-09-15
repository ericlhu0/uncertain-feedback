"""Give a captioned clip set a second-person register alongside its first-person one.

    uv run python \\
        src/uncertain_feedback/data_collection/dataset_auto_correction/augment_captions.py \\
        --clips_dir <clip set>

``autolabel.py`` drafts in the care recipient's voice ("Lower my hand"), so every
line of a set captioned that way is first-person, while a quarter of the
utterances the deployed model is asked to ground are second-person ("Lower your
hand"). CLIP puts those further apart than it puts *raise* from *lower*, so the
register alone can lose the correction. Each line's pronouns are swapped and the
rewrite appended next to the original, which trains both registers onto one
motion without generating or re-captioning a single clip.

Text-only and in place, like ``autolabel.py`` — point it at a copy if the set's
captions matter. Re-running is a no-op: a rewrite already present is not appended
again.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from uncertain_feedback.data_collection.common.paths import DEFAULT_CLIP_SET

_SECOND_PERSON = {
    "my": "your",
    "me": "you",
    "mine": "yours",
    "i": "you",
    "myself": "yourself",
}
_PRONOUN = re.compile(r"\b(" + "|".join(_SECOND_PERSON) + r")\b", re.IGNORECASE)


def second_person(caption: str) -> str:
    """*caption* with its first-person pronouns rewritten as second-person."""

    def swap(match: re.Match[str]) -> str:
        word = _SECOND_PERSON[match.group().lower()]
        return word.capitalize() if match.start() == 0 else word

    return _PRONOUN.sub(swap, caption)


def augment_clip_set(clips_dir: Path) -> None:
    """Append a second-person rewrite of every caption line to the manifest."""
    manifest_path = clips_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    added = 0
    for row in manifest["runs"]:
        captions: list[str] = row.get("captions", [])
        rewrites = [second_person(caption) for caption in captions]
        new = [line for line in rewrites if line not in captions]
        row["captions"] = captions + new
        added += len(new)
        print(f"{row['run_id']}: {len(captions)} -> {len(row['captions'])} lines")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{added} second-person lines added across {len(manifest['runs'])} runs")


def main() -> None:
    """Parse arguments and add the second register to the clip set's captions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips_dir",
        default=str(DEFAULT_CLIP_SET),
        help=f"Clip set to augment in place (default: {DEFAULT_CLIP_SET}).",
    )
    args = parser.parse_args()
    augment_clip_set(Path(args.clips_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
