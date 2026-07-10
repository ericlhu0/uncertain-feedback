"""LLM cost-generator prompts, assembled from text templates.

Every backend now uses the same *staged* strategy — the correction is decomposed
into focused stages so the model reasons about one thing at a time instead of
interpreting, grounding, and coding all at once. Prompts live as plain ``.txt``
files next to this module so they are easy to read and diff:

- ``runtime_api.txt`` and ``output_contract.txt`` are the shared technical contract
  appended to the code-writing stages.
- ``stages/interpret.txt`` — instruction + contrast images + a compact summary ->
  a plain-language preference (no API, no numbers, no code).
- ``stages/ground.txt`` — that preference + the full numeric summaries -> a concrete
  numeric spec of joint features and bounds (single-shot ``llm`` path).
- ``stages/author.txt`` — that spec + the shared contract -> the cost JSON, faithful
  to the spec (single-shot ``llm`` path).
- ``stages/refine.txt`` — the seed for the iterating ``turns`` backend: the fixed
  interpretation + full summaries, grounded and implemented in one conversation that
  revises both the numbers and the code against rollout feedback each turn.

Each head is substituted via ``str.replace`` (so other braces in the guideline text
are left untouched). A head may contain image placeholders from
:data:`IMAGE_PLACEHOLDERS` (only ``interpret.txt`` does); each present placeholder
whose image was rendered is replaced by a self-describing line and its image attached
in placeholder order, and a placeholder whose image is unavailable is dropped with no
attachment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent

# Image placeholders a template head may request, mapped to the self-describing line
# that replaces the placeholder when that image is attached. The key also matches the
# key returned by ``render_prompt_images`` so a present placeholder selects its image.
#
# Each line is self-contained: it carries its own guidance so it can be dropped
# whole when the image is unavailable, leaving no dangling reference in surrounding
# prose. Keep guidance here (not in the static template text) for that reason.
IMAGE_PLACEHOLDERS: dict[str, str] = {
    "current_cluster_traj_img": (
        "An image showing ONLY the chosen path's full arm motion is attached — use it "
        "to read the chosen path's posture clearly."
    ),
    "other_clusters_traj_img": (
        "An image showing the chosen path's full arm alongside the OTHER candidate "
        "paths' full arms (grey, with grey end markers) is attached — use the grey arms "
        "only to see which dimension separates the chosen path from the candidates it "
        "was picked over."
    ),
    "reference_traj_img": (
        "An image showing the chosen path's full arm alongside the ORIGINAL-GOAL "
        "reference arm (green, with a dashed wrist path) is attached — use the green "
        "reference and the gold star to see where your cost must NOT get in the way of "
        "the original goal."
    ),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# Shared technical contract, identical across every prompt variant.
_RUNTIME_API = _read(_DIR / "runtime_api.txt")
_OUTPUT_CONTRACT = _read(_DIR / "output_contract.txt")

# Stage heads: interpret -> ground -> author for single-shot; interpret -> refine loop
# for the iterating backend. Each is a focused subtask.
_STAGES_DIR = _DIR / "stages"
_INTERPRET_HEAD = _read(_STAGES_DIR / "interpret.txt")
_GROUND_HEAD = _read(_STAGES_DIR / "ground.txt")
_AUTHOR_HEAD = _read(_STAGES_DIR / "author.txt")
_REFINE_HEAD = _read(_STAGES_DIR / "refine.txt")


def _substitute_images(
    template: str, images: dict[str, Path]
) -> tuple[str, list[Path]]:
    """Replace image placeholders in ``template`` and return the attached paths.

    Each placeholder in :data:`IMAGE_PLACEHOLDERS` that appears is replaced by its
    self-describing line when its image was rendered, or dropped otherwise. Paths are
    returned in the order their placeholders appear in ``template``.
    """
    present = [
        (template.index("{" + key + "}"), key)
        for key in IMAGE_PLACEHOLDERS
        if "{" + key + "}" in template
    ]
    ordered_paths: list[Path] = []
    for _, key in sorted(present):
        path = images.get(key)
        replacement = IMAGE_PLACEHOLDERS[key] if path is not None else ""
        template = template.replace("{" + key + "}", replacement)
        if path is not None:
            ordered_paths.append(path)
    return template, ordered_paths


def _dump(summaries: dict[str, Any]) -> str:
    return json.dumps(summaries, indent=2, sort_keys=True)


def compact_summaries(summaries: dict[str, Any]) -> dict[str, Any]:
    """Return a slimmed summary for the interpret stage.

    Keeps only the qualitative anatomy the interpretation needs — per-trajectory
    joint-feature ranges and each joint's endpoint position — and drops the raw
    joint-angle arrays and axis-angle values. The full summaries go to the grounding
    stage; this keeps stage one from doing stage two's numeric work.
    """
    slim: dict[str, Any] = {}
    for name, entry in summaries.items():
        if not isinstance(entry, dict):
            slim[name] = entry
            continue
        kept: dict[str, Any] = {}
        if "joint_features" in entry:
            kept["joint_features"] = entry["joint_features"]
        positions = entry.get("positions")
        if isinstance(positions, dict):
            kept["end_positions"] = {
                joint: stats.get("end")
                for joint, stats in positions.items()
                if isinstance(stats, dict)
            }
        slim[name] = kept or entry
    return slim


def build_interpret_prompt(
    instruction: str,
    summaries: dict[str, Any],
    images: dict[str, Path],
) -> tuple[str, list[Path]]:
    """Build the stage-one (interpret) prompt and its attached images.

    Sees the instruction, the contrast images, and a compact summary only — no
    runtime API, output contract, or numeric grounding. Returns a plain-language
    preference JSON.
    """
    template, ordered_paths = _substitute_images(_INTERPRET_HEAD, images)
    text = template.replace("{instruction}", instruction).replace(
        "{summaries}", _dump(compact_summaries(summaries))
    )
    return text, ordered_paths


def build_ground_prompt(interpretation: str, summaries: dict[str, Any]) -> str:
    """Build the stage-two (ground) prompt: preference + full numbers -> numeric spec.

    Text only — no images, no runtime API, no code contract.
    """
    return _GROUND_HEAD.replace("{interpretation}", interpretation).replace(
        "{summaries}", _dump(summaries)
    )


def build_author_prompt(specification: str) -> str:
    """Build the stage-three (author) prompt: numeric spec + code contract -> cost JSON.

    Appends the shared runtime API and output contract; sees no images or summaries.
    """
    head = _AUTHOR_HEAD.replace("{specification}", specification)
    return "\n\n".join([head, _RUNTIME_API, _OUTPUT_CONTRACT]).strip()


def build_staged_task_body(
    instruction: str,
    summaries: dict[str, Any],
    images: dict[str, Path],
) -> tuple[str, list[Path]]:
    """Compose the three stages into one method document for the autonomous agent.

    Unlike the in-process backends (which run interpret/ground/author as separate
    calls), the ``agent`` backend hands the whole method to codex. This inlines the
    three stage heads — interpret with the instruction, compact summary, and image
    lines; ground referring back to the agent's own stage-1 output; author plus the
    shared runtime API and output contract — so codex follows the same decomposition
    and produces one final author-stage cost JSON. Returns ``(text, image_paths)``.
    """
    interpret_text, image_paths = _substitute_images(_INTERPRET_HEAD, images)
    interpret_text = interpret_text.replace("{instruction}", instruction).replace(
        "{summaries}", _dump(compact_summaries(summaries))
    )
    ground_text = _GROUND_HEAD.replace(
        "{interpretation}", "(the preference you wrote in Stage 1)"
    ).replace("{summaries}", _dump(summaries))
    author_text = _AUTHOR_HEAD.replace(
        "{specification}", "(the numeric specification you wrote in Stage 2)"
    )
    body = "\n\n".join(
        [
            "Work through these three stages in order. Record your Stage 1 and Stage 2 "
            "outputs in your iteration log; write ONLY the final Stage 3 cost JSON to "
            "the response file.",
            "## Stage 1 — interpret",
            interpret_text,
            "## Stage 2 — ground",
            ground_text,
            "## Stage 3 — author",
            author_text,
            _RUNTIME_API,
            _OUTPUT_CONTRACT,
        ]
    ).strip()
    return body, image_paths


def build_refine_prompt(interpretation: str, summaries: dict[str, Any]) -> str:
    """Build the seed prompt for the iterating ``turns`` backend.

    Grounds the fixed interpretation and implements it in one conversation, with the
    shared runtime API and output contract appended; rollout feedback (score, joint
    comparison, images) is fed back as follow-up turns to revise both numbers and code.
    Text only — no contrast images (the interpret stage already consumed them).
    """
    head = _REFINE_HEAD.replace("{interpretation}", interpretation).replace(
        "{summaries}", _dump(summaries)
    )
    return "\n\n".join([head, _RUNTIME_API, _OUTPUT_CONTRACT]).strip()
