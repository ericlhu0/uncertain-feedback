"""LLM cost-generator prompts, assembled from text templates.

Prompts live as plain ``.txt`` files next to this module so they are easy to read,
diff, and add to:

- ``runtime_api.txt`` and ``output_contract.txt`` are the shared technical contract
  appended to every prompt.
- ``templates/<name>.txt`` is one *head* (framing + guidelines) per selectable
  prompt. Each is auto-registered in :data:`PROMPTS` under ``<name>`` (the filename
  stem). Add a prompt by dropping a new ``templates/<name>.txt`` — no code change.

A head must contain the literal placeholder ``{instruction}``; the assembled
template also contains ``{summaries}``. Both are substituted by
:func:`build_llm_cost_prompt` via ``str.replace`` (so other braces in the
guideline text are left untouched).

A head may also contain any of the image placeholders in :data:`IMAGE_PLACEHOLDERS`
(e.g. ``{current_cluster_traj_img}``). Each one that appears — and whose image was
actually rendered — is replaced by a short self-describing line and its image is
attached to the request, in the order the placeholders appear in the head. A
placeholder whose image is unavailable (e.g. ``{reference_traj_img}`` with no
reference) is dropped with no attachment. Templates with no image placeholder send
text only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent
_TEMPLATES_DIR = _DIR / "templates"

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


def _assemble(head: str) -> str:
    """Join a variant head with the shared API + output contract + summaries."""
    return "\n\n".join(
        [head, _RUNTIME_API, _OUTPUT_CONTRACT, "Summaries:\n{summaries}"]
    ).strip()


# Registry of available prompt templates, keyed by the head filename stem. Select
# one via ``llm_cost.prompt`` in the MPC config (defaults to ``"default"``).
PROMPTS: dict[str, str] = {
    path.stem: _assemble(_read(path))
    for path in sorted(_TEMPLATES_DIR.glob("*.txt"))
}


def build_llm_cost_prompt(
    instruction: str,
    summaries: dict[str, Any],
    images: dict[str, Path],
    prompt: str = "2",
) -> tuple[str, list[Path]]:
    """Build the text prompt for the cost-generator LLM and its attached images.

    Which images are attached is driven by the image placeholders present in the
    selected template head (see :data:`IMAGE_PLACEHOLDERS`): each placeholder that
    appears and whose image was rendered is replaced by a self-describing line and
    its path is attached, in placeholder order. A placeholder whose image is missing
    from ``images`` is dropped with no attachment.

    Args:
        instruction: The user/caregiver correction text.
        summaries: JSON-serializable trajectory summaries.
        images: ``{placeholder: path}`` rendered overlay images available to attach.
        prompt: Name of the registered template in :data:`PROMPTS`.

    Returns:
        ``(prompt_text, ordered_paths)`` where ``ordered_paths`` are the images to
        attach to the request, in the order their placeholders appear in the template.
    """
    try:
        template = PROMPTS[prompt]
    except KeyError as exc:
        raise ValueError(
            f"Unknown llm_cost prompt {prompt!r}; available: {sorted(PROMPTS)}"
        ) from exc

    # Attach images in the order their placeholders appear in the template head.
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

    text = (
        template.replace("{instruction}", instruction)
        .replace("{summaries}", json.dumps(summaries, indent=2, sort_keys=True))
    )
    return text, ordered_paths
