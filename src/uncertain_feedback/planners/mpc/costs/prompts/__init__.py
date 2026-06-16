"""LLM cost-generator prompts, assembled from text templates.

Prompts live as plain ``.txt`` files next to this module so they are easy to read,
diff, and add to:

- ``runtime_api.txt`` and ``output_contract.txt`` are the shared technical contract
  appended to every prompt.
- ``templates/<name>.txt`` is one *head* (framing + guidelines) per selectable
  prompt. Each is auto-registered in :data:`PROMPTS` under ``<name>`` (the filename
  stem). Add a prompt by dropping a new ``templates/<name>.txt`` — no code change.

A head must contain the literal placeholders ``{instruction}`` and
``{image_section}``; the assembled template also contains ``{summaries}``. All
three are substituted by :func:`build_llm_cost_prompt` via ``str.replace`` (so
other braces in the guideline text are left untouched).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent
_TEMPLATES_DIR = _DIR / "templates"


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
    image_paths: list[Path],
    prompt: str = "default",
) -> str:
    """Build the text prompt for the cost-generator LLM.

    Args:
        instruction: The user/caregiver correction text.
        summaries: JSON-serializable trajectory summaries.
        image_paths: Rendered overlay images attached to the request, if any.
        prompt: Name of the registered template in :data:`PROMPTS`.
    """
    try:
        template = PROMPTS[prompt]
    except KeyError as exc:
        raise ValueError(
            f"Unknown llm_cost prompt {prompt!r}; available: {sorted(PROMPTS)}"
        ) from exc
    image_section = "An image of the trajectory is attached." if image_paths else ""
    return (
        template.replace("{instruction}", instruction)
        .replace("{image_section}", image_section)
        .replace("{summaries}", json.dumps(summaries, indent=2, sort_keys=True))
    )
