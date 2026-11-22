"""Claude wrapper that runs the ``claude`` CLI, so calls bill a Claude subscription.

Log the CLI in once (``claude``, then ``/login``). Each call is a headless
``claude -p`` with no tools, run in an empty directory under ``--safe-mode``:
from the repo, the project's CLAUDE.md and the user's memory would otherwise
reach the prompt.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Union

from .base_model import BaseModel

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class ClaudeModel(BaseModel):
    """``claude -p`` wrapper supporting text and image inputs."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Initialize the Claude model.

        Args:
            model: Claude model name (e.g. "claude-opus-5-5").
            system_prompt: Replaces Claude Code's own system prompt.
            reasoning_effort: ``claude --effort`` level (low, medium, high,
                xhigh, max); the CLI's default when ``None``.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort

    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        return self._run(self._content(text_input, image_input))

    def converse(self, messages: List[Dict[str, Any]]) -> str:
        """Reply to a conversation in ``OpenAIModel.converse``'s message format.

        The CLI cannot be handed earlier assistant turns, so the whole history goes
        in one user turn, each message labelled with its role and followed by its
        images.
        """
        content: List[Dict[str, Any]] = []
        for msg in messages:
            images = msg.get("images") if msg["role"] == "user" else None
            content += self._content(f"[{msg['role']}]\n{msg['text']}", images)
        return self._run(content)

    @staticmethod
    def _content(
        text: str, image_input: Optional[Union[str, List[str]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(image_input, str):
            image_input = [image_input]
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for path in image_input or []:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            media_type = _MEDIA_TYPES[os.path.splitext(path)[1].lower()]
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
        return content

    def _run(self, content: List[Dict[str, Any]]) -> str:
        command = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--system-prompt",
            self.system_prompt,
            "--tools",
            "",
            "--safe-mode",
            "--no-session-persistence",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if self.reasoning_effort is not None:
            command += ["--effort", self.reasoning_effort]
        message = {"type": "user", "message": {"role": "user", "content": content}}
        with tempfile.TemporaryDirectory(prefix="claude-call-") as cwd:
            completed = subprocess.run(
                command,
                input=json.dumps(message) + "\n",
                capture_output=True,
                text=True,
                cwd=cwd,
                check=False,
            )
        lines = completed.stdout.splitlines()
        result = json.loads(lines[-1]) if lines else {}
        if completed.returncode != 0 or result.get("is_error", True):
            raise RuntimeError(
                f"claude -p failed (exit {completed.returncode}): "
                f"{result.get('result') or completed.stderr.strip()}"
            )
        return str(result["result"])
