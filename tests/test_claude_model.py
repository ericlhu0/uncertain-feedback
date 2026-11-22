"""Tests for the ``claude`` CLI model wrapper."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from uncertain_feedback.llm import ClaudeModel, OpenAIModel, make_model


def _fake_claude(monkeypatch, result: dict[str, Any], returncode: int = 0):
    calls: list[dict[str, Any]] = []

    def run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(
            returncode=returncode,
            stdout=f'{{"type": "system"}}\n{json.dumps(result)}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_full_output_sends_text_and_images_without_tools(tmp_path, monkeypatch):
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    calls = _fake_claude(monkeypatch, {"is_error": False, "result": "reply"})
    model = ClaudeModel("claude-opus-5-5", "system", reasoning_effort="high")

    assert model.get_full_output("make json", str(image)) == "reply"

    command = calls[0]["command"]
    assert command[command.index("--model") + 1] == "claude-opus-5-5"
    assert command[command.index("--system-prompt") + 1] == "system"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--effort") + 1] == "high"
    assert "--safe-mode" in command
    content = json.loads(calls[0]["input"])["message"]["content"]
    assert content[0] == {"type": "text", "text": "make json"}
    assert content[1]["source"]["media_type"] == "image/png"


def test_converse_folds_history_into_one_turn(monkeypatch):
    calls = _fake_claude(monkeypatch, {"is_error": False, "result": "second"})
    model = ClaudeModel("claude-opus-5-5", "system")

    model.converse(
        [
            {"role": "user", "text": "first prompt"},
            {"role": "assistant", "text": "first reply"},
            {"role": "user", "text": "feedback"},
        ]
    )

    content = json.loads(calls[0]["input"])["message"]["content"]
    assert [block["text"] for block in content] == [
        "[user]\nfirst prompt",
        "[assistant]\nfirst reply",
        "[user]\nfeedback",
    ]


def test_cli_error_raises(monkeypatch):
    _fake_claude(monkeypatch, {"is_error": True, "result": "Not logged in"}, 1)

    with pytest.raises(RuntimeError, match="Not logged in"):
        ClaudeModel("claude-opus-5-5", "system").get_full_output("hi")


def test_make_model_routes_claude_names_to_the_cli(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    assert isinstance(make_model("claude-opus-5-5", "system"), ClaudeModel)
    assert isinstance(make_model("gpt-6-luna", "system"), OpenAIModel)
