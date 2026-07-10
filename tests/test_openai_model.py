from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from uncertain_feedback.llm.openai_model import OpenAIModel


class _FakeResponses:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="response text")


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="chat text"),
                )
            ]
        )


def _model(model_name: str, tmp_client, **kwargs) -> OpenAIModel:
    model = OpenAIModel.__new__(OpenAIModel)
    model.model = model_name
    model.temperature = kwargs.get("temperature", 0.2)
    model.max_tokens = kwargs.get("max_tokens", 123)
    model.reasoning_effort = kwargs.get("reasoning_effort")
    model.system_prompt = "system"
    model.api_mode = kwargs.get("api_mode", "auto")
    model.client = tmp_client
    return model


def test_gpt5_full_output_uses_responses_api_with_max_output_tokens(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"fake-png")
    responses = _FakeResponses()
    model = _model("gpt-5.4", SimpleNamespace(responses=responses))

    output = model.get_full_output("make json", image_input=str(image_path))

    assert output == "response text"
    assert responses.request is not None
    assert responses.request["model"] == "gpt-5.4"
    assert responses.request["instructions"] == "system"
    assert responses.request["max_output_tokens"] == 123
    content = responses.request["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "make json"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_responses_api_includes_reasoning_effort() -> None:
    responses = _FakeResponses()
    model = _model(
        "gpt-5.6-luna",
        SimpleNamespace(responses=responses),
        reasoning_effort="xhigh",
    )

    model.get_full_output("make json")

    assert responses.request is not None
    assert responses.request["reasoning"] == {"effort": "xhigh"}
    assert "temperature" not in responses.request


def test_chat_mode_preserves_legacy_chat_completion_shape() -> None:
    chat = _FakeChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    model = _model("gpt-4.1", client)

    output = model.get_full_output("hello")

    assert output == "chat text"
    assert chat.request is not None
    assert chat.request["model"] == "gpt-4.1"
    assert chat.request["max_tokens"] == 123
    assert "max_completion_tokens" not in chat.request
    assert chat.request["messages"][0] == {"role": "system", "content": "system"}
    assert chat.request["messages"][1]["content"] == [{"type": "text", "text": "hello"}]
