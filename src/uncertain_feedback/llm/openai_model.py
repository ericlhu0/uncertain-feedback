"""OpenAI model wrapper."""

import base64
import os
from typing import Any, Dict, List, Literal, Optional, Union, cast

from openai import OpenAI

from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API wrapper supporting text and image inputs."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        stream_reasoning_summary: bool = False,
        api_mode: Literal["auto", "chat", "responses"] = "auto",
    ):
        """Initialize OpenAI model.

        Reads OPENAI_API_KEY and optionally OPENAI_ORG_ID from the environment.

        Args:
            model: OpenAI model name (e.g. "gpt-4.1-nano").
            system_prompt: System prompt prepended to every request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            reasoning_effort: Optional reasoning effort for the Responses API.
            stream_reasoning_summary: Print streamed reasoning-summary text.
            api_mode: API surface for full-output requests. ``auto`` uses the
                Responses API for GPT-5-family models and Chat Completions for
                older models.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.stream_reasoning_summary = stream_reasoning_summary
        self.system_prompt = system_prompt
        self.api_mode = api_mode
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
        )

    def encode_image(self, image_path: str) -> str:
        """Base64-encode an image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _create_prompt(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(image_input, str):
            image_input = [image_input]
        if image_input is None:
            image_input = []
        return [{"type": "text", "text": text_input}] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self._image_mime_type(img)};base64,{self.encode_image(img)}"
                },
            }
            for img in image_input
        ]

    def _create_responses_input(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> list[dict[str, Any]]:
        """Build the Responses API message payload."""
        if isinstance(image_input, str):
            image_input = [image_input]
        if image_input is None:
            image_input = []
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text_input}]
        content.extend(
            {
                "type": "input_image",
                "image_url": (
                    f"data:{self._image_mime_type(img)};base64,{self.encode_image(img)}"
                ),
            }
            for img in image_input
        )
        return [{"role": "user", "content": content}]

    def _image_mime_type(self, image_path: str) -> str:
        """Return a data-URL MIME type from the image suffix."""
        suffix = os.path.splitext(image_path)[1].lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        return "application/octet-stream"

    def _chat_token_limit_name(self) -> str:
        """Return the Chat Completions token-limit parameter for this model."""
        if self._is_gpt5_family():
            return "max_completion_tokens"
        return "max_tokens"

    def _use_responses_api(self) -> bool:
        if self.api_mode == "responses":
            return True
        if self.api_mode == "chat":
            return False
        return self._is_gpt5_family()

    def _is_gpt5_family(self) -> bool:
        return self.model.startswith("gpt-5")

    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        if self._use_responses_api():
            return self._get_responses_output(text_input, image_input)
        return self._get_chat_output(text_input, image_input)

    def converse(self, messages: List[Dict[str, Any]]) -> str:
        """Send a multi-turn conversation and return the assistant reply.

        Each message is ``{"role": "user"|"assistant", "text": str,
        "images": Optional[List[str]]}``. Image paths are only honored on ``user``
        turns. The system prompt is prepended automatically. State is held by the
        caller (this method is stateless); pass the growing message list each turn.
        """
        if self._use_responses_api():
            return self._converse_responses(messages)
        return self._converse_chat(messages)

    def _converse_chat(self, messages: List[Dict[str, Any]]) -> str:
        api_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in messages:
            if msg["role"] == "user":
                content: Any = self._create_prompt(msg["text"], msg.get("images"))
            else:
                content = msg["text"]
            api_messages.append({"role": msg["role"], "content": content})
        request: dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, api_messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request[self._chat_token_limit_name()] = self.max_tokens
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned None content")
        return content

    def _converse_responses(self, messages: List[Dict[str, Any]]) -> str:
        api_input: List[Dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "user":
                content = self._create_responses_input(msg["text"], msg.get("images"))[
                    0
                ]["content"]
            else:
                content = [{"type": "output_text", "text": msg["text"]}]
            api_input.append({"role": msg["role"], "content": content})
        request = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": cast(Any, api_input),
        }
        if self.reasoning_effort is None:
            request["temperature"] = self.temperature
        reasoning = self._responses_reasoning()
        if reasoning:
            request["reasoning"] = reasoning
        if self.max_tokens is not None:
            request["max_output_tokens"] = self.max_tokens
        response = self._create_response(request)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        raise ValueError("OpenAI Responses API returned no output_text")

    def _get_responses_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": self._create_responses_input(text_input, image_input),
        }
        if self.reasoning_effort is None:
            request["temperature"] = self.temperature
        reasoning = self._responses_reasoning()
        if reasoning:
            request["reasoning"] = reasoning
        if self.max_tokens is not None:
            request["max_output_tokens"] = self.max_tokens
        response = self._create_response(request)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        raise ValueError("OpenAI Responses API returned no output_text")

    def _responses_reasoning(self) -> dict[str, str]:
        reasoning: dict[str, str] = {}
        if self.reasoning_effort is not None:
            reasoning["effort"] = self.reasoning_effort
        if self.stream_reasoning_summary:
            reasoning["summary"] = "auto"
        return reasoning

    def _create_response(self, request: dict[str, Any]) -> Any:
        if not self.stream_reasoning_summary:
            return self.client.responses.create(**request)

        stream = self.client.responses.create(**request, stream=True)
        response = None
        summary_started = False
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.reasoning_summary_text.delta":
                if not summary_started:
                    print("[cost-gen][llm] reasoning summary:", flush=True)
                    summary_started = True
                print(event.delta, end="", flush=True)
            elif event_type == "response.completed":
                response = event.response
        if summary_started:
            print(flush=True)
        if response is None:
            raise ValueError("OpenAI Responses API stream ended before completion")
        return response

    def _get_chat_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        user_prompt = self._create_prompt(text_input, image_input)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cast(Any, user_prompt)},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request[self._chat_token_limit_name()] = self.max_tokens
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned None content")
        return content
