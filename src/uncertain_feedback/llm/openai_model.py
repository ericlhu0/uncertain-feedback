"""OpenAI model wrapper."""

import base64
import os
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API wrapper supporting text and image inputs."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        api_mode: Literal["auto", "chat", "responses"] = "auto",
    ):
        """Initialize OpenAI model.

        Reads OPENAI_API_KEY and optionally OPENAI_ORG_ID from the environment.

        Args:
            model: OpenAI model name (e.g. "gpt-4.1-nano").
            system_prompt: System prompt prepended to every request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            api_mode: API surface for full-output requests. ``auto`` uses the
                Responses API for GPT-5-family models and Chat Completions for
                older models.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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

    def _get_chat_completion(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> ChatCompletion:
        user_prompt = self._create_prompt(text_input, image_input)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cast(Any, user_prompt)},
            ],
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 10,
        }
        if self.max_tokens is not None:
            request[self._chat_token_limit_name()] = self.max_tokens
        return self.client.chat.completions.create(**request)

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

    def get_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> Dict[Any, Any]:
        response = self._get_chat_completion(text_input, image_input)
        assert response.choices[0].logprobs is not None
        assert response.choices[0].logprobs.content is not None
        assert response.choices[0].logprobs.content[0].top_logprobs is not None
        return {
            lp.token: float(np.exp(lp.logprob))
            for lp in response.choices[0].logprobs.content[0].top_logprobs
        }

    def get_last_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> Dict[Any, Any]:
        response = self._get_chat_completion(text_input, image_input)
        assert response.choices[0].logprobs is not None
        assert response.choices[0].logprobs.content is not None
        assert response.choices[0].logprobs.content[-1].top_logprobs is not None
        return {
            lp.token: float(np.exp(lp.logprob))
            for lp in response.choices[0].logprobs.content[-1].top_logprobs
        }

    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        if self._use_responses_api():
            return self._get_responses_output(text_input, image_input)
        return self._get_chat_output(text_input, image_input)

    def _get_responses_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": self._create_responses_input(text_input, image_input),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_output_tokens"] = self.max_tokens
        response = self.client.responses.create(**request)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        raise ValueError("OpenAI Responses API returned no output_text")

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
