"""OpenAI model wrapper."""

import base64
import os
from typing import Any, Dict, List, Optional, Union, cast

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
    ):
        """Initialize OpenAI model.

        Reads OPENAI_API_KEY and optionally OPENAI_ORG_ID from the environment.

        Args:
            model: OpenAI model name (e.g. "gpt-4.1-nano").
            system_prompt: System prompt prepended to every request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
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
                "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img)}"},
            }
            for img in image_input
        ]

    def _get_chat_completion(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> ChatCompletion:
        user_prompt = self._create_prompt(text_input, image_input)
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cast(Any, user_prompt)},
            ],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=10,
        )

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
        user_prompt = self._create_prompt(text_input, image_input)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cast(Any, user_prompt)},
            ],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned None content")
        return content
