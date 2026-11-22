"""Pick the LLM wrapper from the model name."""

from typing import Optional, Union

from .claude_model import ClaudeModel
from .openai_model import OpenAIModel


def make_model(
    model: str,
    system_prompt: str,
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    stream_reasoning_summary: bool = False,
) -> Union[OpenAIModel, ClaudeModel]:
    """``claude*`` models run on the ``claude`` CLI's subscription, others on OpenAI.

    The CLI takes no temperature or token limit, so a Claude model keeps only
    ``reasoning_effort``.
    """
    if model.startswith("claude"):
        return ClaudeModel(model, system_prompt, reasoning_effort)
    return OpenAIModel(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        stream_reasoning_summary=stream_reasoning_summary,
    )
