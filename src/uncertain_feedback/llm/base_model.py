"""Base model interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseModel(ABC):
    """Abstract base class for LLM wrappers."""

    @abstractmethod
    def get_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]],
    ) -> Dict[Any, Any]:
        """Get logits for the first predicted token.

        Returns:
            Dict mapping token string to probability for the top 10 candidates.
        """

    @abstractmethod
    def get_last_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> Dict[Any, Any]:
        """Get logits for the last predicted token.

        Returns:
            Dict mapping token string to probability for the top 10 candidates.
        """

    @abstractmethod
    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Get full text output from the model."""
