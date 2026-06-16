"""Base model interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union


class BaseModel(ABC):
    """Abstract base class for LLM wrappers."""

    @abstractmethod
    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Get full text output from the model."""
