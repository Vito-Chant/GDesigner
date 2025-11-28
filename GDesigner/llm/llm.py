from abc import ABC, abstractmethod
from typing import List, Dict, Union
import asyncio
import weave
from GDesigner.llm.utils import convert_messages_to_dict
from GDesigner.llm.format import Message


class LLM(ABC):
    """Base class for LLM (Large Language Model), encapsulates common logic"""

    def __init__(self, llm_name: str, max_tokens: int = 2048, temperature: float = 0.7):
        """
        Initialize the BaseLLM instance

        Args:
            llm_name: Name of the LLM model (e.g., "gpt-3.5-turbo", "llama-2")
            max_tokens: Maximum number of tokens allowed for the model's response (default: 2048)
            temperature: Controls randomness in the model's output (0.0 = deterministic, 1.0 = most random; default: 0.7)
        """
        self.llm_name = llm_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    @weave.op()
    async def agen(self, messages: Union[List[Message], List[Dict[str, str]]], **kwargs) -> str:
        """Asynchronously generate model response (to be implemented by subclasses)

        Args:
            messages: Input messages for the model, can be either a list of Message objects or dictionaries
            **kwargs: Additional model-specific parameters (e.g., top_p, frequency_penalty)

        Returns:
            Generated response string from the model
        """
        pass

    @weave.op()
    async def aembed(self, messages: Union[List[Message], List[Dict[str, str]]], **kwargs) -> str:
        raise "Not implemented"

    def gen(self, messages: Union[List[Message], List[Dict[str, str]]], **kwargs) -> str:
        """Synchronously generate model response (wrapped based on the async agen method)

        Args:
            messages: Input messages for the model, can be either a list of Message objects or dictionaries
            **kwargs: Additional model-specific parameters passed to the agen method

        Returns:
            Generated response string from the model
        """
        return asyncio.run(self.agen(messages, **kwargs))

    def _preprocess_messages(self, messages: Union[List[Message], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Preprocess message format (unify conversion to list of dictionaries)

        Converts input messages (either Message objects or dictionaries) to a standard list of dictionaries
        with 'role' and 'content' keys, using the convert_messages_to_dict utility function.

        Args:
            messages: Input messages to be preprocessed

        Returns:
            Standardized list of message dictionaries with 'role' and 'content' keys
        """
        return convert_messages_to_dict(messages)