from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from ..config.settings import LLMConfig


@dataclass
class LLMResponse:
    content: Any
    model: str
    time: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: Optional[int] = None
    output_tokens_details: Optional[Dict[str, int]] = None
    cost_in_cents: Optional[float] = None
    tool_outputs: Optional[List[Dict[str, Any]]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.config = config or LLMConfig()

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass

    @abstractmethod
    def build_params(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        seed: int = 0,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Build parameters for the provider's API call."""
        pass

    @abstractmethod
    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        post_tool_function: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Process the response from the provider."""
        pass

    @abstractmethod
    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        seed: int = 0,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Execute a chat completion with the provider."""
        pass

    @abstractmethod
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling."""
        pass

    @abstractmethod
    def supports_response_format(self, model: str) -> bool:
        """Check if the model supports structured response formats."""
        pass
