import asyncio
import traceback
import time
from typing import Dict, List, Optional, Any, Union, Callable

from .providers import (
    BaseLLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    TogetherProvider,
)
from .providers.base import LLMResponse
from .exceptions import LLMError, ProviderError, ConfigurationError
from .config import LLMConfig
from .llm_providers import LLMProvider
from copy import deepcopy

# Keep the original LLMResponse for backwards compatibility
# (it's now defined in providers.base but we re-export it here)
__all__ = [
    "LLMResponse",
    "chat_async",
    "map_model_to_provider",
    "get_provider_instance",
]


def get_provider_instance(
    provider: Union[LLMProvider, str], config: Optional[LLMConfig] = None
) -> BaseLLMProvider:
    """
    Get a provider instance based on the provider enum or string.

    Args:
        provider: LLMProvider enum or string name
        config: Optional configuration object

    Returns:
        BaseLLMProvider instance

    Raises:
        ConfigurationError: If provider is not supported or misconfigured
    """
    if config is None:
        config = LLMConfig()

    # Handle both enum and string values
    if isinstance(provider, LLMProvider):
        provider_name = provider.value
    else:
        provider_name = provider.lower()

    # Validate provider config
    if not config.validate_provider_config(provider_name):
        raise ConfigurationError(f"No API key found for provider '{provider_name}'")

    # Create provider instances
    provider_classes = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "together": TogetherProvider,
        "deepseek": OpenAIProvider,  # DeepSeek uses OpenAI-compatible API
    }

    if provider_name not in provider_classes:
        raise ConfigurationError(f"Unsupported provider: {provider_name}")

    provider_class = provider_classes[provider_name]

    # Handle special cases for providers that need custom configuration
    if provider_name == "deepseek":
        return provider_class(
            api_key=config.get_api_key("deepseek"),
            base_url=config.get_base_url("deepseek"),
            config=config,
        )
    elif provider_name == "openai":
        return provider_class(
            api_key=config.get_api_key("openai"),
            base_url=config.get_base_url("openai"),
            config=config,
        )
    else:
        return provider_class(api_key=config.get_api_key(provider_name), config=config)


async def chat_async(
    provider: Union[LLMProvider, str],
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format=None,
    seed: int = 0,
    store: bool = True,
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    backup_model: Optional[str] = None,
    backup_provider: Optional[Union[LLMProvider, str]] = None,
    prediction: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
    tools: Optional[List[Callable]] = None,
    tool_choice: Optional[str] = None,
    max_retries: Optional[int] = None,
    post_tool_function: Optional[Callable] = None,
    config: Optional[LLMConfig] = None,
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
) -> LLMResponse:
    """
    Execute a chat completion with explicit provider parameter.

    Args:
        provider: LLMProvider enum or string specifying which provider to use
        model: Model name to use
        messages: List of message dictionaries
        max_completion_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        response_format: Structured output format (Pydantic model)
        seed: Random seed for reproducibility
        store: Whether to store the conversation
        metadata: Additional metadata
        timeout: Request timeout in seconds
        backup_model: Fallback model to use on retry
        backup_provider: Fallback provider to use on retry
        prediction: Predicted output configuration (OpenAI only)
        reasoning_effort: Reasoning effort level (o1/o3 models only)
        tools: List of tools the model can call
        tool_choice: Tool calling behavior ("auto", "required", function name)
        max_retries: Maximum number of retry attempts
        post_tool_function: Function to call after each tool execution
        config: LLM configuration object
        mcp_servers: List of MCP server configurations (Anthropic only)

    Returns:
        LLMResponse object containing the result

    Raises:
        ConfigurationError: If provider configuration is invalid
        ProviderError: If the provider API call fails
        LLMError: For other LLM-related errors
    """
    # create a deep copy of the messages to avoid modifying the original messages
    messages = deepcopy(messages)

    if config is None:
        config = LLMConfig()

    if max_retries is None:
        max_retries = config.max_retries

    base_delay = 1  # Initial delay in seconds
    error_trace = None

    for attempt in range(max_retries):
        try:
            # Use backup provider/model on subsequent attempts
            current_provider = provider
            current_model = model

            if attempt > 0:
                if backup_provider is not None:
                    current_provider = backup_provider
                if backup_model is not None:
                    current_model = backup_model

            # Get provider instance
            provider_instance = get_provider_instance(current_provider, config)

            # Execute the chat completion
            return await provider_instance.execute_chat(
                messages=messages,
                model=current_model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                response_format=response_format,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                store=store,
                metadata=metadata,
                timeout=timeout,
                prediction=prediction,
                reasoning_effort=reasoning_effort,
                post_tool_function=post_tool_function,
                mcp_servers=mcp_servers,
            )

        except Exception as e:
            error_trace = traceback.format_exc()
            delay = base_delay * (2**attempt)  # Exponential backoff

            if attempt < max_retries - 1:  # Don't log on final attempt
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...",
                    flush=True,
                )
                print(f"Error: {e}", flush=True)
                await asyncio.sleep(delay)
            else:
                # Final attempt failed, re-raise the exception
                if isinstance(e, LLMError):
                    raise e
                else:
                    raise LLMError(f"All attempts failed. Latest error: {e}") from e

    # This should never be reached, but just in case
    raise LLMError(
        f"All {max_retries} attempts at calling the chat_async function failed. "
        f"The latest error traceback was: {error_trace}"
    )


# Legacy compatibility functions - these map the old model-based routing to the new provider-based system
def map_model_to_provider(model: str) -> LLMProvider:
    """
    Map a model name to its provider for backwards compatibility.

    Args:
        model: Model name

    Returns:
        LLMProvider enum value

    Raises:
        ConfigurationError: If model cannot be mapped to a provider
    """
    if model.startswith("claude"):
        return LLMProvider.ANTHROPIC
    elif model.startswith("gemini"):
        return LLMProvider.GEMINI
    elif (
        model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("chatgpt")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return LLMProvider.OPENAI
    elif model.startswith("deepseek"):
        return LLMProvider.DEEPSEEK
    elif (
        model.startswith("meta-llama")
        or model.startswith("mistralai")
        or model.startswith("Qwen")
    ):
        return LLMProvider.TOGETHER
    else:
        raise ConfigurationError(f"Unknown model: {model}")


async def chat_async_legacy(
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format=None,
    seed: int = 0,
    store: bool = True,
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    backup_model: Optional[str] = None,
    prediction: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
    tools: Optional[List[Callable]] = None,
    tool_choice: Optional[str] = None,
    max_retries: int = 3,
    post_tool_function: Optional[Callable] = None,
) -> LLMResponse:
    """
    Legacy function that infers provider from model name.

    Deprecated: Use chat_async with explicit provider parameter instead.
    """
    provider = map_model_to_provider(model)
    backup_provider = None
    if backup_model:
        backup_provider = map_model_to_provider(backup_model)

    return await chat_async(
        provider=provider,
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        response_format=response_format,
        seed=seed,
        store=store,
        metadata=metadata,
        timeout=timeout,
        backup_model=backup_model,
        backup_provider=backup_provider,
        prediction=prediction,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        max_retries=max_retries,
        post_tool_function=post_tool_function,
    )
