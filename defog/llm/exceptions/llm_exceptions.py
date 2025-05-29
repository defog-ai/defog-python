class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class ProviderError(LLMError):
    """Exception raised when there's an error with a specific LLM provider."""

    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Provider '{provider}': {message}")


class ToolError(LLMError):
    """Exception raised when there's an error with tool calling."""

    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}': {message}")


class MaxTokensError(LLMError):
    """Exception raised when maximum tokens are reached."""

    pass


class ConfigurationError(LLMError):
    """Exception raised when there's a configuration error."""

    pass


class AuthenticationError(LLMError):
    """Exception raised when there's an authentication error with a provider."""

    pass


class APIError(LLMError):
    """Exception raised when there's a general API error."""

    pass
