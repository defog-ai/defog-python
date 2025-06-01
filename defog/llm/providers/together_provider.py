import os
import time
from typing import Dict, List, Any, Optional, Callable, Tuple

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..cost import CostCalculator


class TogetherProvider(BaseLLMProvider):
    """Together AI provider implementation."""

    def __init__(self, api_key: Optional[str] = None, config=None):
        super().__init__(api_key or os.getenv("TOGETHER_API_KEY"), config=config)

    def get_provider_name(self) -> str:
        return "together"

    def supports_tools(self, model: str) -> bool:
        return (
            False  # Currently Together models don't support tools in our implementation
        )

    def supports_response_format(self, model: str) -> bool:
        return False  # Currently Together models don't support structured output in our implementation

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
        timeout: int = 100,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Build parameters for Together's API call."""
        return {
            "messages": messages,
            "model": model,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "seed": seed,
        }, messages

    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        post_tool_function: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Process Together API response."""
        if response.choices[0].finish_reason == "length":
            raise MaxTokensError("Max tokens reached")
        if len(response.choices) == 0:
            raise MaxTokensError("Max tokens reached")

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return content, [], input_tokens, output_tokens, None, None

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
        timeout: int = 100,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Together."""
        from together import AsyncTogether

        t = time.time()
        client_together = AsyncTogether(timeout=timeout)
        params, _ = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            seed=seed,
        )

        try:
            response = await client_together.chat.completions.create(**params)
            (
                content,
                tool_outputs,
                input_toks,
                output_toks,
                cached_toks,
                output_details,
            ) = await self.process_response(
                client=client_together,
                response=response,
                request_params=params,
                tools=tools,
                tool_dict={},
                response_format=response_format,
                post_tool_function=post_tool_function,
            )
        except Exception as e:
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Calculate cost
        cost = CostCalculator.calculate_cost(
            model, input_toks, output_toks, cached_toks
        )

        return LLMResponse(
            model=model,
            content=content,
            time=round(time.time() - t, 3),
            input_tokens=input_toks,
            output_tokens=output_toks,
            cached_input_tokens=cached_toks,
            output_tokens_details=output_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
        )
