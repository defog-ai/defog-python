import os
import time
import json
import re
from typing import Dict, List, Any, Optional, Callable, Tuple

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..cost import CostCalculator
from ..tools import ToolHandler
from ..utils_function_calling import get_function_specs, convert_tool_choice


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or os.getenv("MISTRAL_API_KEY"),
            base_url or "https://api.mistral.ai/v1/",
            config=config,
        )
        self.tool_handler = ToolHandler()

    def get_provider_name(self) -> str:
        return "mistral"

    def supports_tools(self, model: str) -> bool:
        # Most Mistral models support tools
        return True

    def supports_response_format(self, model: str) -> bool:
        # Most Mistral models support structured output
        return True

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
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Build parameters for Mistral API (OpenAI-compatible)."""
        import copy

        messages = copy.deepcopy(messages)

        request_params = {
            "messages": messages,
            "model": model,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "random_seed": seed,
        }

        # Tools are supported
        if tools and len(tools) > 0:
            function_specs = get_function_specs(tools, model)
            request_params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                request_params["tool_choice"] = tool_choice
            else:
                request_params["tool_choice"] = "auto"

        # Response format (JSON mode)
        if response_format:
            request_params["response_format"] = {"type": "json_object"}

        return request_params, messages

    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        model: str = "",
        post_tool_function: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Process response from Mistral API."""
        if len(response.choices) == 0:
            raise ProviderError(self.get_provider_name(), "No response from Mistral")
        if response.choices[0].finish_reason == "length":
            raise MaxTokensError("Max tokens reached")

        # Handle tool calls if present
        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        
        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens
                message = response.choices[0].message
                
                if message.tool_calls:
                    try:
                        # Prepare tool calls for execution
                        tool_calls_batch = []
                        for tool_call in message.tool_calls:
                            func_name = tool_call.function.name
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {}

                            tool_calls_batch.append(
                                {
                                    "id": tool_call.id,
                                    "function": {"name": func_name, "arguments": args},
                                }
                            )

                        # Execute tool calls
                        results = await self.tool_handler.execute_tool_calls_batch(
                            tool_calls_batch,
                            tool_dict,
                            enable_parallel=self.config.enable_parallel_tool_calls,
                            post_tool_function=post_tool_function,
                        )

                        consecutive_exceptions = 0

                        # Append assistant response with tool calls
                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "tool_calls": message.tool_calls,
                            }
                        )

                        # Process results and append tool messages
                        for tool_call, result in zip(message.tool_calls, results):
                            func_name = tool_call.function.name
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {}

                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "text": (
                                        message.content if message.content else None
                                    ),
                                }
                            )

                            # Append tool message
                            request_params["messages"].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(result),
                                }
                            )

                        request_params["tool_choice"] = "auto"
                    except Exception as e:
                        consecutive_exceptions += 1

                        if (
                            consecutive_exceptions
                            >= self.tool_handler.max_consecutive_errors
                        ):
                            raise ProviderError(
                                self.get_provider_name(),
                                f"Consecutive errors during tool chaining: {e}",
                                e,
                            )

                        print(
                            f"{e}. Retries left: {self.tool_handler.max_consecutive_errors - consecutive_exceptions}"
                        )

                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "content": str(e),
                            }
                        )

                    # Make next call
                    response = await client.chat.completions.create(**request_params)
                else:
                    content = message.content
                    break
        else:
            # No tools provided
            if response_format:
                content = response.choices[0].message.content
                try:
                    # Parse as JSON
                    content = re.sub(r"```(.*)```", r"\1", content, flags=re.DOTALL)
                    content = json.loads(content)
                except Exception as e:
                    raise ProviderError(
                        self.get_provider_name(), f"Error parsing JSON response: {e}", e
                    )
            else:
                content = response.choices[0].message.content

        usage = response.usage
        total_input_tokens += usage.prompt_tokens or 0
        total_output_tokens += usage.completion_tokens

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            None,  # completion token details not available
        )

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
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Mistral AI."""
        from openai import AsyncOpenAI  # Mistral API is OpenAI-compatible

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client_mistral = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        request_params, messages = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            timeout=timeout,
        )

        # Build tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0:
            tool_dict = self.tool_handler.build_tool_dict(tools)

        try:
            response = await client_mistral.chat.completions.create(**request_params)

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
            ) = await self.process_response(
                client=client_mistral,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                post_tool_function=post_tool_function,
            )
        except Exception as e:
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Calculate cost
        cost = CostCalculator.calculate_cost(
            model, input_tokens, output_tokens, cached_input_tokens
        )

        return LLMResponse(
            model=model,
            content=content,
            time=round(time.time() - t, 3),
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=completion_token_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
        )