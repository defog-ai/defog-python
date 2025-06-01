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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or os.getenv("OPENAI_API_KEY"),
            base_url or "https://api.openai.com/v1/",
            config=config,
        )
        self.tool_handler = ToolHandler()

    def get_provider_name(self) -> str:
        return "openai"

    def supports_tools(self, model: str) -> bool:
        True

    def supports_response_format(self, model: str) -> bool:
        True

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
        """
        Build the parameter dictionary for OpenAI's chat.completions.create().
        Also handles special logic for o1-mini, o1-preview, deepseek-chat, etc.
        """
        # Potentially move system message to user message for certain model families:
        # if a message is called "system", rename it to "developer"
        # create a new list of messages
        import copy

        messages = copy.deepcopy(messages)

        for i in range(len(messages)):
            if model not in ["gpt-4o", "gpt-4o-mini"]:
                if messages[i].get("role") == "system":
                    messages[i]["role"] = "developer"

        request_params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "seed": seed,
            "store": store,
            "metadata": metadata,
            "timeout": timeout,
        }

        # Tools are only supported for certain models
        if tools and len(tools) > 0:
            function_specs = get_function_specs(tools, model)
            request_params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                request_params["tool_choice"] = tool_choice
            else:
                request_params["tool_choice"] = "auto"

            # Set parallel_tool_calls based on configuration
            # this parameter is not available on the o-series models, though
            if not model.startswith("o"):
                request_params["parallel_tool_calls"] = (
                    self.config.enable_parallel_tool_calls
                )

        # Some models do not allow temperature or response_format:
        if model.startswith("o") or model == "deepseek-reasoner":
            request_params.pop("temperature", None)

        # Reasoning effort
        if model.startswith("o") and reasoning_effort is not None:
            request_params["reasoning_effort"] = reasoning_effort

        # Special case: model in ["gpt-4o", "gpt-4o-mini"] with `prediction`
        if model in ["gpt-4o", "gpt-4o-mini"] and prediction is not None:
            request_params["prediction"] = prediction
            request_params.pop("max_completion_tokens", None)
            request_params.pop("response_format", None)

        # Finally, set response_format if still relevant:
        if response_format:
            request_params["response_format"] = response_format

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
        """
        Extract content (including any tool calls) and usage info from OpenAI response.
        Handles chaining of tool calls.
        """
        if len(response.choices) == 0:
            raise ProviderError(self.get_provider_name(), "No response from OpenAI")
        if response.choices[0].finish_reason == "length":
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                # Check if prompt_tokens_details exists in the response
                if (
                    hasattr(response.usage, "prompt_tokens_details")
                    and response.usage.prompt_tokens_details is not None
                ):
                    cached_tokens = (
                        response.usage.prompt_tokens_details.cached_tokens or 0
                    )
                    total_input_tokens += (
                        response.usage.prompt_tokens or 0 - cached_tokens
                    )
                    total_cached_input_tokens += cached_tokens
                else:
                    # If prompt_tokens_details doesn't exist, assume all tokens are uncached
                    total_input_tokens += response.usage.prompt_tokens or 0

                total_output_tokens += response.usage.completion_tokens
                message = response.choices[0].message
                if message.tool_calls:
                    try:
                        # Prepare tool calls for batch execution
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

                        # Execute all tool calls (parallel or sequential based on config)
                        results = await self.tool_handler.execute_tool_calls_batch(
                            tool_calls_batch,
                            tool_dict,
                            enable_parallel=self.config.enable_parallel_tool_calls,
                            post_tool_function=post_tool_function,
                        )

                        # Reset consecutive_exceptions when tool calls are successful
                        consecutive_exceptions = 0

                        # Append the tool calls as an assistant response
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

                            # Store the tool call, result, and text
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

                            # Append the tool message
                            request_params["messages"].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(result),
                                }
                            )

                        # Set tool_choice to "auto" so that the next message will be generated normally
                        request_params["tool_choice"] = (
                            "auto" if request_params["tool_choice"] != "auto" else None
                        )
                    except Exception as e:
                        consecutive_exceptions += 1

                        # Break the loop if consecutive exceptions exceed the threshold
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

                        # Append error message to request_params and retry
                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "content": str(e),
                            }
                        )

                    # Make next call
                    response = await client.chat.completions.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    content = message.content
                    break
        else:
            # No tools provided
            if response_format:
                try:
                    content = response.choices[0].message.parsed
                except Exception as e:
                    content = response.choices[0].message.content
                    # parse the content as json
                    try:
                        # clean up any markdown formatting
                        content = re.sub(r"```(.*)```", r"\1", content)
                        content = json.loads(content)
                    except Exception as e:
                        raise ProviderError(
                            self.get_provider_name(), f"Error parsing content: {e}", e
                        )
            else:
                content = response.choices[0].message.content

        usage = response.usage
        # Check if prompt_tokens_details exists in the response
        if (
            hasattr(usage, "prompt_tokens_details")
            and usage.prompt_tokens_details is not None
        ):
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            total_cached_input_tokens += cached_tokens
            total_input_tokens += usage.prompt_tokens or 0 - cached_tokens
        else:
            # If prompt_tokens_details doesn't exist, assume all tokens are uncached
            total_input_tokens += usage.prompt_tokens or 0

        total_output_tokens += usage.completion_tokens
        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            usage.completion_tokens_details,
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
        timeout: int = 100,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with OpenAI."""
        from openai import AsyncOpenAI

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client_openai = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        request_params, messages = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            prediction=prediction,
            reasoning_effort=reasoning_effort,
            store=store,
            metadata=metadata,
            timeout=timeout,
        )

        # Build a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = self.tool_handler.build_tool_dict(tools)

        try:
            # If response_format is set, we do parse
            if request_params.get("response_format"):
                response = await client_openai.beta.chat.completions.parse(
                    **request_params
                )
            else:
                response = await client_openai.chat.completions.create(**request_params)

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
            ) = await self.process_response(
                client=client_openai,
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
