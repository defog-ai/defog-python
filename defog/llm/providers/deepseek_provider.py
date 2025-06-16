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


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider implementation using OpenAI-compatible API."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url or "https://api.deepseek.com",
            config=config,
        )
        self.tool_handler = ToolHandler()

    def get_provider_name(self) -> str:
        return "deepseek"

    def supports_tools(self, model: str) -> bool:
        # Only deepseek-chat supports tools, deepseek-reasoner does not
        return model == "deepseek-chat"

    def supports_response_format(self, model: str) -> bool:
        # Both models support JSON response format
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
        """
        Build the parameter dictionary for DeepSeek's chat.completions.create().
        """
        import copy

        messages = copy.deepcopy(messages)

        # Convert system messages to developer messages for DeepSeek
        for i in range(len(messages)):
            if messages[i].get("role") == "system":
                messages[i]["role"] = "developer"

        request_params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "seed": seed,
            "store": store,
            "metadata": metadata,
            "timeout": timeout,
        }

        # DeepSeek-reasoner doesn't support temperature
        if model != "deepseek-reasoner":
            request_params["temperature"] = temperature

        # Tools are supported for deepseek-chat
        if tools and len(tools) > 0:
            if not self.supports_tools(model):
                raise ProviderError(
                    self.get_provider_name(),
                    f"Model {model} does not support tools/function calling"
                )
            
            function_specs = get_function_specs(tools, model)
            request_params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                request_params["tool_choice"] = tool_choice
            else:
                request_params["tool_choice"] = "auto"

            # Set parallel_tool_calls based on configuration
            request_params["parallel_tool_calls"] = (
                self.config.enable_parallel_tool_calls
            )

        # Handle structured output for DeepSeek models
        if response_format and self.supports_response_format(model):
            # Check if response_format is a Pydantic model
            if isinstance(response_format, type) and hasattr(
                response_format, "model_json_schema"
            ):
                # Convert Pydantic model to JSON schema and add instructions
                schema = response_format.model_json_schema()
                schema_str = json.dumps(schema, indent=2)

                # Set JSON mode for DeepSeek
                request_params["response_format"] = {"type": "json_object"}

                # Append structured output instructions to the latest user message
                # DeepSeek requires the word "json" in the prompt
                structured_instruction = f"""

IMPORTANT: You must respond with ONLY a valid JSON object that conforms to the following JSON schema:
{schema_str}

Please format your response as a JSON object. Make sure to:
1. Include all required properties from the schema
2. Use proper JSON formatting with double quotes
3. Ensure the JSON is valid and parseable
4. Do not include any text outside the JSON object

Respond with JSON only.
"""
                # Find the latest user message and append the structured instruction
                if messages and len(messages) > 0:
                    # Find the last user message
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i]["content"] += structured_instruction
                            break
            else:
                # For non-Pydantic response formats, use the standard approach
                request_params["response_format"] = response_format

        # DeepSeek doesn't support MCP servers
        if mcp_servers:
            raise ProviderError(
                self.get_provider_name(),
                "DeepSeek does not support MCP servers"
            )

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
        Extract content (including any tool calls) and usage info from DeepSeek response.
        Handles chaining of tool calls.
        """
        if len(response.choices) == 0:
            raise ProviderError(self.get_provider_name(), "No response from DeepSeek")
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
            content = response.choices[0].message.content
            
            # Parse structured output if response_format is provided
            if response_format:
                # Check if response_format is a Pydantic model
                if isinstance(response_format, type) and hasattr(
                    response_format, "model_json_schema"
                ):
                    try:
                        # Extract the raw text response and clean it
                        content = content.strip()
                        # Remove any markdown formatting
                        if "```json" in content:
                            content = content[content.index("```json") + len("```json") :]
                        if "```" in content:
                            content = content[: content.index("```")]
                        # Strip the content again
                        content = content.strip()
                        
                        # Parse as JSON first
                        json_content = json.loads(content)
                        
                        # Parse the response into the specified Pydantic model
                        content = response_format.model_validate(json_content)
                    except Exception as e:
                        # If parsing fails, try to get parsed content from OpenAI SDK
                        try:
                            content = response.choices[0].message.parsed
                            if content is None:
                                raise ValueError("No parsed content available")
                        except Exception:
                            raise ProviderError(
                                self.get_provider_name(), 
                                f"Error parsing structured output: {e}. Raw content: {content}", 
                                e
                            )
                else:
                    # For non-Pydantic response formats, use standard OpenAI parsing
                    try:
                        content = response.choices[0].message.parsed
                    except Exception:
                        # Fallback to manual JSON parsing
                        try:
                            # Clean up any markdown formatting
                            content = re.sub(r"```(.*)```", r"\1", content)
                            content = json.loads(content)
                        except Exception as e:
                            raise ProviderError(
                                self.get_provider_name(), f"Error parsing content: {e}", e
                            )

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
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with DeepSeek."""
        from openai import AsyncOpenAI

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client_deepseek = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
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
            mcp_servers=mcp_servers,
        )

        # Build a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = self.tool_handler.build_tool_dict(tools)

        try:
            # Use regular Chat Completions API
            if request_params.get("response_format"):
                response = await client_deepseek.beta.chat.completions.parse(
                    **request_params
                )
            else:
                response = await client_deepseek.chat.completions.create(
                    **request_params
                )

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
            ) = await self.process_response(
                client=client_deepseek,
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