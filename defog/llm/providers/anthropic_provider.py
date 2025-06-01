import os
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..cost import CostCalculator
from ..tools import ToolHandler
from ..utils_function_calling import get_function_specs, convert_tool_choice


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: Optional[str] = None, config=None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), config=config)
        self.tool_handler = ToolHandler()

    def get_provider_name(self) -> str:
        return "anthropic"

    def supports_tools(self, model: str) -> bool:
        return True  # All current Claude models support tools

    def supports_response_format(self, model: str) -> bool:
        return True  # All current Claude models support structured output via system prompts

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
        """Create the parameter dict for Anthropic's .messages.create()."""
        if len(messages) >= 1 and messages[0].get("role") == "system":
            sys_msg = messages[0]["content"]
            messages = messages[1:]
        else:
            sys_msg = ""

        if reasoning_effort is not None:
            temperature = 1.0
            if reasoning_effort == "low":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": 2048,
                }
            elif reasoning_effort == "medium":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": 4096,
                }
            elif reasoning_effort == "high":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": 8192,
                }
        else:
            thinking = {
                "type": "disabled",
            }

        # Anthropic does not allow `None` as a value for max_completion_tokens
        if max_completion_tokens is None:
            max_completion_tokens = 32000

        params = {
            "system": sys_msg,
            "messages": messages,
            "model": model,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "timeout": timeout,
            "thinking": thinking,
        }

        # Handle structured output for Anthropic models
        if response_format:
            # Add instructions to the system message to enforce structured output
            if isinstance(response_format, type) and hasattr(
                response_format, "model_json_schema"
            ):
                schema = response_format.model_json_schema()
                schema_str = json.dumps(schema, indent=2)

                # Append structured output instructions to system prompt
                structured_instruction = f"""
IMPORTANT: You must respond with ONLY a valid, properly formatted JSON object that conforms to the following JSON schema:
{schema_str}

RESPONSE FORMAT INSTRUCTIONS:
1. Your entire response must be ONLY the JSON object, with no additional text before or after.
2. Format the JSON properly with no line breaks within property values.
3. Use double quotes for all property names and string values.
4. Do not add comments or explanations outside the JSON structure.
5. Ensure all required properties in the schema are included.
6. Make sure the JSON is properly formatted and can be parsed by standard JSON parsers.

THE RESPONSE SHOULD START WITH '{{' AND END WITH '}}' WITH NO OTHER CHARACTERS BEFORE OR AFTER.
"""
                # Update system message
                if sys_msg:
                    params["system"] = sys_msg + "\n\n" + structured_instruction
                else:
                    params["system"] = structured_instruction

        if tools:
            function_specs = get_function_specs(tools, model)
            params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                params["tool_choice"] = tool_choice
            else:
                params["tool_choice"] = {"type": "auto"}

            # Add parallel tool calls configuration
            if "tool_choice" in params and isinstance(params["tool_choice"], dict):
                if not self.config.enable_parallel_tool_calls:
                    params["tool_choice"]["disable_parallel_tool_use"] = True

        return params, messages

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
        """
        Extract content (including any tool calls) and usage info from Anthropic response.
        Handles chaining of tool calls and structured output parsing.
        """
        from anthropic.types import ToolUseBlock, TextBlock, ThinkingBlock

        if response.stop_reason == "max_tokens":
            raise MaxTokensError("Max tokens reached")
        if len(response.content) == 0:
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_output_tokens = 0

        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens
                # Check if the response contains a tool call
                # Collect all blocks by type
                tool_call_blocks = [
                    block
                    for block in response.content
                    if isinstance(block, ToolUseBlock)
                ]
                thinking_blocks = [
                    block
                    for block in response.content
                    if isinstance(block, ThinkingBlock)
                ]
                text_blocks = [
                    block for block in response.content if isinstance(block, TextBlock)
                ]
                if len(tool_call_blocks) > 0:
                    try:
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tool_call_block in tool_call_blocks:
                            try:
                                func_name = tool_call_block.name
                                args = tool_call_block.input
                                tool_id = tool_call_block.id

                                tool_calls_batch.append(
                                    {
                                        "id": tool_id,
                                        "function": {
                                            "name": func_name,
                                            "arguments": args,
                                        },
                                    }
                                )
                            except Exception as e:
                                raise ProviderError(
                                    self.get_provider_name(),
                                    f"Error parsing tool call: {e}",
                                    e,
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

                        # Build assistant content with all tool calls
                        assistant_content = []
                        if len(thinking_blocks) > 0:
                            assistant_content += thinking_blocks

                        for tool_call_block in tool_call_blocks:
                            assistant_content.append(tool_call_block)

                        # Append the tool calls as an assistant response
                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "content": assistant_content,
                            }
                        )

                        # Build user response with all tool results
                        tool_results_content = []
                        for tool_call_block, result in zip(tool_call_blocks, results):
                            func_name = tool_call_block.name
                            args = tool_call_block.input
                            tool_id = tool_call_block.id

                            # Store the tool call, result, and text
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                }
                            )

                            tool_results_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": str(result),
                                }
                            )

                        # Append all tool results in a single user message
                        request_params["messages"].append(
                            {
                                "role": "user",
                                "content": tool_results_content,
                            }
                        )

                        # Set tool_choice to "auto" so that the next message will be generated normally
                        request_params["tool_choice"] = (
                            {"type": "auto"}
                            if request_params["tool_choice"] != "auto"
                            else None
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
                    response = await client.messages.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    content = "\n".join([block.text for block in text_blocks])
                    break
        else:
            # No tools provided
            for block in response.content:
                if isinstance(block, TextBlock):
                    content = block.text
                    break

        # Parse structured output if response_format is provided
        if response_format and not tools:
            # Check if response_format is a Pydantic model
            try:
                # Extract the raw text response and clean it
                content = content.strip()
                print(content)
                content = json.loads(content)

                # Parse the response into the specified Pydantic model
                content = response_format.parse_obj(content)
            except Exception as e:
                # If parsing fails, return the raw content
                print(f"Warning: Failed to parse structured output: {e}")
                # We keep the raw content in this case
                content = content

        usage = response.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            None,
            None,
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
        """Execute a chat completion with Anthropic."""
        from anthropic import AsyncAnthropic

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client = AsyncAnthropic(
            api_key=self.api_key,
            default_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
        )
        params, _ = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

        # Construct a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in params:
            tool_dict = self.tool_handler.build_tool_dict(tools)

        try:
            response = await client.messages.create(**params)
            (
                content,
                tool_outputs,
                input_toks,
                output_toks,
                cached_toks,
                output_details,
            ) = await self.process_response(
                client=client,
                response=response,
                request_params=params,
                tools=tools,
                tool_dict=tool_dict,
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
