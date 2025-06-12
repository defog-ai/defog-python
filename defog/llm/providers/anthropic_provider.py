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
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Create the parameter dict for Anthropic's .messages.create()."""
        if len(messages) >= 1 and messages[0].get("role") == "system":
            sys_msg = messages[0]["content"]
            messages = messages[1:]
        else:
            sys_msg = ""

        if reasoning_effort is not None and ("3-7" in model or "-4-" in model):
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
            # Add instructions to the latest user message to enforce structured output
            if isinstance(response_format, type) and hasattr(
                response_format, "model_json_schema"
            ):
                schema = response_format.model_json_schema()
                schema_str = json.dumps(schema, indent=2)

                # Append structured output instructions to the latest user message
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
                # Find the latest user message and append the structured instruction
                if messages and len(messages) > 0:
                    # Find the last user message
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            # Handle both string content and list content
                            if isinstance(messages[i]["content"], str):
                                messages[i]["content"] += structured_instruction
                            elif isinstance(messages[i]["content"], list):
                                # For list content, append a text block with the instruction
                                messages[i]["content"].append({
                                    "type": "text",
                                    "text": structured_instruction
                                })
                            break

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

        # Add MCP servers if provided
        if mcp_servers:
            params["mcp_servers"] = mcp_servers

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
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """
        Extract content (including any tool calls) and usage info from Anthropic response.
        Handles chaining of tool calls and structured output parsing.
        """
        # Note: We check block.type property instead of using isinstance with specific block classes
        # This ensures compatibility with both regular and beta API responses

        if response.stop_reason == "max_tokens":
            raise MaxTokensError("Max tokens reached")
        if len(response.content) == 0:
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Handle tool processing for both local tools and MCP server tools
        if (tools and len(tools) > 0) or (mcp_servers and len(mcp_servers) > 0):
            consecutive_exceptions = 0
            while True:
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens
                # Check if the response contains a tool call
                # Collect all blocks by type - check type property instead of isinstance
                # Handle both regular tool_use and MCP mcp_tool_use blocks
                tool_call_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type")
                    and block.type in ["tool_use", "mcp_tool_use"]
                ]
                # Collect MCP tool result blocks (these contain results from MCP server execution)
                mcp_tool_result_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type") and block.type == "mcp_tool_result"
                ]
                thinking_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type") and block.type == "thinking"
                ]
                text_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type") and block.type == "text"
                ]
                if len(tool_call_blocks) > 0:
                    try:
                        # Separate MCP tools from regular tools
                        mcp_tool_calls = []
                        regular_tool_calls = []

                        for tool_call_block in tool_call_blocks:
                            try:
                                func_name = tool_call_block.name
                                args = tool_call_block.input
                                tool_id = tool_call_block.id

                                # Check if this is an MCP tool call
                                is_mcp_tool = (
                                    hasattr(tool_call_block, "type")
                                    and tool_call_block.type == "mcp_tool_use"
                                )

                                tool_call_info = {
                                    "id": tool_id,
                                    "function": {
                                        "name": func_name,
                                        "arguments": args,
                                    },
                                }

                                if is_mcp_tool:
                                    # Add MCP-specific info if available
                                    if hasattr(tool_call_block, "server_name"):
                                        tool_call_info["server_name"] = (
                                            tool_call_block.server_name
                                        )
                                    mcp_tool_calls.append(tool_call_info)
                                else:
                                    regular_tool_calls.append(tool_call_info)

                            except Exception as e:
                                raise ProviderError(
                                    self.get_provider_name(),
                                    f"Error parsing tool call: {e}",
                                    e,
                                )

                        # Execute regular tool calls (not MCP tools, which are already executed by the API)
                        results = []
                        if regular_tool_calls:
                            results = await self.tool_handler.execute_tool_calls_batch(
                                regular_tool_calls,
                                tool_dict,
                                enable_parallel=self.config.enable_parallel_tool_calls,
                                post_tool_function=post_tool_function,
                            )

                        # For MCP tools, extract results from mcp_tool_result blocks
                        mcp_results = []
                        for mcp_result_block in mcp_tool_result_blocks:
                            try:
                                # Extract result content
                                result_content = ""
                                if (
                                    hasattr(mcp_result_block, "content")
                                    and mcp_result_block.content
                                ):
                                    for content_item in mcp_result_block.content:
                                        if (
                                            hasattr(content_item, "type")
                                            and content_item.type == "text"
                                        ):
                                            result_content += content_item.text
                                mcp_results.append(result_content)
                            except Exception as e:
                                print(f"Warning: Failed to parse MCP tool result: {e}")
                                mcp_results.append("Error parsing MCP result")

                        # Combine results in the order they were called
                        all_results = []
                        regular_idx = 0
                        mcp_idx = 0
                        for tool_call_block in tool_call_blocks:
                            is_mcp_tool = (
                                hasattr(tool_call_block, "type")
                                and tool_call_block.type == "mcp_tool_use"
                            )
                            if is_mcp_tool:
                                if mcp_idx < len(mcp_results):
                                    all_results.append(mcp_results[mcp_idx])
                                    mcp_idx += 1
                                else:
                                    all_results.append("MCP result not found")
                            else:
                                if regular_idx < len(results):
                                    all_results.append(results[regular_idx])
                                    regular_idx += 1
                                else:
                                    all_results.append("Regular tool result not found")

                        results = all_results

                        # Reset consecutive_exceptions when tool calls are successful
                        consecutive_exceptions = 0

                        # Store tool outputs for all tools (both MCP and regular)
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

                        # Check stop_reason to determine if we should continue
                        if response.stop_reason == "end_turn":
                            # Conversation is complete, extract final content and break
                            content = "\n".join([block.text for block in text_blocks])
                            break
                        elif response.stop_reason == "tool_use":
                            # Need to continue conversation with tool results (for regular tools only)
                            # MCP tools are already executed, so this shouldn't apply to them
                            if (
                                regular_tool_calls
                            ):  # Only continue if we have regular tools to execute
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
                                for tool_call_block, result in zip(
                                    tool_call_blocks, results
                                ):
                                    tool_id = tool_call_block.id
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
                            else:
                                # Only MCP tools, conversation is complete
                                content = "\n".join(
                                    [block.text for block in text_blocks]
                                )
                                break
                        else:
                            # For other stop reasons, extract content and break
                            content = "\n".join([block.text for block in text_blocks])
                            break
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

                    # Make next call - use the same API endpoint (beta or regular) as the initial call
                    if mcp_servers and len(mcp_servers) > 0:
                        response = await client.beta.messages.create(**request_params)
                    else:
                        response = await client.messages.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    content = "\n".join([block.text for block in text_blocks])
                    break
        else:
            # No tools provided
            content = ""
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    content = block.text
                    break

        # Parse structured output if response_format is provided
        if response_format and not tools:
            # Check if response_format is a Pydantic model
            try:
                # Extract the raw text response and clean it
                content = content.strip()
                # remove the ```json and ``` from the content
                if "```json" in content:
                    content = content[content.index("```json") + len("```json") :]
                if "```" in content:
                    content = content[: content.index("```")]
                # strip the content again
                content = content.strip()
                content = json.loads(content)

                # Parse the response into the specified Pydantic model
                content = response_format.parse_obj(content)
            except Exception as e:
                # If parsing fails, return the raw content
                print(f"Warning: Failed to parse structured output: {e}")
                print(content)
                # We keep the raw content in this case
                content = content

        usage = response.usage
        total_input_tokens += usage.input_tokens + usage.cache_creation_input_tokens
        total_output_tokens += usage.output_tokens
        cached_input_tokens = usage.cache_read_input_tokens

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            cached_input_tokens,
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
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Anthropic."""
        from anthropic import AsyncAnthropic

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()

        # Set up headers based on whether MCP servers are provided
        headers = {}
        if mcp_servers:
            headers["anthropic-beta"] = "mcp-client-2025-04-04"
        else:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        client = AsyncAnthropic(
            api_key=self.api_key,
            default_headers=headers,
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
            mcp_servers=mcp_servers,
        )

        # Construct a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in params:
            tool_dict = self.tool_handler.build_tool_dict(tools)

        if mcp_servers and len(mcp_servers) > 0:
            func_to_call = client.beta.messages.create
        else:
            func_to_call = client.messages.create

        try:
            response = await func_to_call(**params)
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
                mcp_servers=mcp_servers,
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
