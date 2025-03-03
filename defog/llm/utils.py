import os
import time
import json
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Callable

from defog.llm.utils_function_calling import (
    get_function_specs,
    convert_tool_choice,
    execute_tool,
    execute_tool_async,
    verify_post_tool_function,
)
import inspect
import asyncio

LLM_COSTS_PER_TOKEN = {
    "chatgpt-4o": {"input_cost_per1k": 0.0025, "output_cost_per1k": 0.01},
    "gpt-4o": {
        "input_cost_per1k": 0.0025,
        "cached_input_cost_per1k": 0.00125,
        "output_cost_per1k": 0.01,
    },
    "gpt-4o-mini": {
        "input_cost_per1k": 0.00015,
        "cached_input_cost_per1k": 0.000075,
        "output_cost_per1k": 0.0006,
    },
    "o1": {
        "input_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.0075,
        "output_cost_per1k": 0.06,
    },
    "o1-preview": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.06},
    "o1-mini": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.00055,
        "output_cost_per1k": 0.012,
    },
    "o3-mini": {
        "input_cost_per1k": 0.0011,
        "cached_input_cost_per1k": 0.00055,
        "output_cost_per1k": 0.0044,
    },
    "gpt-4-turbo": {"input_cost_per1k": 0.01, "output_cost_per1k": 0.03},
    "gpt-3.5-turbo": {"input_cost_per1k": 0.0005, "output_cost_per1k": 0.0015},
    "claude-3-5-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-3-5-haiku": {"input_cost_per1k": 0.00025, "output_cost_per1k": 0.00125},
    "claude-3-opus": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.075},
    "claude-3-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-3-haiku": {"input_cost_per1k": 0.00025, "output_cost_per1k": 0.00125},
    "gemini-1.5-pro": {"input_cost_per1k": 0.00125, "output_cost_per1k": 0.005},
    "gemini-1.5-flash": {"input_cost_per1k": 0.000075, "output_cost_per1k": 0.0003},
    "gemini-1.5-flash-8b": {
        "input_cost_per1k": 0.0000375,
        "output_cost_per1k": 0.00015,
    },
    "gemini-2.0-flash": {
        "input_cost_per1k": 0.00010,
        "output_cost_per1k": 0.0004,
    },
    "deepseek-chat": {
        "input_cost_per1k": 0.00027,
        "cached_input_cost_per1k": 0.00007,
        "output_cost_per1k": 0.0011,
    },
    "deepseek-reasoner": {
        "input_cost_per1k": 0.00055,
        "cached_input_cost_per1k": 0.00014,
        "output_cost_per1k": 0.00219,
    },
}


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
    tool_outputs: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.model in LLM_COSTS_PER_TOKEN:
            model_name = self.model
        else:
            # Attempt partial matches if no exact match
            model_name = None
            potential_model_names = []
            for mname in LLM_COSTS_PER_TOKEN.keys():
                if mname in self.model:
                    potential_model_names.append(mname)
            if len(potential_model_names) > 0:
                model_name = max(potential_model_names, key=len)

        if model_name:
            self.cost_in_cents = (
                self.input_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["input_cost_per1k"]
                + self.output_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["output_cost_per1k"]
            ) * 100

            # Add cached input cost if available
            if "cached_input_cost_per1k" in LLM_COSTS_PER_TOKEN[model_name]:
                self.cost_in_cents += (
                    self.cached_input_tokens
                    / 1000
                    * LLM_COSTS_PER_TOKEN[model_name]["cached_input_cost_per1k"]
                ) * 100


#
# --------------------------------------------------------------------------------
# 1) ANTHROPIC
# --------------------------------------------------------------------------------
#


def _build_anthropic_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int = None,
    temperature: float = 0.0,
    stop: List[str] = [],
    tools: List[Callable] = None,
    tool_choice: str = None,
    timeout: int = 100,
    response_format=None,
):
    """Create the parameter dict for Anthropic's .messages.create()."""
    if len(messages) >= 1 and messages[0].get("role") == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        sys_msg = ""

    # Anthropic does not allow `None` as a value for max_completion_tokens
    # Therefore, set it to 8191, which is the max output tokens limit for 3.5 sonnet right now
    if max_completion_tokens is None:
        max_completion_tokens = 8191

    params = {
        "system": sys_msg,
        "messages": messages,
        "model": model,
        "max_tokens": max_completion_tokens,
        "temperature": temperature,
        "stop_sequences": stop,
        "timeout": timeout,
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

    return params, messages  # returning updated messages in case we want them


async def _process_anthropic_response(
    client,
    response,
    request_params: Dict[str, Any],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    is_async: bool,
    response_format=None,
    post_tool_function: Callable = None,
):
    """
    Extract content (including any tool calls) and usage info from Anthropic response.
    Handles chaining of tool calls and structured output parsing.
    """
    from anthropic.types import ToolUseBlock, TextBlock

    if response.stop_reason == "max_tokens":
        raise Exception("Max tokens reached")
    if len(response.content) == 0:
        raise Exception("Max tokens reached")

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
            tool_call_block = next(
                (
                    block
                    for block in response.content
                    if isinstance(block, ToolUseBlock)
                ),
                None,
            )
            text_block = next(
                (block for block in response.content if isinstance(block, TextBlock)),
                None,
            )
            if tool_call_block:
                try:
                    try:
                        func_name = tool_call_block.name
                        args = tool_call_block.input
                        tool_id = tool_call_block.id
                    except Exception as e:
                        raise Exception(f"Error parsing tool call: {e}")

                    tool_to_call = tool_dict.get(func_name)
                    if tool_to_call is None:
                        raise Exception(f"Tool `{func_name}` not found")

                    # Execute tool depending on whether it is async
                    try:
                        if inspect.iscoroutinefunction(tool_to_call):
                            result = await execute_tool_async(tool_to_call, args)
                        else:
                            result = execute_tool(tool_to_call, args)
                    except Exception as e:
                        raise Exception(f"Error executing tool `{func_name}`: {e}")

                    # Execute post-tool function if provided
                    if post_tool_function:
                        try:
                            if inspect.iscoroutinefunction(post_tool_function):
                                await post_tool_function(
                                    function_name=func_name,
                                    input_args=args,
                                    tool_result=result,
                                )
                            else:
                                post_tool_function(
                                    function_name=func_name,
                                    input_args=args,
                                    tool_result=result,
                                )
                        except Exception as e:
                            raise Exception(f"Error executing post_tool_function: {e}")

                    # Reset consecutive_exceptions when tool call is successful
                    consecutive_exceptions = 0

                    # Store the tool call, result, and text
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "name": func_name,
                            "args": args,
                            "result": result,
                            "text": text_block.text if text_block else None,
                        }
                    )

                    # Append the tool call as an assistant response
                    request_params["messages"].append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": func_name,
                                    "input": args,
                                }
                            ],
                        }
                    )

                    # Append the tool result as a user response
                    request_params["messages"].append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": str(result),
                                }
                            ],
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
                    if consecutive_exceptions >= 3:
                        raise Exception(f"Consecutive errors during tool chaining: {e}")

                    print(f"{e}. Retries left: {3 - consecutive_exceptions}")

                    # Append error message to request_params and retry
                    request_params["messages"].append(
                        {
                            "role": "assistant",
                            "content": str(e),
                        }
                    )

                # Make next call
                if is_async:
                    response = await client.messages.create(**request_params)
                else:
                    response = client.messages.create(**request_params)
            else:
                # Break out of loop when tool calls are finished
                content = response.content[0].text
                break
    else:
        # No tools provided
        content = response.content[0].text

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
    return content, tool_outputs, total_input_tokens, total_output_tokens


def _process_anthropic_response_handler(
    client,
    response,
    request_params: Dict[str, Any],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    is_async: bool = False,
    response_format=None,
    post_tool_function: Callable = None,
):
    """
    Processes Anthropic's response by determining whether to execute the response handling
    synchronously or asynchronously. This function acts as a wrapper around _process_anthropic_response,
    deciding the execution mode based on the is_async parameter.

    Parameters:
    - client: The client instance used for communication.
    - response: The response object from Anthropic.
    - request_params: A dictionary of request parameters that resulted in the response.
    - tools: A list of callable tools available for function calling.
    - tool_dict: A dictionary mapping tool names to their callable functions.
    - is_async: A boolean flag indicating whether to execute asynchronously.
    - response_format: Optional format specification for structured output.
    - post_tool_function: Optional function to handle tool responses.

    Returns:
    - The processed response content, input tokens, and output tokens from the response.
    """
    try:
        if is_async:
            return _process_anthropic_response(
                client=client,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                is_async=is_async,
                response_format=response_format,
                post_tool_function=post_tool_function,
            )  # Caller must await this
        else:
            return asyncio.run(
                _process_anthropic_response(
                    client=client,
                    response=response,
                    request_params=request_params,
                    tools=tools,
                    tool_dict=tool_dict,
                    is_async=is_async,
                    response_format=response_format,
                    post_tool_function=post_tool_function,
                )
            )
    except Exception as e:
        raise Exception("Error processing Anthropic response:", e)


def chat_anthropic(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    post_tool_function: Callable = None,
):
    """
    Synchronous Anthropic chat.

    Parameters:
    - messages: A list of dictionaries representing the conversation so far.
    - model: The anthropic model to use for the chat.
    - max_completion_tokens: The maximum number of tokens for the completion.
    - temperature: Ranges from 0.0 to 1.0. Defaults to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks.
    - stop: Custom text sequences that will cause the model to stop generating.
    - response_format: Format specification for structured output (Pydantic model).
        For Anthropic models, this will modify the system message to instruct the model to return valid JSON.
    - seed: NA
    - tools: The list of tools the model may call.
    - tool_choice: Controls which (if any) tool is called by the model.
        "auto": calls 0, 1, or multiple functions,
        "required": calls at least one function,
        "<function_name>": calls only the specified function
    - post_tool_function: An optional function to handle tool responses post-generation. The function takes exactly 3 arguments: function_name, input_args, tool_result

    Returns:
    - LLMResponse which contains the response content, input tokens, output tokens, tools used, and tool outputs
    """
    from anthropic import Anthropic

    if post_tool_function:
        verify_post_tool_function(post_tool_function)

    t = time.time()
    client = Anthropic()

    request_params, _ = _build_anthropic_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

    response = client.messages.create(**request_params)
    content, tool_outputs, input_toks, output_toks = (
        _process_anthropic_response_handler(
            client=client,
            response=response,
            request_params=request_params,
            tools=tools,
            tool_dict=tool_dict,
            is_async=False,
            response_format=response_format,
            post_tool_function=post_tool_function,
        )
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
        tool_outputs=tool_outputs,
    )


async def chat_anthropic_async(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
    post_tool_function: Callable = None,
):
    """
    Asynchronous Anthropic chat.

    Parameters:
    - messages: A list of dictionaries representing the conversation so far.
    - model: The anthropic model to use for the chat.
    - max_completion_tokens: The maximum number of tokens for the completion.
    - temperature: Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks.
    - stop: Custom text sequences that will cause the model to stop generating.
    - response_format: Format specification for structured output (Pydantic model).
        For Anthropic models, this will modify the system message to instruct the model to return valid JSON.
    - seed: NA
    - tools: The list of tools the model may call.
    - tool_choice: Controls which (if any) tool is called by the model.
        "auto": calls 0, 1, or multiple functions,
        "required": calls at least one function,
        "<function_name>": calls only the specified function
    - store: NA
    - metadata: NA
    - timeout: NA
    - prediction: NA
    - reasoning_effort: NA
    - post_tool_function: An optional function to handle tool responses post-generation. The function takes exactly 3 arguments: function_name, input_args, tool_result

    Returns:
    - LLMResponse which contains the response content, input tokens, output tokens, tools used, and tool outputs
    """
    from anthropic import AsyncAnthropic

    if post_tool_function:
        verify_post_tool_function(post_tool_function)

    t = time.time()
    client = AsyncAnthropic()
    params, _ = _build_anthropic_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in params:
        tool_dict = {tool.__name__: tool for tool in tools}

    response = await client.messages.create(**params)
    content, tool_outputs, input_toks, output_toks = (
        await _process_anthropic_response_handler(
            client=client,
            response=response,
            request_params=params,
            tools=tools,
            tool_dict=tool_dict,
            is_async=True,
            response_format=response_format,
            post_tool_function=post_tool_function,
        )
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
        tool_outputs=tool_outputs,
    )


#
# --------------------------------------------------------------------------------
# 2) OPENAI
# --------------------------------------------------------------------------------
#


def _build_openai_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    prediction: Dict[str, str] = None,
    reasoning_effort: str = None,
    store: bool = True,
    metadata: Dict[str, str] = None,
    timeout: int = 100,
):
    """
    Build the parameter dictionary for OpenAI's chat.completions.create().
    Also handles special logic for o1-mini, o1-preview, deepseek-chat, etc.
    """
    # Potentially move system message to user message for certain model families:
    if model in [
        "o1-mini",
        "o1-preview",
        "o1",
        "deepseek-chat",
        "deepseek-reasoner",
        "o3-mini",
    ]:
        sys_msg = None
        for i in range(len(messages)):
            if messages[i].get("role") == "system":
                sys_msg = messages.pop(i)["content"]
                break
        if sys_msg:
            for i in range(len(messages)):
                if messages[i].get("role") == "user":
                    messages[i]["content"] = sys_msg + "\n" + messages[i]["content"]
                    break

    request_params = {
        "messages": messages,
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "store": store,
        "metadata": metadata,
        "timeout": timeout,
    }

    # Tools are only supported for certain models
    if (
        tools
        and len(tools) > 0
        and model
        not in [
            "o1-mini",
            "o1-preview",
            "deepseek-chat",
            "deepseek-reasoner",
        ]
    ):
        function_specs = get_function_specs(tools, model)
        request_params["tools"] = function_specs
        if tool_choice:
            tool_names_list = [func.__name__ for func in tools]
            tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
            request_params["tool_choice"] = tool_choice
        else:
            request_params["tool_choice"] = "auto"

        # only set parallel_tool_calls for gpt-4o based models
        # not supported in o models
        if model in ["gpt-4o", "gpt-4o-mini"]:
            request_params["parallel_tool_calls"] = False

    # Some models do not allow temperature or response_format:
    if model.startswith("o") or model == "deepseek-reasoner":
        request_params.pop("temperature", None)
    if model in ["o1-mini", "o1-preview", "deepseek-chat", "deepseek-reasoner"]:
        request_params.pop("response_format", None)

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
        # cannot have stop when using response_format
        request_params.pop("stop", None)

    return request_params


async def _process_openai_response(
    client,
    response,
    request_params: Dict[str, Any],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    response_format,
    model: str,
    is_async: bool,
    post_tool_function: Callable = None,
):
    """
    Extract content (including any tool calls) and usage info from OpenAI response.
    Handles chaining of tool calls.
    """
    if len(response.choices) == 0:
        raise Exception("No response from OpenAI.")
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")

    # If we have tools, handle dynamic chaining:
    tool_outputs = []
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    if tools and len(tools) > 0:
        consecutive_exceptions = 0
        while True:
            total_input_tokens += (
                response.usage.prompt_tokens
                - response.usage.prompt_tokens_details.cached_tokens
            )
            total_cached_input_tokens += (
                response.usage.prompt_tokens_details.cached_tokens
            )
            total_output_tokens += response.usage.completion_tokens
            message = response.choices[0].message
            if message.tool_calls:
                try:
                    tool_call = message.tool_calls[0]
                    func_name = tool_call.function.name
                    tool_call_id = tool_call.id
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_to_call = tool_dict.get(func_name)
                    if tool_to_call is None:
                        raise Exception(f"Tool `{func_name}` not found")

                    # Execute tool depending on whether it is async
                    try:
                        if inspect.iscoroutinefunction(tool_to_call):
                            result = await execute_tool_async(tool_to_call, args)
                        else:
                            result = execute_tool(tool_to_call, args)
                    except Exception as e:
                        raise Exception(f"Error executing tool `{func_name}`: {e}")

                    # Execute post-tool function if provided
                    if post_tool_function:
                        try:
                            if inspect.iscoroutinefunction(post_tool_function):
                                await post_tool_function(
                                    function_name=func_name,
                                    input_args=args,
                                    tool_result=result,
                                )
                            else:
                                post_tool_function(
                                    function_name=func_name,
                                    input_args=args,
                                    tool_result=result,
                                )
                        except Exception as e:
                            raise Exception(f"Error executing post_tool_function: {e}")

                    # Reset consecutive_exceptions when tool call is successful
                    consecutive_exceptions = 0

                    # Store the tool call, result, and text
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_call_id,
                            "name": func_name,
                            "args": args,
                            "result": result,
                            "text": message.content if message.content else None,
                        }
                    )

                    # Append the tool calls as an assistant response
                    request_params["messages"].append(
                        {
                            "role": "assistant",
                            "tool_calls": message.tool_calls,
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
                    if consecutive_exceptions >= 3:
                        raise Exception(f"Consecutive errors during tool chaining: {e}")

                    print(f"{e}. Retries left: {3 - consecutive_exceptions}")

                    # Append error message to request_params and retry
                    request_params["messages"].append(
                        {
                            "role": "assistant",
                            "content": str(e),
                        }
                    )

                # Make next call
                if is_async:
                    response = await client.chat.completions.create(**request_params)
                else:
                    response = client.chat.completions.create(**request_params)
            else:
                # Break out of loop when tool calls are finished
                content = message.content
                break
    else:
        # No tools provided
        if response_format and model not in ["o1-mini", "o1-preview"]:
            content = response.choices[0].message.parsed
        else:
            content = response.choices[0].message.content

    usage = response.usage
    total_cached_input_tokens += usage.prompt_tokens_details.cached_tokens
    total_input_tokens += (
        usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens
    )
    total_output_tokens += usage.completion_tokens
    return (
        content,
        tool_outputs,
        total_input_tokens,
        total_cached_input_tokens,
        total_output_tokens,
        usage.completion_tokens_details,
    )


def _process_openai_response_handler(
    client,
    response,
    request_params: Dict[str, Any],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    response_format,
    model: str,
    is_async: bool = False,
    post_tool_function: Callable = None,
):
    """
    Processes OpenAI's response by determining whether to execute the response handling
    synchronously or asynchronously. This function acts as a wrapper around _process_openai_response,
    deciding the execution mode based on the is_async parameter.

    Parameters:
    - client: The client instance used for communication.
    - response: The response object from OpenAI.
    - request_params: A dictionary of request parameters that resulted in the response.
    - tools: A list of callable tools available for function calling.
    - tool_dict: A dictionary mapping tool names to their callable functions.
    - is_async: A boolean flag indicating whether to execute asynchronously.

    Returns:
    - The processed response content, input tokens, cached input tokens and output tokens from the response.
    """
    try:
        if is_async:
            return _process_openai_response(
                client=client,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                is_async=is_async,
                post_tool_function=post_tool_function,
            )  # Caller must await this
        else:
            return asyncio.run(
                _process_openai_response(
                    client=client,
                    response=response,
                    request_params=request_params,
                    tools=tools,
                    tool_dict=tool_dict,
                    response_format=response_format,
                    model=model,
                    is_async=is_async,
                    post_tool_function=post_tool_function,
                )
            )

    except Exception as e:
        raise Exception("Error processing OpenAI response:", e)


def chat_openai(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    base_url: str = "https://api.openai.com/v1/",
    api_key: str = os.environ.get("OPENAI_API_KEY", ""),
    prediction: Dict[str, str] = None,
    reasoning_effort: str = None,
    store: bool = True,
    metadata: Dict[str, str] = None,
    timeout: int = 100,
    post_tool_function: Callable = None,
):
    """
    Synchronous OpenAI chat.

    Parameters:
    - messages: The list of messages to send to the LLM.
    - model: The OpenAI model to use for the chat.
    - max_completion_tokens: The maximum number of tokens to return in the response.
    - temperature: Between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - stop: Up to 4 sequences where the API will stop generating further tokens.
    - response_format: The format that the model must output.
    - seed: If specified, OpenAI will try their best to sample deterministically
    - tools: The list of tools the model may call.
    - tool_choice: Controls which (if any) tool is called by the model.
        "auto": calls 0, 1, or multiple functions,
        "required": calls at least one function,
        "<function_name>": calls only the specified function
    - base_url: The base URL to use for the chat.
    - api_key: The OpenAI API key
    - prediction: Configuration for a Predicted Output.
    - reasoning_effort: "low", "medium", or "high". Only for o1 and o3 models
    - store: Whether or not to store the output of this chat completion request for use in model distillation or evals products.
    - metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
    - timeout: No. of seconds before the request times out.
    - post_tool_function: An optional function to handle tool responses post-generation. The function takes exactly 3 arguments: function_name, input_args, tool_result

    Returns:
    - LLMResponse which contains the response content, input tokens, cached input tokens, output tokens, output token details, tools used, and tool outputs
    """
    from openai import OpenAI

    if post_tool_function:
        verify_post_tool_function(post_tool_function)

    t = time.time()
    client_openai = OpenAI(base_url=base_url, api_key=api_key)
    request_params = _build_openai_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        seed=seed,
        tools=tools,
        tool_choice=tool_choice,
        store=store,
        metadata=metadata,
        timeout=timeout,
        prediction=prediction,
        reasoning_effort=reasoning_effort,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

    # If response_format is set, we do parse
    if request_params.get("response_format"):
        response = client_openai.beta.chat.completions.parse(**request_params)
    else:
        response = client_openai.chat.completions.create(**request_params)

    (
        content,
        tool_outputs,
        input_tokens,
        cached_input_tokens,
        output_tokens,
        completion_token_details,
    ) = _process_openai_response_handler(
        client=client_openai,
        response=response,
        request_params=request_params,
        tools=tools,
        tool_dict=tool_dict,
        response_format=response_format,
        model=model,
        is_async=False,
        post_tool_function=post_tool_function,
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        output_tokens_details=completion_token_details,
        tool_outputs=tool_outputs,
    )


async def chat_openai_async(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    store: bool = True,
    metadata: Dict[str, str] = None,
    timeout: int = 100,
    base_url: str = "https://api.openai.com/v1/",
    api_key: str = os.environ.get("OPENAI_API_KEY", ""),
    prediction: Dict[str, str] = None,
    reasoning_effort: str = None,
    post_tool_function: Callable = None,
):
    """
    Asynchronous OpenAI chat.

    Parameters:
    - messages: The list of messages to send to the LLM.
    - model: The OpenAI model to use for the chat.
    - max_completion_tokens: The maximum number of tokens to return in the response.
    - temperature: Between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - stop: Up to 4 sequences where the API will stop generating further tokens.
    - response_format: The format that the model must output.
    - seed: If specified, OpenAI will try their best to sample deterministically
    - tools: The list of tools the model may call.
    - tool_choice: Controls which (if any) tool is called by the model.
        "auto": calls 0, 1, or multiple functions,
        "required": calls at least one function,
        "<function_name>": calls only the specified function
    - base_url: The base URL to use for the chat.
    - api_key: The OpenAI API key
    - prediction: Configuration for a Predicted Output.
    - reasoning_effort: "low", "medium", or "high". Only for o1 and o3 models
    - store: Whether or not to store the output of this chat completion request for use in model distillation or evals products.
    - metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
    - timeout: No. of seconds before the request times out.
    - post_tool_function: An optional function to handle tool responses post-generation. The function takes exactly 3 arguments: function_name, input_args, tool_result

    Returns:
    - LLMResponse which contains the response content, input tokens, cached input tokens, output tokens, output token details, tools used, and tool outputs
    """
    from openai import AsyncOpenAI

    t = time.time()
    client_openai = AsyncOpenAI(base_url=base_url, api_key=api_key)
    request_params = _build_openai_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
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
        tool_dict = {tool.__name__: tool for tool in tools}

    # If response_format is set, we do parse
    if request_params.get("response_format"):
        response = await client_openai.beta.chat.completions.parse(**request_params)
    else:
        response = await client_openai.chat.completions.create(**request_params)

    (
        content,
        tool_outputs,
        input_tokens,
        cached_input_tokens,
        output_tokens,
        completion_token_details,
    ) = await _process_openai_response_handler(
        client=client_openai,
        response=response,
        request_params=request_params,
        tools=tools,
        tool_dict=tool_dict,
        response_format=response_format,
        model=model,
        is_async=True,
        post_tool_function=post_tool_function,
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        output_tokens_details=completion_token_details,
        tool_outputs=tool_outputs,
    )


#
# --------------------------------------------------------------------------------
# 3) TOGETHER
# --------------------------------------------------------------------------------
#


def _build_together_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    seed: int = 0,
):
    return {
        "messages": messages,
        "model": model,
        "max_tokens": max_completion_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
    }


def _process_together_response(response):
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        raise Exception("Max tokens reached")
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def chat_together(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools=None,
    tool_choice=None,
    post_tool_function: Callable = None,
):
    """Synchronous Together chat."""
    from together import Together

    t = time.time()
    client_together = Together()
    params = _build_together_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
    )
    response = client_together.chat.completions.create(**params)

    content, input_toks, output_toks = _process_together_response(response)
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
    )


async def chat_together_async(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools=None,
    tool_choice=None,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
    post_tool_function: Callable = None,
):
    """Asynchronous Together chat."""
    from together import AsyncTogether

    t = time.time()
    client_together = AsyncTogether(timeout=timeout)
    params = _build_together_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
    )
    response = await client_together.chat.completions.create(**params)

    content, input_toks, output_toks = _process_together_response(response)
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
    )


#
# --------------------------------------------------------------------------------
# 4) GEMINI
# --------------------------------------------------------------------------------
#


def _build_gemini_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
):
    """Construct parameters for Gemini's generate_content call."""

    from google.genai import types

    if messages[0]["role"] == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        system_msg = None

    # Combine all user/assistant messages into one string and create a types.Content object
    messages_str = "\n".join([m["content"] for m in messages])
    user_prompt_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=messages_str)],
    )
    messages = [user_prompt_content]
    request_params = {
        "temperature": temperature,
        "system_instruction": system_msg,
        "max_output_tokens": max_completion_tokens,
        "stop_sequences": stop,
    }

    if tools:
        function_specs = get_function_specs(tools, model)
        request_params["tools"] = function_specs

        # Set up automatic_function_calling and tool_config based on tool_choice
        if tool_choice:
            tool_names_list = [func.__name__ for func in tools]
            tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
        if tool_choice:
            request_params["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
            request_params["tool_config"] = tool_choice

    if response_format:
        # If we want a JSON / Pydantic format
        # "response_schema" is only recognized if the google.genai library supports it
        request_params["response_mime_type"] = "application/json"
        request_params["response_schema"] = response_format

    return messages, request_params


async def _process_gemini_response(
    client,
    response,
    request_params: Dict[str, Any],
    messages: List[Dict[str, str]],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    response_format,
    model: str,
    is_async: bool,
    post_tool_function: Callable = None,
):
    """Extract content (including any tool calls) and usage info from Gemini response.
    Handles chaining of tool calls.
    """

    from google.genai import types

    if len(response.candidates) == 0:
        raise Exception("No response from Gemini.")
    if response.candidates[0].finish_reason == "MAX_TOKENS":
        raise Exception("Max tokens reached")

    # If we have tools, handle dynamic chaining:
    tool_outputs = []
    total_input_tokens = 0
    total_output_tokens = 0
    if tools and len(tools) > 0:
        consecutive_exceptions = 0
        while True:
            total_input_tokens += response.usage_metadata.prompt_token_count
            total_output_tokens += response.usage_metadata.candidates_token_count
            if response.function_calls:
                try:
                    tool_call = response.function_calls[0]
                    func_name = tool_call.name
                    args = tool_call.args

                    # set tool_id to None, as Gemini models do not return a tool_id by default
                    # strangely, tool_call.id exists, but is always set to `None`
                    tool_id = None

                    tool_to_call = tool_dict.get(func_name)
                    if tool_to_call is None:
                        raise Exception(f"Tool `{func_name}` not found")

                    # Execute tool depending on whether it is async
                    try:
                        if inspect.iscoroutinefunction(tool_to_call):
                            result = await execute_tool_async(tool_to_call, args)
                        else:
                            result = execute_tool(tool_to_call, args)
                    except Exception as e:
                        raise Exception(f"Error executing tool `{func_name}`: {e}")

                    # Execute post-tool function if provided
                    if post_tool_function:
                        try:
                            if inspect.iscoroutinefunction(post_tool_function):
                                await post_tool_function(
                                    function_name=func_name,
                                    input_args=args,
                                    tool_result=result,
                                )
                            else:
                                post_tool_function(
                                    function_name=func_name,
                                    input_args=args,
                                    tool_result=result,
                                )
                        except Exception as e:
                            raise Exception(f"Error executing post_tool_function: {e}")

                    # Reset consecutive_exceptions when tool call is successful
                    consecutive_exceptions = 0

                    # Store the tool call, result, and text
                    try:
                        text = response.text
                    except Exception as e:
                        text = None
                    tool_outputs.append(
                        {
                            "tool_id": tool_id,
                            "name": func_name,
                            "args": args,
                            "result": result,
                            "text": text,
                        }
                    )

                    # Append the tool call content to messages
                    tool_call_content = response.candidates[0].content
                    messages.append(tool_call_content)

                    # Append the tool result to messages
                    messages.append(
                        types.Content(
                            role="tool",
                            parts=[
                                types.Part.from_function_response(
                                    name=func_name,
                                    response={"result": str(result)},
                                )
                            ],
                        )
                    )

                    # Set tool_choice to AUTO so that the next message will be generated normally without required tool calls
                    request_params["automatic_function_calling"] = (
                        types.AutomaticFunctionCallingConfig(disable=False)
                    )
                    request_params["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    )
                except Exception as e:
                    consecutive_exceptions += 1

                    # Break the loop if consecutive exceptions exceed the threshold
                    if consecutive_exceptions >= 3:
                        raise Exception(f"Consecutive errors during tool chaining: {e}")

                    print(f"{e}. Retries left: {3 - consecutive_exceptions}")

                    # Append error message to messages and retry
                    messages.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=str(e))],
                        )
                    )

                # Make next call
                if is_async:
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=messages,
                        config=types.GenerateContentConfig(**request_params),
                    )
                else:
                    response = client.models.generate_content(
                        model=model,
                        contents=messages,
                        config=types.GenerateContentConfig(**request_params),
                    )
            else:
                # Break out of loop when tool calls are finished
                content = response.text.strip() if response.text else None
                break
    else:
        # No tools provided
        if response_format:
            # Attempt to parse with pydantic model
            content = response_format.model_validate_json(response.text)
        else:
            content = response.text.strip() if response.text else None

    usage = response.usage_metadata
    total_input_tokens += usage.prompt_token_count
    total_output_tokens += usage.candidates_token_count
    return (
        content,
        tool_outputs,
        total_input_tokens,
        total_output_tokens,
    )


def _process_gemini_response_handler(
    client,
    response,
    request_params: Dict[str, Any],
    messages: List,
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    response_format,
    model: str,
    is_async=False,
    post_tool_function: Callable = None,
):
    """
    Processes Gemini's response by determining whether to execute the response handling
    synchronously or asynchronously. This function acts as a wrapper around _process_gemini_response,
    deciding the execution mode based on the is_async parameter.

    Parameters:
    - client: The client instance used for communication.
    - response: The response object from Gemini.
    - request_params: A dictionary of request parameters that resulted in the response.
    - tools: A list of callable tools available for function calling.
    - tool_dict: A dictionary mapping tool names to their callable functions.
    - is_async: A boolean flag indicating whether to execute asynchronously.

    Returns:
    - The processed response content, input tokens, and output tokens from the response.
    """
    try:
        if is_async:
            return _process_gemini_response(
                client=client,
                response=response,
                request_params=request_params,
                messages=messages,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                is_async=is_async,
                post_tool_function=post_tool_function,
            )  # Caller must await this
        else:
            return asyncio.run(
                _process_gemini_response(
                    client=client,
                    response=response,
                    request_params=request_params,
                    messages=messages,
                    tools=tools,
                    tool_dict=tool_dict,
                    response_format=response_format,
                    model=model,
                    is_async=is_async,
                    post_tool_function=post_tool_function,
                )
            )

    except Exception as e:
        raise Exception("Error processing Gemini response:", e)


def chat_gemini(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    post_tool_function: Callable = None,
):
    """
    Synchronous Gemini chat.

    Parameters:
    - messages: The list of messages to send to the LLM.
    - model: The Gemini model to use for the chat.
    - max_completion_tokens: The maximum number of tokens to return in the response.
    - temperature: Higher values will make the output more random, while lower values like will make it more focused and deterministic.
        Between 0 to 2: gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro-002, gemini-2.0-flash
        Between 0 to 1: gemini-1.0-pro-vision, gemini-1.0-pro-001
    - stop: List of strings that will stop the model from generating further tokens.
    - response_format: The format that the model must output. See https://ai.google.dev/gemini-api/docs/structured-output?lang=python
    - seed: NA
    - tools: The list of tools the model may call.
    - tool_choice: Controls which (if any) tool is called by the model.
        "auto": calls 0, 1, or multiple functions,
        "required": calls at least one function,
        "<function_name>": calls only the specified function
    - post_tool_function: An optional function to handle tool responses post-generation. The function takes exactly 3 arguments: function_name, input_args, tool_result

    Returns:
    - LLMResponse which contains the response content, input tokens, output tokens, tools used, and tool outputs
    """
    from google import genai
    from google.genai import types

    if post_tool_function:
        verify_post_tool_function(post_tool_function)

    t = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    messages, request_params = _build_gemini_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        seed=seed,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

    try:
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(**request_params),
        )
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    content, tool_outputs, input_toks, output_toks = _process_gemini_response_handler(
        client=client,
        response=response,
        request_params=request_params,
        tools=tools,
        tool_dict=tool_dict,
        response_format=response_format,
        messages=messages,
        model=model,
        is_async=False,
        post_tool_function=post_tool_function,
    )
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
        tool_outputs=tool_outputs,
    )


async def chat_gemini_async(
    messages: List[Dict[str, str]],
    max_completion_tokens: int = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: str = None,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
    post_tool_function: Callable = None,
):
    """
    Asynchronous Gemini chat.

    Parameters:
    - messages: The list of messages to send to the LLM.
    - model: The Gemini model to use for the chat.
    - max_completion_tokens: The maximum number of tokens to return in the response.
    - temperature: Higher values will make the output more random, while lower values like will make it more focused and deterministic.
        Between 0 to 2: gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro-002, gemini-2.0-flash
        Between 0 to 1: gemini-1.0-pro-vision, gemini-1.0-pro-001
    - stop: List of strings that will stop the model from generating further tokens.
    - response_format: The format that the model must output. See https://ai.google.dev/gemini-api/docs/structured-output?lang=python
    - seed: NA
    - tools: The list of tools the model may call.
    - tool_choice: Controls which (if any) tool is called by the model.
        "auto": calls 0, 1, or multiple functions,
        "required": calls at least one function,
        "<function_name>": calls only the specified function
    - store: NA
    - metadata: NA
    - timeout: NA
    - prediction: NA
    - reasoning_effort: NA
    - post_tool_function: An optional function to handle tool responses post-generation. The function takes exactly 3 arguments: function_name, input_args, tool_result

    Returns:
    - LLMResponse which contains the response content, input tokens, output tokens, tools used, and tool outputs
    """
    from google import genai
    from google.genai import types

    t = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    messages, request_params = _build_gemini_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        seed=seed,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

        # Set up automatic_function_calling and tool_config based on tool_choice
        if tool_choice:
            tool_names_list = [func.__name__ for func in tools]
            tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
        if tool_choice:
            request_params["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
            request_params["tool_config"] = tool_choice

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(**request_params),
        )
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    content, tool_outputs, input_toks, output_toks = (
        await _process_gemini_response_handler(
            client=client,
            response=response,
            request_params=request_params,
            messages=messages,
            tools=tools,
            tool_dict=tool_dict,
            response_format=response_format,
            model=model,
            is_async=True,
            post_tool_function=post_tool_function,
        )
    )
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
        tool_outputs=tool_outputs,
    )


def map_model_to_chat_fn_async(model: str) -> Callable:
    """
    Returns the appropriate chat function based on the model.
    """
    if model.startswith("claude"):
        return chat_anthropic_async
    if model.startswith("gemini"):
        return chat_gemini_async
    if (
        model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("chatgpt")
        or model.startswith("o3")
    ):
        return chat_openai_async
    if model.startswith("deepseek"):
        return chat_openai_async
    if (
        model.startswith("meta-llama")
        or model.startswith("mistralai")
        or model.startswith("Qwen")
    ):
        return chat_together_async
    raise ValueError(f"Unknown model: {model}")


async def chat_async(
    model,
    messages,
    max_completion_tokens=None,
    temperature=0.0,
    stop=[],
    response_format=None,
    seed=0,
    store=True,
    metadata=None,
    timeout=100,  # in seconds
    backup_model=None,
    prediction=None,
    reasoning_effort=None,
    tools=None,
    tool_choice=None,
    max_retries=3,
    post_tool_function: Callable = None,
) -> LLMResponse:
    """
    Returns the response from the LLM API for a single model that is passed in.
    Includes retry logic with exponential backoff for up to 3 attempts.
    """
    llm_function = map_model_to_chat_fn_async(model)
    base_delay = 1  # Initial delay in seconds

    if post_tool_function:
        verify_post_tool_function(post_tool_function)

    for attempt in range(max_retries):
        try:
            if attempt > 0 and backup_model is not None:
                # For the first attempt, use the original model
                # For subsequent attempts, use the backup model if it is provided
                model = backup_model
                llm_function = map_model_to_chat_fn_async(model)
            if not model.startswith("deepseek"):
                return await llm_function(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    stop=stop,
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
                )
            else:
                if not os.getenv("DEEPSEEK_API_KEY"):
                    raise Exception("DEEPSEEK_API_KEY is not set")
                return await llm_function(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    stop=stop,
                    response_format=response_format,
                    seed=seed,
                    store=store,
                    metadata=metadata,
                    timeout=timeout,
                    prediction=prediction,
                    reasoning_effort=reasoning_effort,
                    base_url="https://api.deepseek.com",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    post_tool_function=post_tool_function,
                )
        except Exception as e:
            delay = base_delay * (2**attempt)  # Exponential backoff
            print(
                f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...",
                flush=True,
            )
            print(f"Error: {e}", flush=True)
            error_trace = traceback.format_exc()
            await asyncio.sleep(delay)

    # If we get here, all attempts failed
    raise Exception(
        "All attempts at calling the chat_async function failed. The latest error traceback was: ",
        error_trace,
    )
