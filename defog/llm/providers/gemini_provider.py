import os
import time
from typing import Dict, List, Any, Optional, Callable, Tuple

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..cost import CostCalculator
from ..tools import ToolHandler
from ..utils_function_calling import get_function_specs, convert_tool_choice


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, api_key: Optional[str] = None, config=None):
        super().__init__(api_key or os.getenv("GEMINI_API_KEY"), config=config)
        self.tool_handler = ToolHandler()

    def get_provider_name(self) -> str:
        return "gemini"

    def supports_tools(self, model: str) -> bool:
        return True  # All current Gemini models support tools

    def supports_response_format(self, model: str) -> bool:
        return True  # All current Gemini models support structured output

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

            # Note: Gemini handles parallel tool calling automatically
            # The model decides when to call multiple functions in parallel
            # This is controlled internally and cannot be disabled

        if response_format:
            # If we want a JSON / Pydantic format
            # "response_schema" is only recognized if the google.genai library supports it
            request_params["response_mime_type"] = "application/json"
            request_params["response_schema"] = response_format

        return request_params, messages

    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        messages: List[Dict[str, str]],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        model: str = "",
        post_tool_function: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Extract content (including any tool calls) and usage info from Gemini response.
        Handles chaining of tool calls.
        """

        from google.genai import types

        if len(response.candidates) == 0:
            raise ProviderError(self.get_provider_name(), "No response from Gemini")
        if response.candidates[0].finish_reason == "MAX_TOKENS":
            raise MaxTokensError("Max tokens reached")

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
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tool_call in response.function_calls:
                            func_name = tool_call.name
                            args = tool_call.args
                            # set tool_id to None, as Gemini models do not return a tool_id by default
                            tool_id = getattr(tool_call, "id", None)

                            tool_calls_batch.append(
                                {
                                    "id": tool_id,
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

                        # Try to get text if available
                        try:
                            text = response.text
                        except Exception as e:
                            text = None

                        # Append the tool call content to messages
                        tool_call_content = response.candidates[0].content
                        messages.append(tool_call_content)

                        # Build tool result parts
                        tool_result_parts = []
                        for tool_call, result in zip(response.function_calls, results):
                            func_name = tool_call.name
                            args = tool_call.args
                            tool_id = getattr(tool_call, "id", None)

                            # Store the tool call, result, and text
                            tool_outputs.append(
                                {
                                    "tool_id": tool_id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "text": text,
                                }
                            )

                            tool_result_parts.append(
                                types.Part.from_function_response(
                                    name=func_name,
                                    response={"result": str(result)},
                                )
                            )

                        # Append all tool results in a single message
                        messages.append(
                            types.Content(
                                role="tool",
                                parts=tool_result_parts,
                            )
                        )

                        # Set tool_choice to AUTO so that the next message will be generated normally without required tool calls
                        request_params["automatic_function_calling"] = (
                            types.AutomaticFunctionCallingConfig(disable=False)
                        )
                        request_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="AUTO"
                            )
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

                        # Append error message to messages and retry
                        messages.append(
                            types.Content(
                                role="model",
                                parts=[types.Part.from_text(text=str(e))],
                            )
                        )

                    # Make next call
                    response = await client.aio.models.generate_content(
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
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Gemini."""
        from google import genai
        from google.genai import types

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client = genai.Client(api_key=self.api_key)
        request_params, messages = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Construct a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = self.tool_handler.build_tool_dict(tools)

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
                request_params=request_params,
                messages=messages,
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
