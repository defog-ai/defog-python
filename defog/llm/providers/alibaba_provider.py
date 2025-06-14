import os
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..cost import CostCalculator
from ..tools import ToolHandler
from ..utils_function_calling import get_function_specs, convert_tool_choice


class AlibabaProvider(BaseLLMProvider):
    """Alibaba Cloud (DashScope) provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or os.getenv("ALIBABA_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
            base_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            config=config,
        )
        self.tool_handler = ToolHandler()

    def get_provider_name(self) -> str:
        return "alibaba"

    def supports_tools(self, model: str) -> bool:
        # Most Qwen models support tools
        return True

    def supports_response_format(self, model: str) -> bool:
        # Most Qwen models support structured output
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
        """Build parameters for Alibaba DashScope API."""
        import copy

        messages = copy.deepcopy(messages)

        # Convert messages to DashScope format
        input_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # DashScope uses different role names
            if role == "system":
                input_messages.append({"role": "system", "content": content})
            elif role == "user":
                input_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                input_messages.append({"role": "assistant", "content": content})

        request_params = {
            "model": model,
            "input": {
                "messages": input_messages
            },
            "parameters": {
                "max_tokens": max_completion_tokens,
                "temperature": temperature,
                "seed": seed,
            }
        }

        # Add tools if provided
        if tools and len(tools) > 0:
            function_specs = get_function_specs(tools, model)
            # Convert OpenAI format to DashScope format
            tools_list = []
            for spec in function_specs:
                if spec.get("type") == "function":
                    tools_list.append({
                        "type": "function",
                        "function": spec["function"]
                    })
            
            request_params["parameters"]["tools"] = tools_list
            
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                request_params["parameters"]["tool_choice"] = tool_choice

        # Add response format if provided
        if response_format:
            request_params["parameters"]["result_format"] = "message"

        return request_params, messages

    async def process_response(
        self,
        client,
        response_data,
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
        """Process response from Alibaba DashScope API."""
        if response_data.get("status_code") != 200:
            raise ProviderError(
                self.get_provider_name(), 
                f"API error: {response_data.get('message', 'Unknown error')}"
            )

        output = response_data.get("output", {})
        usage = response_data.get("usage", {})
        
        # Handle tool calls if present
        tool_outputs = []
        choices = output.get("choices", [])
        if not choices:
            raise ProviderError(self.get_provider_name(), "No response choices")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Handle tool calls
        if tools and message.get("tool_calls"):
            tool_calls = message["tool_calls"]
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                if func_name in tool_dict:
                    try:
                        result = await tool_dict[func_name](**args)
                        tool_outputs.append({
                            "tool_call_id": tool_call.get("id", ""),
                            "name": func_name,
                            "args": args,
                            "result": result,
                            "text": content
                        })
                    except Exception as e:
                        raise ProviderError(
                            self.get_provider_name(),
                            f"Tool call failed: {e}",
                            e
                        )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return (
            content,
            tool_outputs,
            input_tokens,
            0,  # cached tokens not supported
            output_tokens,
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
        """Execute a chat completion with Alibaba DashScope."""
        import aiohttp

        if post_tool_function:
            self.tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=request_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    response_data = await response.json()

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
            ) = await self.process_response(
                client=None,
                response_data=response_data,
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