import inspect
import asyncio
from typing import Dict, List, Callable, Any, Optional
from ..exceptions import ToolError
from ..utils_function_calling import (
    execute_tool,
    execute_tool_async,
    execute_tools_parallel,
    verify_post_tool_function,
)


class ToolHandler:
    """Handles tool calling logic for LLM providers."""

    def __init__(self, max_consecutive_errors: int = 3):
        self.max_consecutive_errors = max_consecutive_errors

    async def execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_dict: Dict[str, Callable],
        post_tool_function: Optional[Callable] = None,
    ) -> Any:
        """Execute a single tool call."""
        tool_to_call = tool_dict.get(tool_name)
        if tool_to_call is None:
            raise ToolError(tool_name, "Tool not found")

        try:
            # Execute tool depending on whether it is async
            if inspect.iscoroutinefunction(tool_to_call):
                result = await execute_tool_async(tool_to_call, args)
            else:
                result = execute_tool(tool_to_call, args)
        except Exception as e:
            raise ToolError(tool_name, f"Error executing tool: {e}", e)

        # Execute post-tool function if provided
        if post_tool_function:
            try:
                if inspect.iscoroutinefunction(post_tool_function):
                    await post_tool_function(
                        function_name=tool_name,
                        input_args=args,
                        tool_result=result,
                    )
                else:
                    post_tool_function(
                        function_name=tool_name,
                        input_args=args,
                        tool_result=result,
                    )
            except Exception as e:
                raise ToolError(
                    tool_name, f"Error executing post_tool_function: {e}", e
                )

        return result

    async def execute_tool_calls_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_dict: Dict[str, Callable],
        enable_parallel: bool = False,
        post_tool_function: Optional[Callable] = None,
    ) -> List[Any]:
        """Execute multiple tool calls either in parallel or sequentially."""
        try:
            results = await execute_tools_parallel(
                tool_calls, tool_dict, enable_parallel
            )

            # Execute post-tool function for each result if provided
            if post_tool_function:
                for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
                    func_name = tool_call.get("function", {}).get(
                        "name"
                    ) or tool_call.get("name")
                    func_args = tool_call.get("function", {}).get(
                        "arguments"
                    ) or tool_call.get("arguments", {})

                    try:
                        if inspect.iscoroutinefunction(post_tool_function):
                            await post_tool_function(
                                function_name=func_name,
                                input_args=func_args,
                                tool_result=result,
                            )
                        else:
                            post_tool_function(
                                function_name=func_name,
                                input_args=func_args,
                                tool_result=result,
                            )
                    except Exception as e:
                        # Don't fail the entire batch for post-tool function errors
                        print(
                            f"Warning: Error executing post_tool_function for {func_name}: {e}"
                        )

            return results
        except Exception as e:
            raise ToolError("batch", f"Error executing tool batch: {e}", e)

    def build_tool_dict(self, tools: List[Callable]) -> Dict[str, Callable]:
        """Build a dictionary mapping tool names to functions."""
        return {tool.__name__: tool for tool in tools}

    def validate_post_tool_function(
        self, post_tool_function: Optional[Callable]
    ) -> None:
        """Validate the post-tool function signature."""
        if post_tool_function:
            verify_post_tool_function(post_tool_function)
