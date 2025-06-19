import inspect
import jsonref
import asyncio
from typing import Callable, List, Dict, Any, Union
from pydantic import BaseModel
from defog.llm.models import OpenAIFunctionSpecs, AnthropicFunctionSpecs


def cleanup_obj(obj: dict, model: str):
    """
    Converts a pydantic model's json to a format that gemini supports (recursively):
    - Converts properties called "anyOf" to "any_of"
    - Converts all "types" to uppercase
    - Removes "$defs" properties that are created by nested pydantic models
    - Removes $ref properties that are also created by nested pydantic models
    """
    new_obj = obj

    keys = new_obj.keys()
    if "anyOf" in new_obj:
        new_obj["any_of"] = new_obj["anyOf"]
        del new_obj["anyOf"]

    if "type" in new_obj:
        if model.startswith("gemini"):
            new_obj["type"] = new_obj["type"].upper()
        else:
            new_obj["type"] = new_obj["type"]

    if "$defs" in new_obj:
        del new_obj["$defs"]

    if "$ref" in new_obj:
        del new_obj["$ref"]

    for k in keys:
        if isinstance(new_obj[k], dict):
            new_obj[k] = cleanup_obj(new_obj[k], model)

    return new_obj


def get_function_specs(
    functions: List[Callable], model: str
) -> List[Union[OpenAIFunctionSpecs, AnthropicFunctionSpecs]]:
    """Return a list of dictionaries describing each function's name, docstring, and input schema."""
    function_specs = []

    for func in functions:
        # Get docstring
        docstring = inspect.getdoc(func) or ""

        # Get the function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # We assume each function has exactly one parameter that is a Pydantic model
        if len(params) != 1:
            raise ValueError(
                f"Function {func.__name__} does not have exactly one parameter."
            )

        param = params[0]
        model_class = param.annotation  # The Pydantic model

        # Safety check to ensure param.annotation is indeed a Pydantic BaseModel
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise ValueError(
                f"Parameter for function {func.__name__} is not a Pydantic model."
            )

        # Get the JSON schema from the model
        input_schema: dict = model_class.model_json_schema()

        # if there are references (due to nested pydantic models), dereference them
        # these show up as "$defs"
        # need proxies=False to replace the refs with actual objects and not just JsonRef instances
        input_schema = jsonref.replace_refs(input_schema, proxies=False)

        # cleanup object
        input_schema = cleanup_obj(input_schema, model)

        # Remove title from input_schema
        input_schema.pop("title")
        # Remove default and title values from input_schema["properties"]
        for prop in input_schema["properties"].values():
            keys_to_remove = [k for k in prop if k in ["default", "title"]]
            for k in keys_to_remove:
                prop.pop(k)

        if (
            model.startswith("gpt")
            or model.startswith("o1")
            or model.startswith("chatgpt")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("deepseek")
        ):
            function_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": docstring,
                        "parameters": input_schema,
                    },
                }
            )
        elif model.startswith("claude"):
            input_schema["type"] = "object"
            function_specs.append(
                {
                    "name": func.__name__,
                    "description": docstring,
                    "input_schema": input_schema,
                }
            )
        elif model.startswith("gemini"):
            from google.genai import types

            func_spec = {
                "name": func.__name__,
                "description": docstring,
                "parameters": input_schema,
            }

            function_declaration = types.FunctionDeclaration(**func_spec)
            tool = types.Tool(function_declarations=[function_declaration])

            function_specs.append(tool)
        elif model.startswith("mistral"):
            function_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": docstring,
                        "parameters": input_schema,
                    },
                }
            )
        else:
            raise ValueError(f"Model does not support function calling: {model}")

    return function_specs


def convert_tool_choice(tool_choice: str, tool_name_list: List[str], model: str):
    """
    Convert a tool choice to a function calling tool choice that is compatible with the model.
    """
    model_map = {
        "openai": {
            "prefixes": ["gpt", "o1", "chatgpt", "o3", "o4", "deepseek"],
            "choices": {
                "auto": "auto",
                "required": "required",
                "any": "required",
                "none": "none",
            },
            "custom": {"type": "function", "function": {"name": tool_choice}},
        },
        "anthropic": {
            "prefixes": ["claude"],
            "choices": {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "any": {"type": "any"},
            },
            "custom": {"type": "tool", "name": tool_choice},
        },
        "gemini": {"prefixes": ["gemini"]},
        "mistral": {
            "prefixes": ["mistral"],
            "choices": {
                "auto": "auto",
                "required": "any",
                "any": "any",
                "none": "none",
            },
            "custom": tool_choice,
        },
    }

    for model_type, config in model_map.items():
        if any(model.startswith(prefix) for prefix in config["prefixes"]):
            if model_type == "gemini":
                from google.genai import types

                config = {
                    "choices": {
                        "auto": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="AUTO"
                            )
                        ),
                        "required": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY"
                            )
                        ),
                        "any": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY"
                            )
                        ),
                        "none": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="NONE"
                            )
                        ),
                    },
                    "custom": types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=[tool_choice]
                        )
                    ),
                }
            if tool_choice not in config["choices"]:
                # Validate custom tool_choice
                if tool_choice not in tool_name_list:
                    raise ValueError(
                        f"Forced function `{tool_choice}` is not in the list of provided tools"
                    )
                return config["custom"]
            return config["choices"][tool_choice]

    raise ValueError(f"Model `{model}` does not support tools and function calling")


def execute_tool(tool: Callable, inputs: Dict[str, Any]):
    """
    Execute a tool function with the given inputs.
    The inputs are raw values that are expacted to match the function's parameter schema.
    However, the function only takes a Pydantic model as input, so we need to convert the inputs to a Pydantic model.
    We use the tool's function signature to get the Pydantic model class.
    """
    # Get the function signature
    sig = inspect.signature(tool)
    # Get the Pydantic model class from the function signature
    model_class = sig.parameters["input"].annotation
    # Convert the inputs to a Pydantic model
    model = model_class.model_validate(inputs)
    # Call the tool function with the Pydantic model
    return tool(model)


async def execute_tool_async(tool: Callable, inputs: Dict[str, Any]):
    """
    Execute a tool function with the given inputs.
    The inputs are raw values that are expacted to match the function's parameter schema.
    However, the function only takes a Pydantic model as input, so we need to convert the inputs to a Pydantic model.
    We use the tool's function signature to get the Pydantic model class.
    """
    # Get the function signature
    sig = inspect.signature(tool)
    # Get the Pydantic model class from the function signature
    model_class = sig.parameters["input"].annotation
    # Convert the inputs to a Pydantic model
    model = model_class.model_validate(inputs)
    # Call the tool function with the Pydantic model
    return await tool(model)


async def execute_tools_parallel(
    tool_calls: List[Dict[str, Any]],
    tool_dict: Dict[str, Callable],
    enable_parallel: bool = False,
) -> List[Any]:
    """
    Execute multiple tool calls either in parallel or sequentially.

    Args:
        tool_calls: List of tool call dictionaries with function name and arguments
        tool_dict: Dictionary mapping function names to callable functions
        enable_parallel: Whether to execute tools in parallel (True) or sequentially (False)

    Returns:
        List of tool execution results in the same order as input tool_calls
    """
    if not enable_parallel:
        # Sequential execution (current behavior)
        results = []
        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name") or tool_call.get(
                "name"
            )
            func_args = tool_call.get("function", {}).get("arguments") or tool_call.get(
                "arguments", {}
            )

            if func_name in tool_dict:
                tool = tool_dict[func_name]
                if inspect.iscoroutinefunction(tool):
                    result = await execute_tool_async(tool, func_args)
                else:
                    result = execute_tool(tool, func_args)
                results.append(result)
            else:
                results.append(f"Error: Function {func_name} not found")
        return results
    else:
        # Parallel execution
        async def execute_single_tool(tool_call):
            try:
                func_name = tool_call.get("function", {}).get("name") or tool_call.get(
                    "name"
                )
                func_args = tool_call.get("function", {}).get(
                    "arguments"
                ) or tool_call.get("arguments", {})

                if func_name in tool_dict:
                    tool = tool_dict[func_name]
                    if inspect.iscoroutinefunction(tool):
                        return await execute_tool_async(tool, func_args)
                    else:
                        # Run sync function in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            None, execute_tool, tool, func_args
                        )
                else:
                    return f"Error: Function {func_name} not found"
            except Exception as e:
                return f"Error executing {func_name}: {str(e)}"

        # Execute all tool calls concurrently
        tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to error strings
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)

        return processed_results


def verify_post_tool_function(function: Callable):
    """
    Verify that the post_tool_function is a function that takes exactly 3 arguments: function_name, input_args, tool_result
    """
    sig = inspect.signature(function)
    if sig.parameters.get("function_name") is None:
        raise ValueError("post_tool_function must have parameter named `function_name`")
    if sig.parameters.get("input_args") is None:
        raise ValueError("post_tool_function must have parameter named `input_args`")
    if sig.parameters.get("tool_result") is None:
        raise ValueError("post_tool_function must have parameter named `tool_result`")

    return function
