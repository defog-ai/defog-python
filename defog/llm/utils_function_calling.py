import inspect
from typing import Callable, List, Dict, Any, Union
from pydantic import BaseModel
from defog.llm.models import OpenAIFunctionSpecs, AnthropicFunctionSpecs


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
        input_schema = model_class.model_json_schema()

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
            function_specs.append(
                {
                    "name": func.__name__,
                    "description": docstring,
                    "input_schema": input_schema,
                }
            )
        elif model.startswith("gemini"):
            from google.genai import types

            # change all "type" values to uppercase
            input_schema["type"] = input_schema["type"].upper()
            for prop in input_schema["properties"].values():
                prop["type"] = prop["type"].upper()

            func_spec = {
                "name": func.__name__,
                "description": docstring,
                "parameters": input_schema,
            }
            function_declaration = types.FunctionDeclaration(**func_spec)
            tool = types.Tool(function_declarations=[function_declaration])

            function_specs.append(tool)
        else:
            raise ValueError(f"Model does not support function calling: {model}")

    return function_specs


def convert_tool_choice(tool_choice: str, tool_name_list: List[str], model: str):
    """
    Convert a tool choice to a function calling tool choice that is compatible with the model.
    """
    model_map = {
        "openai": {
            "prefixes": ["gpt", "o1", "chatgpt", "o3"],
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


def verify_post_tool_function(function: Callable):
    """
    Verify that the post_tool_function is a function that takes exactly 3 arguments: function_name, input_args, tool_result
    """
    if not inspect.isfunction(function):
        raise ValueError(f"post_tool_function must be a function, not {type(function)}")
    sig = inspect.signature(function)
    if len(sig.parameters) != 3:
        raise ValueError(
            "post_tool_function must have exactly three parameters: function_name, input_args, and tool_result"
        )
    if sig.parameters.get("function_name") is None:
        raise ValueError("post_tool_function must have parameter named `function_name`")
    if sig.parameters.get("input_args") is None:
        raise ValueError("post_tool_function must have parameter named `input_args`")
    if sig.parameters.get("tool_result") is None:
        raise ValueError("post_tool_function must have parameter named `tool_result`")

    return function
