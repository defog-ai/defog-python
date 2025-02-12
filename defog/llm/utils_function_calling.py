import inspect
from typing import Callable, List, Dict, Any
from pydantic import BaseModel
from defog.llm.models import OpenAIFunction


def get_function_specs(functions: List[Callable]) -> List[OpenAIFunction]:
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

    return function_specs


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
