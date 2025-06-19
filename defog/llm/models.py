from pydantic import BaseModel
from typing import Optional, Union, Dict, Any


class OpenAIFunctionSpecs(BaseModel):
    name: str  # name of the function to call
    description: Optional[str] = None  # description of the function
    parameters: Optional[Union[str, Dict[str, Any]]] = (
        None  # parameters of the function
    )


class AnthropicFunctionSpecs(BaseModel):
    name: str  # name of the function to call
    description: Optional[str] = None  # description of the function
    input_schema: Optional[Union[str, Dict[str, Any]]] = (
        None  # parameters of the function
    )
