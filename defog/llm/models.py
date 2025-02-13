from pydantic import BaseModel
from enum import Enum
from typing import Optional, Union, Dict, Any

class OpenAIToolChoice(Enum):
    AUTO = "auto" # default if not provided. calls 0, 1, or multiple functions
    REQUIRED = "required" # calls at least 1 function
    NONE = "none" # calls no functions

class OpenAIFunctionSpecs(BaseModel):
    name: str # name of the function to call
    description: Optional[str] = None # description of the function
    parameters: Optional[Union[str, Dict[str, Any]]] = None # parameters of the function

class OpenAIForcedFunction(BaseModel):
    # a forced function call - forces a call to one specific function
    type: str = "function"
    function: OpenAIFunctionSpecs

class AnthropicToolChoice(Enum):
    AUTO = "auto" # default if not provided. calls 0, 1, or multiple functions
    REQUIRED = "required" # calls at least 1 function

class AnthropicFunctionSpecs(BaseModel):
    name: str # name of the function to call
    description: Optional[str] = None # description of the function
    input_schema: Optional[Union[str, Dict[str, Any]]] = None # parameters of the function

class AnthropicForcedFunction(BaseModel):
    # a forced function call - forces a call to one specific function
    type: str = "function"
    function: AnthropicFunctionSpecs