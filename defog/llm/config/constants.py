DEFAULT_TIMEOUT = 600  # seconds
MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.0

# Provider-specific constants
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENAI_BASE_URL = "https://api.openai.com/v1/"
ALIBABA_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Model families that require special handling
O_MODELS = ["o1-mini", "o1-preview", "o1", "o3-mini", "o3", "o4-mini"]

DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]

MODELS_WITHOUT_TEMPERATURE = ["deepseek-reasoner"] + [model for model in O_MODELS]

MODELS_WITHOUT_RESPONSE_FORMAT = [
    "o1-mini",
    "o1-preview",
]

MODELS_WITHOUT_TOOLS = ["o1-mini", "o1-preview", "deepseek-reasoner"]

MODELS_WITH_PARALLEL_TOOL_CALLS = ["gpt-4o", "gpt-4o-mini"]

MODELS_WITH_PREDICTION = ["gpt-4o", "gpt-4o-mini"]
