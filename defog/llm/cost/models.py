MODEL_COSTS = {
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
    "gpt-4.1": {
        "input_cost_per1k": 0.002,
        "cached_input_cost_per1k": 0.0005,
        "output_cost_per1k": 0.008,
    },
    "gpt-4.1-mini": {
        "input_cost_per1k": 0.0004,
        "cached_input_cost_per1k": 0.0001,
        "output_cost_per1k": 0.0016,
    },
    "gpt-4.1-nano": {
        "input_cost_per1k": 0.0001,
        "cached_input_cost_per1k": 0.000025,
        "output_cost_per1k": 0.0004,
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
    "o3": {
        "input_cost_per1k": 0.01,
        "cached_input_cost_per1k": 0.0025,
        "output_cost_per1k": 0.04,
    },
    "o4-mini": {
        "input_cost_per1k": 0.0011,
        "cached_input_cost_per1k": 0.000275,
        "output_cost_per1k": 0.0044,
    },
    "gpt-4-turbo": {"input_cost_per1k": 0.01, "output_cost_per1k": 0.03},
    "gpt-3.5-turbo": {"input_cost_per1k": 0.0005, "output_cost_per1k": 0.0015},
    "claude-3-5-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-sonnet-4": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.00075,
        "output_cost_per1k": 0.015,
    },
    "claude-opus-4": {
        "input_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.075,
    },
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
    "gemini-2.5-flash": {
        "input_cost_per1k": 0.00015,
        "output_cost_per1k": 0.0035,
    },
    "gemini-2.5-pro": {
        "input_cost_per1k": 0.00125,
        "output_cost_per1k": 0.01,
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
    # Alibaba Cloud (Qwen) models
    "qwen-turbo": {
        "input_cost_per1k": 0.0003,
        "output_cost_per1k": 0.0006,
    },
    "qwen-plus": {
        "input_cost_per1k": 0.0008,
        "output_cost_per1k": 0.002,
    },
    "qwen-max": {
        "input_cost_per1k": 0.002,
        "output_cost_per1k": 0.006,
    },
    "qwen2-72b-instruct": {
        "input_cost_per1k": 0.0009,
        "output_cost_per1k": 0.0009,
    },
    "qwen2-7b-instruct": {
        "input_cost_per1k": 0.0003,
        "output_cost_per1k": 0.0003,
    },
    # Mistral AI models
    "mistral-tiny": {
        "input_cost_per1k": 0.00025,
        "output_cost_per1k": 0.00025,
    },
    "mistral-small": {
        "input_cost_per1k": 0.002,
        "output_cost_per1k": 0.006,
    },
    "mistral-medium": {
        "input_cost_per1k": 0.0027,
        "output_cost_per1k": 0.0081,
    },
    "mistral-large": {
        "input_cost_per1k": 0.004,
        "output_cost_per1k": 0.012,
    },
    "open-mistral-7b": {
        "input_cost_per1k": 0.00025,
        "output_cost_per1k": 0.00025,
    },
    "open-mixtral-8x7b": {
        "input_cost_per1k": 0.0007,
        "output_cost_per1k": 0.0007,
    },
    "open-mixtral-8x22b": {
        "input_cost_per1k": 0.002,
        "output_cost_per1k": 0.006,
    },
}
