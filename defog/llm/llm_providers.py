from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"
    DEEPSEEK = "deepseek"
    TOGETHER = "together"
    MISTRAL = "mistral"
    ALIBABA = "alibaba"
