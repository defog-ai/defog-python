# Code Style and Conventions

## General Behavior
- Do not be obsequious or use emotional words/exclamations
- Be stoic and professional in code and comments

## Code Style
- Python 3.12 target version
- Use type hints extensively (as seen in utils.py and other files)
- Follow async/await patterns for asynchronous operations
- Use Pydantic models for structured data (e.g., ResponseFormat, BaseModel)
- Docstrings follow standard Python conventions with Args, Returns, Raises sections
- Import organization: standard library, third-party, local imports
- No aliases in SQL generation
- Functions and methods use descriptive names with underscores (snake_case)

## Key Patterns
- Extensive use of `async def` for asynchronous functions
- Type annotations with `Union`, `Optional`, `List`, `Dict` from typing
- Error handling with custom exceptions (LLMError, ConfigurationError, ProviderError)
- Deep copying of mutable arguments to avoid side effects
- Exponential backoff for retries

## Documentation
- Always update README.md and docs/ folder after implementing or changing code
- Use markdown format for documentation
- Include code examples in documentation

## Security
- Never introduce code that exposes or logs secrets and keys
- Never commit secrets or keys to the repository
- Follow security best practices

## LLM Usage
- Use `chat_async` function and its parameters instead of using LLM clients directly
- Use structured outputs with `response_format` parameter
- Consider new 2025 models: Claude 4 Sonnet/Opus, GPT-4.1/o3/o4-mini, Gemini 2.5 flash/pro