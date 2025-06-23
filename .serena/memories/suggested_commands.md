# Suggested Commands for Defog Python Development

## Testing Commands
```bash
# Run all tests
PYTHONPATH=. python3.12 -m pytest

# Run specific test file
PYTHONPATH=. python3.12 -m pytest tests/test_llm.py

# Run tests with verbose output
PYTHONPATH=. python3.12 -m pytest -v

# Run tests matching a pattern
PYTHONPATH=. python3.12 -m pytest -k "test_chat_async"
```

## Code Formatting and Linting
```bash
# Format code with black
black .
black defog/llm/utils.py

# Check code with ruff
ruff check .
ruff check defog/

# Format code with ruff (via pre-commit)
ruff format .
```

## Development
```bash
# Install in development mode
pip install -e .

# Install with extras (e.g., postgres support)
pip install -e ".[postgres]"

# Run examples
PYTHONPATH=. python examples/sql_agent_example.py
```

## Git Commands
```bash
# Check status
git status

# View changes
git diff

# Stage changes
git add file.py

# Commit changes
git commit -m "feat: add new functionality"
```

## macOS Utilities
```bash
# Search for text in files (using ripgrep)
rg "pattern" .
rg "chat_async" --type py

# Find files
find . -name "*.py" -type f

# List directory contents
ls -la
ls defog/

# Change directory
cd defog/llm/

# View file contents
cat file.py
less file.py
```

## Environment Setup
```bash
# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```