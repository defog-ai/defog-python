# Task Completion Checklist

When completing any coding task in the defog-python project, follow these steps:

1. **Run Tests**
   - Run scoped tests that directly affect changed code
   - Use: `PYTHONPATH=. python3.12 -m pytest tests/test_specific_file.py`
   - Always use python3.12 to ensure correct Python version
   - Add PYTHONPATH=. to ensure tests use the local version of defog

2. **Format Code**
   - The project uses ruff for formatting (configured in pyproject.toml)
   - Pre-commit hooks are configured to run ruff-format
   - Black is also available: `black .` or `black specific_file.py`

3. **Linting**
   - Ruff is configured to ignore E402 (module import not at top of file)
   - Run: `ruff check .` or `ruff check specific_file.py`

4. **Update Documentation**
   - Check if README.md needs updates
   - Update relevant files in docs/ folder if functionality changed
   - Documentation uses markdown format

5. **Security Check**
   - Ensure no secrets or API keys are exposed in code
   - Verify no sensitive information in logs

6. **PR Reviews**
   - Be honest, technically focused, and blunt
   - Pay special attention to security issues