"""Centralized configuration management for Defog.

This module provides a unified interface for accessing configuration values,
checking both environment variables and saved config files.
"""

from typing import Optional, Dict
from .server_config_manager import ConfigManager

# Global instance of the config manager
_config_manager = ConfigManager()
_env_vars: Optional[Dict[str, str]] = None


def _ensure_config_loaded():
    """Ensure configuration is loaded from both environment and config file."""
    global _env_vars
    if _env_vars is None:
        _env_vars = _config_manager.get_env_with_config()


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a configuration value with fallback to config file.

    Args:
        key: The configuration key to look up
        default: Default value if key is not found

    Returns:
        The configuration value or default
    """
    _ensure_config_loaded()
    return _env_vars.get(key, default)


def get_required(key: str) -> str:
    """Get a required configuration value.

    Args:
        key: The configuration key to look up

    Returns:
        The configuration value

    Raises:
        ValueError: If the key is not found
    """
    value = get(key)
    if value is None:
        raise ValueError(f"Required configuration '{key}' not found")
    return value


def reload():
    """Reload configuration from environment and config file."""
    global _env_vars
    _env_vars = None
    _ensure_config_loaded()


def get_all() -> Dict[str, str]:
    """Get all configuration values.

    Returns:
        Dictionary of all configuration values
    """
    _ensure_config_loaded()
    return _env_vars.copy()


# Convenience functions for common checks
def has_openai_key() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(get("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is configured."""
    return bool(get("ANTHROPIC_API_KEY"))


def has_gemini_key() -> bool:
    """Check if Gemini API key is configured."""
    return bool(get("GEMINI_API_KEY"))


def has_any_llm_key() -> bool:
    """Check if any LLM API key is configured."""
    return has_openai_key() or has_anthropic_key() or has_gemini_key()


def get_db_type() -> Optional[str]:
    """Get the configured database type."""
    return get("DB_TYPE")


def has_db_configured() -> bool:
    """Check if database is configured."""
    return bool(get_db_type())
