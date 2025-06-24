"""Configuration manager for defog serve command."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration storage and retrieval for defog serve."""

    CONFIG_FILENAME = "config.json"
    CONFIG_DIR_NAME = ".defog"

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the config manager.

        Args:
            config_dir: Directory to store config. Defaults to ~/.defog/
        """
        if config_dir is None:
            # Use standard location in user's home directory
            self.config_dir = Path.home() / self.CONFIG_DIR_NAME
        else:
            self.config_dir = config_dir

        # Ensure the config directory exists
        self.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        self.config_path = self.config_dir / self.CONFIG_FILENAME

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from disk.

        Returns:
            Configuration dictionary or empty dict if not found.
        """
        if not self.config_path.exists():
            return {}

        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to disk.

        Args:
            config: Configuration dictionary to save.
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            # Set file permissions to be readable/writable by user only
            self.config_path.chmod(0o600)
        except IOError as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise

    def get_env_with_config(self) -> Dict[str, str]:
        """Get environment variables, falling back to saved config.

        Returns:
            Dictionary of environment variables with config fallbacks.
        """
        config = self.load_config()
        env_vars = {}

        # Merge environment variables with saved config
        # Environment variables take precedence
        for key, value in config.items():
            if key not in os.environ:
                env_vars[key] = value
            else:
                env_vars[key] = os.environ[key]

        # Add any environment variables not in config
        for key, value in os.environ.items():
            if key not in env_vars:
                env_vars[key] = value

        return env_vars

    def update_config(self, updates: Dict[str, Optional[str]]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of key-value pairs to update.
        """
        config = self.load_config()
        for key, value in updates.items():
            if value is None:
                config.pop(key, None)  # Remove key
            else:
                config[key] = value  # Update key
        self.save_config(config)

    def clear_config(self) -> None:
        """Remove the configuration file."""
        if self.config_path.exists():
            self.config_path.unlink()
