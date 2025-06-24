"""CLI wizard for configuring defog serve environment variables."""

from typing import Dict, Any, Optional
import pwinput
import logging

from .server_config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CLIWizard:
    """Interactive CLI wizard for configuring defog serve."""

    # Define required environment variables and their descriptions
    LLM_PROVIDERS = {
        "openai": {
            "key": "OPENAI_API_KEY",
            "name": "OpenAI",
            "description": "GPT models (gpt-4.1, o3, etc.)",
        },
        "anthropic": {
            "key": "ANTHROPIC_API_KEY",
            "name": "Anthropic",
            "description": "Claude models (claude-sonnet-4, claude-opus-4, etc.)",
        },
        "gemini": {
            "key": "GEMINI_API_KEY",
            "name": "Google Gemini",
            "description": "Gemini models (gemini-pro-2.5, gemini-flash-2.5, etc.)",
        },
    }

    DATABASE_CONFIGS = {
        "postgres": {
            "name": "PostgreSQL",
            "vars": [
                ("DB_HOST", "Host (e.g., localhost or db.example.com)", False, None),
                ("DB_PORT", "Port", False, "5432"),
                ("DB_USER", "Username", False, None),
                ("DB_PASSWORD", "Password", True, None),
                ("DB_NAME", "Database name", False, None),
            ],
        },
        "mysql": {
            "name": "MySQL",
            "vars": [
                ("DB_HOST", "Host (e.g., localhost or db.example.com)", False, None),
                ("DB_USER", "Username", False, None),
                ("DB_PASSWORD", "Password", True, None),
                ("DB_NAME", "Database name", False, None),
            ],
        },
        "sqlserver": {
            "name": "SQL Server",
            "vars": [
                (
                    "DB_HOST",
                    "Host (e.g., localhost or server.database.windows.net)",
                    False,
                    None,
                ),
                ("DB_USER", "Username", False, None),
                ("DB_PASSWORD", "Password", True, None),
                ("DB_NAME", "Database name", False, None),
            ],
        },
        "bigquery": {
            "name": "Google BigQuery",
            "vars": [
                ("DB_KEY_PATH", "Path to service account key JSON file", False, None)
            ],
        },
        "snowflake": {
            "name": "Snowflake",
            "vars": [
                ("DB_USER", "Username", False, None),
                ("DB_PASSWORD", "Password", True, None),
                (
                    "DB_ACCOUNT",
                    "Account identifier (e.g., xy12345.us-east-1)",
                    False,
                    None,
                ),
                ("DB_WAREHOUSE", "Warehouse name", False, None),
                ("DB_NAME", "Database name", False, None),
            ],
        },
        "redshift": {
            "name": "Redshift",
            "vars": [
                ("DB_HOST", "Host (e.g., localhost or db.example.com)", False, None),
                ("DB_PORT", "Port", False, "5439"),
                ("DB_USER", "Username", False, None),
                ("DB_PASSWORD", "Password", True, None),
                ("DB_NAME", "Database name", False, None),
            ],
        },
        "databricks": {
            "name": "Databricks",
            "vars": [
                (
                    "DB_HOST",
                    "Server hostname (e.g., xxx.cloud.databricks.com)",
                    False,
                    None,
                ),
                ("DB_PATH", "HTTP path (e.g., /sql/1.0/warehouses/xxx)", False, None),
                ("DB_TOKEN", "Access token", True, None),
            ],
        },
        "sqlite": {
            "name": "SQLite",
            "vars": [
                (
                    "DATABASE_PATH",
                    "Path to database file (e.g., ./data.db)",
                    False,
                    None,
                )
            ],
        },
        "duckdb": {
            "name": "DuckDB",
            "vars": [
                (
                    "DATABASE_PATH",
                    "Path to database file (e.g., ./data.duckdb)",
                    False,
                    None,
                )
            ],
        },
    }

    def __init__(self, config_manager: ConfigManager):
        """Initialize the CLI wizard.

        Args:
            config_manager: ConfigManager instance for storing configuration.
        """
        self.config_manager = config_manager

    def run(self) -> Dict[str, str]:
        """Run the interactive configuration wizard.

        Returns:
            Dictionary of configured environment variables.
        """
        self._display_welcome_message()
        self._display_service_description()

        # Get current environment with saved config
        env_vars = self.config_manager.get_env_with_config()
        updates = {}

        # Check for LLM provider
        if not self._has_llm_provider(env_vars):
            self._display_api_key_setup_instructions()
            llm_updates = self._configure_llm_provider()
            updates.update(llm_updates)
            env_vars.update(llm_updates)
        else:
            print("✅ AI provider already configured\n")

        # Ask about database configuration
        if "DB_TYPE" not in env_vars:
            self._display_database_setup_instructions()
            if self._ask_yes_no("Set up database connection?", default="n"):
                db_updates = self._configure_database()
                updates.update(db_updates)
                env_vars.update(db_updates)
            else:
                print("   Skipping database setup.\n")
        else:
            print("✅ Database already configured\n")

        # Save configuration if any updates were made
        if updates:
            self._handle_configuration_save(updates)

        print("\n🎉 All set! Starting your Defog server...\n")
        return env_vars

    def _display_welcome_message(self) -> None:
        """Display welcome message and header."""
        print("\n🚀 Welcome to Defog Server Setup!")
        print("─" * 50)
        print("Let's configure your server. Press Ctrl+C anytime to cancel.\n")

    def _display_service_description(self) -> None:
        """Display description of what defog serve does."""
        print("📚 What is `defog serve`?")
        print(
            "   Defog serve starts an MCP (Model Context Protocol) server that provides"
        )
        print("   AI-powered tools for:")
        print("   • SQL Generation - Convert natural language to SQL queries")
        print("   • Code Interpreter - Execute Python code for data analysis")
        print("   • Web Search - Search the web with AI-powered summaries")
        print(
            "   • YouTube Transcripts - Extract & summarize YouTube videos (requires GEMINI_API_KEY)"
        )
        print(
            "   • PDF Data Extraction - Extract structured data from PDFs (requires ANTHROPIC_API_KEY)"
        )
        print()
        print(
            "   These tools can be used with any MCP-compatible client like Claude Desktop.\n"
        )

    def _display_api_key_setup_instructions(self) -> None:
        """Display instructions for API key setup."""
        print("📝 Step 1: Configure API Keys")
        print("   You'll need API keys from one or more of these services:")
        print("   • OpenAI - For GPT models (gpt-4.1, o3, o4-mini)")
        print("   • Anthropic - For Claude models + PDF extraction tool")
        print("   • Google Gemini - For Gemini models + YouTube transcript tool")
        print()
        print("   💡 Tip: Providing all three API keys unlocks all available tools!\n")

    def _display_database_setup_instructions(self) -> None:
        """Display instructions for database setup."""
        print("\n📊 Step 2: Database setup (optional)")
        print("   Connect a database to enable SQL generation.\n")

    def _handle_configuration_save(self, updates: Dict[str, str]) -> None:
        """Handle saving configuration updates.

        Args:
            updates: Dictionary of configuration updates.
        """
        self._display_configuration_summary(updates)

        if self._ask_yes_no("\nSave these settings for next time?", default="y"):
            self.config_manager.update_config(updates)
            print("\n✅ Settings saved! They'll be used automatically next time.")
        else:
            print("\n⚠️  Settings not saved. You'll need to configure again next time.")

    def _display_configuration_summary(self, updates: Dict[str, str]) -> None:
        """Display summary of configuration updates.

        Args:
            updates: Dictionary of configuration updates.
        """
        print("\n💾 Configuration Summary")
        print("─" * 30)
        for key, value in updates.items():
            display_value = self._mask_sensitive_value(key, value)
            print(f"   {key}: {display_value}")

    def _mask_sensitive_value(self, key: str, value: str) -> str:
        """Mask sensitive values for display.

        Args:
            key: Configuration key.
            value: Configuration value.

        Returns:
            Masked value if sensitive, original value otherwise.
        """
        if "PASSWORD" in key or "TOKEN" in key or "KEY" in key:
            return "****" + value[-4:] if len(value) > 4 else "****"
        return value

    def _has_llm_provider(self, env_vars: Dict[str, str]) -> bool:
        """Check if any LLM provider is configured.

        Args:
            env_vars: Current environment variables.

        Returns:
            True if at least one LLM provider is configured.
        """
        return any(
            provider["key"] in env_vars for provider in self.LLM_PROVIDERS.values()
        )

    def _configure_llm_provider(self) -> Dict[str, str]:
        """Configure LLM provider interactively.

        Returns:
            Dictionary with LLM provider configuration.
        """
        print("   We recommend providing all three API keys for maximum functionality.")
        print("   You can skip any key by pressing Enter.\n")

        api_keys = {}

        # Prompt for each API key
        for provider_key, provider_info in self.LLM_PROVIDERS.items():
            api_key = self._prompt_for_api_key(provider_key, provider_info)
            if api_key:
                api_keys[provider_info["key"]] = api_key

        if not api_keys:
            print(
                "   ⚠️  No API keys configured. At least one is required to use Defog."
            )
            print("   Let's try again...\n")
            return self._configure_llm_provider()

        self._display_configured_keys_summary(api_keys)
        return api_keys

    def _prompt_for_api_key(
        self, provider_key: str, provider_info: Dict[str, str]
    ) -> str:
        """Prompt user for a single API key.

        Args:
            provider_key: Key identifying the provider.
            provider_info: Provider information dictionary.

        Returns:
            API key string or empty string if skipped.
        """
        print(f"   {provider_info['name']} API Key:")
        print(f"   • Models: {provider_info['description']}")

        # Add tool-specific information
        special_features = {
            "anthropic": "PDF data extraction tool",
            "gemini": "YouTube transcript tool",
        }
        if provider_key in special_features:
            print(f"   • Special feature: {special_features[provider_key]}")

        # Provide helpful links
        api_links = {
            "openai": "https://platform.openai.com/api-keys",
            "anthropic": "https://console.anthropic.com/account/keys",
            "gemini": "https://makersuite.google.com/app/apikey",
        }
        print(f"   • Get one at: {api_links.get(provider_key, 'N/A')}")

        api_key = pwinput.pwinput(
            prompt="   Enter API key (or press Enter to skip): "
        ).strip()

        if api_key:
            print("   ✅ API key configured!")
        else:
            print("   ⏭️  Skipped")

        print()  # Add spacing between providers
        return api_key

    def _display_configured_keys_summary(self, api_keys: Dict[str, str]) -> None:
        """Display summary of configured API keys.

        Args:
            api_keys: Dictionary of configured API keys.
        """
        print("   📋 Configured API Keys:")
        for key in api_keys:
            provider_name = next(
                p["name"] for k, p in self.LLM_PROVIDERS.items() if p["key"] == key
            )
            print(f"   • {provider_name}")

    def _configure_database(self) -> Dict[str, str]:
        """Configure database connection interactively.

        Returns:
            Dictionary with database configuration.
        """
        print("   Choose your database type:\n")
        db_types = list(self.DATABASE_CONFIGS.items())

        self._display_grouped_databases(db_types)

        idx = self._get_database_choice(len(db_types))
        db_type, db_config = db_types[idx]

        config_vars = {"DB_TYPE": db_type}
        config_vars.update(self._collect_database_credentials(db_config))

        print("   ✅ Database configured!")
        return config_vars

    def _display_grouped_databases(self, db_types: list) -> None:
        """Display databases grouped by category.

        Args:
            db_types: List of database type tuples.
        """
        # Group databases by type for better UX
        categories = {
            "Server Databases": ["postgres", "mysql", "sqlserver"],
            "Cloud Databases": ["bigquery", "snowflake", "databricks", "redshift"],
            "Local Databases": ["sqlite", "duckdb"],
        }

        for category, db_list in categories.items():
            print(f"   {category}:")
            for i, (key, config) in enumerate(db_types, 1):
                if key in db_list:
                    print(f"   {i}) {config['name']}")
            if category != "Local Databases":  # Don't add newline after last category
                print()

    def _get_database_choice(self, num_options: int) -> int:
        """Get and validate database choice from user.

        Args:
            num_options: Number of available database options.

        Returns:
            Zero-based index of selected database.
        """
        while True:
            try:
                choice = input(f"\n   Select database [1-{num_options}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < num_options:
                    return idx
                print(f"   Please enter a number between 1 and {num_options}")
            except ValueError:
                print("   Please enter a valid number")

    def _collect_database_credentials(
        self, db_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Collect database credentials from user.

        Args:
            db_config: Database configuration dictionary.

        Returns:
            Dictionary of database credentials.
        """
        config_vars = {}
        print(f"\n   Configuring {db_config['name']}:")
        print("   " + "─" * 40)

        for var_name, prompt, is_password, default in db_config["vars"]:
            value = self._prompt_for_value(prompt, is_password, default)
            if value:  # Only add non-empty values
                config_vars[var_name] = value

        return config_vars

    def _prompt_for_value(
        self, prompt: str, is_password: bool, default: Optional[str]
    ) -> str:
        """Prompt user for a configuration value.

        Args:
            prompt: Prompt text to display.
            is_password: Whether to mask input.
            default: Default value if any.

        Returns:
            User input or default value.
        """
        prompt_text = f"   {prompt}"
        if default:
            prompt_text += f" [{default}]"
        prompt_text += ": "

        if is_password:
            value = pwinput.pwinput(prompt=prompt_text).strip()
        else:
            value = input(prompt_text).strip()

        # Use default if no value provided and default exists
        if not value and default:
            value = default

        return value

    def configure_database_standalone(self):
        """Standalone database configuration for 'defog db' command."""
        print("\n🗄️  Database Configuration")
        print("─" * 50)

        # Get current configuration
        env_vars = self.config_manager.get_env_with_config()

        # Check if database is configured
        if "DB_TYPE" in env_vars:
            self._handle_existing_database_config(env_vars)
        else:
            self._handle_no_database_config()

    def _handle_existing_database_config(self, env_vars: Dict[str, str]) -> None:
        """Handle case where database is already configured.

        Args:
            env_vars: Current environment variables.
        """
        self._display_current_database_config(env_vars)
        self._display_database_options()

        choice = self._get_menu_choice(1, 4)
        self._handle_database_menu_choice(choice, env_vars)

    def _handle_no_database_config(self) -> None:
        """Handle case where no database is configured."""
        print("\n⚠️  No database configured yet.")
        if self._ask_yes_no(
            "\n   Would you like to configure a database now?", default="y"
        ):
            updates = self._configure_database()
            self.config_manager.update_config(updates)
            print("\n✅ Database configuration saved!")
        else:
            print("\n   No changes made.")

    def _display_current_database_config(self, env_vars: Dict[str, str]) -> None:
        """Display current database configuration.

        Args:
            env_vars: Current environment variables.
        """
        print("\n📊 Current Database Configuration:")
        print("─" * 40)

        # Get the database type and show its name
        db_type = env_vars.get("DB_TYPE")
        db_name = self.DATABASE_CONFIGS.get(db_type, {}).get("name", db_type)
        print(f"   Database Type: {db_name}")

        # Show relevant configuration (hide sensitive data)
        db_config = self.DATABASE_CONFIGS.get(db_type, {})
        for var_name, prompt, is_password, default in db_config.get("vars", []):
            if var_name in env_vars:
                value = env_vars[var_name]
                display_value = self._mask_sensitive_value(var_name, value)
                print(f"   {var_name}: {display_value}")

    def _display_database_options(self) -> None:
        """Display database configuration options menu."""
        print("\n   Options:")
        print("   1) Update current configuration")
        print("   2) Configure a different database")
        print("   3) Remove database configuration")
        print("   4) Exit")

    def _get_menu_choice(self, min_choice: int, max_choice: int) -> str:
        """Get and validate menu choice from user.

        Args:
            min_choice: Minimum valid choice.
            max_choice: Maximum valid choice.

        Returns:
            User's choice as string.
        """
        while True:
            choice = input(
                f"\n   Select an option [{min_choice}-{max_choice}]: "
            ).strip()
            try:
                choice_int = int(choice)
                if min_choice <= choice_int <= max_choice:
                    return choice
                print(f"   Please enter a number between {min_choice} and {max_choice}")
            except ValueError:
                print("   Please enter a valid number")

    def _handle_database_menu_choice(
        self, choice: str, env_vars: Dict[str, str]
    ) -> None:
        """Handle database menu choice.

        Args:
            choice: User's menu choice.
            env_vars: Current environment variables.
        """
        db_type = env_vars.get("DB_TYPE")
        db_config = self.DATABASE_CONFIGS.get(db_type, {})

        if choice == "1":
            # Update existing configuration
            updates = self._update_existing_database(db_type, env_vars)
            if updates:
                self.config_manager.update_config(updates)
                print("\n✅ Database configuration updated!")
        elif choice == "2":
            # Configure new database
            updates = self._configure_database()
            self.config_manager.update_config(updates)
            print("\n✅ Database configuration updated!")
        elif choice == "3":
            # Remove database configuration
            self._remove_database_configuration(db_config)
        elif choice == "4":
            print("\n   Exiting without changes.")

    def _remove_database_configuration(self, db_config: Dict[str, Any]) -> None:
        """Remove database configuration.

        Args:
            db_config: Current database configuration.
        """
        if self._ask_yes_no(
            "\n   Are you sure you want to remove database configuration?",
            default="n",
        ):
            # Get all DB-related keys to remove
            keys_to_remove = ["DB_TYPE"]
            for var_name, _, _, _ in db_config.get("vars", []):
                keys_to_remove.append(var_name)

            # Create update dict with None values to remove keys
            updates = {key: None for key in keys_to_remove}
            self.config_manager.update_config(updates)
            print("\n✅ Database configuration removed!")

    def _update_existing_database(
        self, db_type: str, current_env: Dict[str, str]
    ) -> Dict[str, str]:
        """Update existing database configuration.

        Args:
            db_type: Current database type.
            current_env: Current environment variables.

        Returns:
            Dictionary with updated configuration.
        """
        db_config = self.DATABASE_CONFIGS[db_type]
        updates = {}

        print(f"\n   Updating {db_config['name']} configuration:")
        print("   (Press Enter to keep current value)")
        print("   " + "─" * 40)

        for var_name, prompt, is_password, default in db_config["vars"]:
            current_value = current_env.get(var_name, "")

            if is_password:
                # For passwords, show masked current value
                if current_value:
                    masked_value = (
                        "****" + current_value[-4:]
                        if len(current_value) > 4
                        else "****"
                    )
                    prompt_text = f"   {prompt} [current: {masked_value}]: "
                else:
                    prompt_text = f"   {prompt}: "

                new_value = pwinput.pwinput(prompt=prompt_text).strip()
            else:
                # For non-passwords, show current value
                if current_value:
                    prompt_text = f"   {prompt} [{current_value}]: "
                else:
                    prompt_text = f"   {prompt}: "

                new_value = input(prompt_text).strip()

            # Only update if a new value was provided
            if new_value:
                updates[var_name] = new_value

        return updates

    def _ask_yes_no(self, question: str, default: str = None) -> bool:
        """Ask a yes/no question with better UX.

        Args:
            question: Question to ask.
            default: Default answer ('y' or 'n').

        Returns:
            True for yes, False for no.
        """
        if default == "y":
            prompt = f"{question} [Y/n]: "
        elif default == "n":
            prompt = f"{question} [y/N]: "
        else:
            prompt = f"{question} [y/n]: "

        while True:
            answer = input(prompt).strip().lower()

            # Handle default
            if not answer and default:
                return default == "y"

            if answer in ["y", "yes"]:
                return True
            elif answer in ["n", "no"]:
                return False
            print("Please answer 'y' or 'n'")
