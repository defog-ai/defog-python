#!/usr/bin/env python3
"""
Defog CLI - Command line interface for Defog
"""

import argparse
import sys
import logging
import os
from defog.server_config_manager import ConfigManager
from defog.cli_wizard import CLIWizard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serve_command(args):
    """Run the MCP server"""
    # Initialize config manager and wizard
    config_manager = ConfigManager()
    wizard = CLIWizard(config_manager)

    # Run wizard to get/update environment variables
    env_vars = wizard.run()

    # Update os.environ with the configured variables
    for key, value in env_vars.items():
        os.environ[key] = value

    # Reload the config module to pick up new environment variables
    from defog import config

    config.reload()

    # Import and run the MCP server after config is updated
    from defog.mcp_server import run_server

    logger.info("Starting Defog MCP server...")
    run_server(transport=args.transport, port=args.port)


def db_command(args):
    """View and update database configuration"""
    # Initialize config manager and wizard
    config_manager = ConfigManager()
    wizard = CLIWizard(config_manager)

    # Run database configuration
    wizard.configure_database_standalone()


def main():
    parser = argparse.ArgumentParser(
        description="Defog CLI - Natural language to SQL and more"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add serve command
    serve_parser = subparsers.add_parser("serve", help="Start the Defog MCP server")
    serve_parser.add_argument(
        "--transport",
        type=str,
        default=None,
        help="Transport type (e.g., 'stdio', 'streamable-http'). Default: stdio",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number for streamable-http transport",
    )
    serve_parser.set_defaults(func=serve_command)

    # Add db command
    db_parser = subparsers.add_parser(
        "db", help="View and update database configuration"
    )
    db_parser.set_defaults(func=db_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
