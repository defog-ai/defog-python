#!/usr/bin/env python3
"""
Defog CLI - Command line interface for Defog
"""

import argparse
import sys
import logging
from defog.mcp_server import run_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serve_command(args):
    """Run the MCP server"""
    logger.info("Starting Defog MCP server...")
    run_server()


def main():
    parser = argparse.ArgumentParser(
        description="Defog CLI - Natural language to SQL and more"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add serve command
    serve_parser = subparsers.add_parser("serve", help="Start the Defog MCP server")
    serve_parser.set_defaults(func=serve_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
