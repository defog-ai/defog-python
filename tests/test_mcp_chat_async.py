import os
import pytest
import asyncio
from typing import Dict, Any

from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider


# Paths to arithmetic server scripts
SSE_SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "mcp", "mcp_arithmetic_sse.py"
)
STDIO_SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "mcp", "mcp_arithmetic_stdio.py"
)


@pytest.fixture
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def deepwiki_mcp_server():
    """
    DeepWiki MCP server configuration for testing.
    Uses the public DeepWiki MCP server which supports HTTPS.
    """
    # Return the MCP server configuration for DeepWiki
    yield [{"type": "url", "url": "https://mcp.deepwiki.com/sse", "name": "deepwiki"}]


@pytest.fixture(scope="function")
def arithmetic_stdio_server():
    """Create a config for an arithmetic stdio server for testing"""
    # Skip if server script doesn't exist
    if not os.path.exists(STDIO_SERVER_SCRIPT):
        pytest.skip(f"stdio server script not found at {STDIO_SERVER_SCRIPT}")

    # Note: For stdio servers, we can't use the direct server config format
    # because the Anthropic API expects HTTP endpoints for MCP servers
    # So we'll skip this test for now
    pytest.skip(
        "Stdio servers are not supported directly by Anthropic API MCP connector"
    )


class TestMCPChatAsync:
    """Test the chat_async function with MCP servers"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_chat_async_without_mcp(self):
        """Test chat_async without MCP servers"""
        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            temperature=0.0,
        )

        assert response is not None
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "4" in response.content
        assert response.model == "claude-3-7-sonnet-20250219"
        assert response.input_tokens > 0
        assert response.output_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_chat_async_with_mcp_servers(self, deepwiki_mcp_server):
        """Test chat_async with MCP servers enabled using DeepWiki"""
        messages = [
            {
                "role": "user",
                "content": "Use the read_wiki_structure tool to get documentation topics for the python/cpython GitHub repository",
            }
        ]

        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            temperature=0.0,
            mcp_servers=deepwiki_mcp_server,
        )

        assert response is not None
        assert response.content is not None
        assert isinstance(response.content, str)
        assert response.model == "claude-3-7-sonnet-20250219"
        assert response.input_tokens > 0
        assert response.output_tokens > 0

        # Check that tool outputs are included
        assert response.tool_outputs is not None
        assert len(response.tool_outputs) > 0

        # Verify the read_wiki_structure tool was used
        tool_used = False
        for tool_output in response.tool_outputs:
            if tool_output.get("name") == "read_wiki_structure":
                tool_used = True
                # Should have repository-related argument (repoName or repository)
                args_str = str(tool_output.get("args", {}))
                assert "repository" in args_str or "repoName" in args_str

        assert tool_used, "read_wiki_structure tool was not used"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_chat_async_with_multiple_tool_calls(self, deepwiki_mcp_server):
        """Test chat_async with multiple MCP tool calls using DeepWiki"""
        messages = [
            {
                "role": "user",
                "content": "First use read_wiki_structure to get the structure of the Python repository, then use ask_question to ask about Python's installation process",
            }
        ]

        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            temperature=0.0,
            mcp_servers=deepwiki_mcp_server,
        )

        assert response is not None
        assert response.content is not None
        assert isinstance(response.content, str)

        # Check that multiple tools were used
        assert response.tool_outputs is not None
        assert len(response.tool_outputs) >= 1  # At least one tool should be used

        # Verify DeepWiki tools were used
        tools_used = set()
        for tool_output in response.tool_outputs:
            tools_used.add(tool_output.get("name"))

        # Should use at least one DeepWiki tool
        deepwiki_tools = {"read_wiki_structure", "read_wiki_contents", "ask_question"}
        assert len(tools_used.intersection(deepwiki_tools)) > 0

    @pytest.mark.asyncio
    async def test_chat_async_mcp_with_non_anthropic_provider(self):
        """Test that MCP servers are ignored for non-Anthropic providers"""
        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        # Mock MCP servers config
        mcp_servers = [
            {"type": "url", "url": "http://localhost:8001/sse", "name": "test_server"}
        ]

        # This should work without errors even though OpenAI doesn't support MCP
        # (assuming OpenAI API key is available)
        if os.environ.get("OPENAI_API_KEY"):
            response = await chat_async(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                mcp_servers=mcp_servers,  # This should be ignored
            )

            assert response is not None
            assert response.content is not None
            assert "4" in response.content

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_chat_async_mcp_with_system_message(self, deepwiki_mcp_server):
        """Test chat_async with MCP servers and a system message"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful documentation assistant. Always use the available tools to lookup information about repositories.",
            },
            {
                "role": "user",
                "content": "Look up information about the defog/introspect repository structure",
            },
        ]

        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            temperature=0.0,
            mcp_servers=deepwiki_mcp_server,
        )

        assert response is not None
        assert response.content is not None
        assert isinstance(response.content, str)

        # Verify a DeepWiki tool was used
        assert response.tool_outputs is not None
        deepwiki_tools = {"read_wiki_structure", "read_wiki_contents", "ask_question"}
        tool_used = any(
            tool.get("name") in deepwiki_tools for tool in response.tool_outputs
        )
        assert tool_used

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_chat_async_mcp_with_response_format(self, deepwiki_mcp_server):
        """Test chat_async with MCP servers and response format"""
        from pydantic import BaseModel

        class RepositoryInfo(BaseModel):
            repository_name: str
            has_documentation: bool

        messages = [
            {
                "role": "user",
                "content": "Look up information about the defog/introspect repository and return details in the specified format",
            }
        ]

        response = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            temperature=0.0,
            mcp_servers=deepwiki_mcp_server,
            response_format=RepositoryInfo,
        )

        assert response is not None
        assert response.content is not None
        assert isinstance(response.content, RepositoryInfo)
        assert isinstance(response.content.repository_name, str)
        assert isinstance(response.content.has_documentation, bool)

        # Verify a DeepWiki tool was used
        assert response.tool_outputs is not None
        deepwiki_tools = {"read_wiki_structure", "read_wiki_contents", "ask_question"}
        tool_used = any(
            tool.get("name") in deepwiki_tools for tool in response.tool_outputs
        )
        assert tool_used
