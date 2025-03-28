import os
import json
import pytest
import asyncio
import subprocess
import tempfile
import time
from unittest.mock import patch, MagicMock, AsyncMock

from defog.llm.utils_mcp import MCPClient, initialize_mcp_client

# Paths to arithmetic server scripts
SSE_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_arithmetic_sse.py")
STDIO_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_arithmetic_stdio.py")

# Verify that server scripts exist
if not os.path.exists(SSE_SERVER_SCRIPT):
    print(f"Warning: SSE server script not found at {SSE_SERVER_SCRIPT}")
if not os.path.exists(STDIO_SERVER_SCRIPT):
    print(f"Warning: stdio server script not found at {STDIO_SERVER_SCRIPT}")


@pytest.fixture
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def arithmetic_sse_server(request):
    """
    Start an arithmetic SSE server for testing.
    The server will be terminated only after the test using this fixture completes.
    """
    # Skip if server script doesn't exist
    if not os.path.exists(SSE_SERVER_SCRIPT):
        pytest.skip(f"SSE server script not found at {SSE_SERVER_SCRIPT}")

    # Create a config file for the server
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    config = {
        "mcpServers": {
            "arithmetic_sse": {"command": "sse", "args": ["http://localhost:8001/sse"]}
        }
    }
    json.dump(config, config_file)
    config_file.close()

    # Start the server in a subprocess
    process = subprocess.Popen(["python", SSE_SERVER_SCRIPT])
    print(f"Started SSE server: {process.pid}")

    # Wait for server to start
    time.sleep(2)

    # Store the process in the request object for cleanup
    request.addfinalizer(lambda: process.terminate())

    yield config_file.name

    # Clean up config file
    os.unlink(config_file.name)


@pytest.fixture(scope="function")
def arithmetic_stdio_server():
    """Create a config for an arithmetic stdio server for testing"""
    # Skip if server script doesn't exist
    if not os.path.exists(STDIO_SERVER_SCRIPT):
        pytest.skip(f"stdio server script not found at {STDIO_SERVER_SCRIPT}")

    # Create a config file for the server
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

    config = {
        "mcpServers": {
            "arithmetic_stdio": {"command": "python", "args": [STDIO_SERVER_SCRIPT]}
        }
    }
    json.dump(config, config_file)
    config_file.close()

    yield config_file.name

    # Clean up config file
    os.unlink(config_file.name)


class TestMCPClient:
    """Test the MCPClient class"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test basic initialization of MCPClient"""
        client = MCPClient(model_name="claude-3-7-sonnet-20250219")
        assert client.model_name == "claude-3-7-sonnet-20250219"
        assert client.model_provider == "anthropic"
        assert client.anthropic is not None
        assert client.openai is None
        assert client.gemini is None

        client = MCPClient(model_name="gpt-4")
        assert client.model_name == "gpt-4"
        assert client.model_provider == "openai"
        assert client.anthropic is None
        assert client.openai is not None
        assert client.gemini is None

        client = MCPClient(model_name="gemini-1.5-pro")
        assert client.model_name == "gemini-1.5-pro"
        assert client.model_provider == "gemini"
        assert client.anthropic is None
        assert client.openai is None
        assert client.gemini is not None

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Unsupported model name"):
            MCPClient(model_name="invalid-model")

    @pytest.mark.asyncio
    async def test_call_tool_no_connections(self):
        """Test call_tool with no server connections"""
        client = MCPClient()

        with pytest.raises(ValueError, match="No server connections available"):
            await client.call_tool("test_tool", {})

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    async def test_call_tool_nonexistent_tool(self):
        """Test call_tool with a tool that doesn't exist"""
        client = MCPClient()
        # Manually set up connections to avoid actual connection
        client.connections = {"test_server": {"session": AsyncMock()}}

        with pytest.raises(ValueError, match="not found in any connected server"):
            await client.call_tool("nonexistent_tool", {})

        # Cleanup
        await client.cleanup()


# Live tests that require API keys and running servers
class TestLiveMCPClient:
    """Live tests for MCPClient functionality"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_anthropic_live(self):
        """Test live Anthropic API integration (requires API key)"""
        # Create client
        client = MCPClient(model_name="claude-3-7-sonnet-20250219")

        try:
            # Simple query that should not use tools
            result = await client.process_query("What is 2+2?")

            # Verify result
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"Anthropic response: {result}")

            # Basic check for correct answer
            assert "4" in result
        finally:
            # Clean up
            await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    async def test_openai_live(self):
        """Test live OpenAI API integration (requires API key)"""
        # Create client
        client = MCPClient(model_name="gpt-3.5-turbo")

        try:
            # Simple query that should not use tools
            result = await client.process_query("What is 2+2?")

            # Verify result
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"OpenAI response: {result}")

            # Basic check for correct answer
            assert "4" in result
        finally:
            # Clean up
            await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    async def test_gemini_live(self):
        """Test live Gemini API integration (requires API key)"""
        # Create client
        client = MCPClient(model_name="gemini-1.5-pro")

        try:
            # Simple query that should not use tools
            result = await client.process_query("What is 2+2?")

            # Verify result
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"Gemini response: {result}")

            # Basic check for correct answer
            assert "4" in result
        finally:
            # Clean up
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_sse_server_connection(self, arithmetic_sse_server):
        """Test connection to SSE server"""
        client = None
        try:
            print("Starting SSE server connection test...")
            # Initialize client
            client = await initialize_mcp_client(
                arithmetic_sse_server, "claude-3-7-sonnet-20250219"
            )

            # Check the connection
            assert client is not None
            assert len(client.connections) > 0

            # Print available tools
            print(f"Connected to server with {len(client.all_tools)} tools:")
            for tool in client.all_tools:
                print(f"- {tool.name}: {tool.description}")

            # Verify we have arithmetic tools
            assert any(tool.name == "add" for tool in client.all_tools)
            assert any(tool.name == "multiply" for tool in client.all_tools)

            # Test calling a tool directly
            result, result_text = await client._handle_tool_call(
                "add", {"a": 5, "b": 3}
            )
            assert result.isError is False
            assert "8" in result_text
            print(f"Successfully called add tool, result: {result_text}")

        except Exception as e:
            pytest.fail(f"Failed to connect to SSE server: {str(e)}")
        finally:
            # Clean up client
            if client:
                print("Cleaning up MCP client...")
                await client.cleanup()
            print("SSE server connection test completed")

    @pytest.mark.asyncio
    async def test_stdio_server_connection(self, arithmetic_stdio_server):
        """Test connection to stdio server"""
        client = None
        try:
            print("Starting stdio server connection test...")
            # Initialize client
            client = await initialize_mcp_client(
                arithmetic_stdio_server, "claude-3-7-sonnet-20250219"
            )

            # Check the connection
            assert client is not None
            assert len(client.connections) > 0

            # Print available tools
            print(f"Connected to stdio server with {len(client.all_tools)} tools:")
            for tool in client.all_tools:
                print(f"- {tool.name}: {tool.description}")

            # Verify we have arithmetic tools
            assert any(tool.name == "add" for tool in client.all_tools)
            assert any(tool.name == "multiply" for tool in client.all_tools)

            # Test calling a tool directly
            result, result_text = await client._handle_tool_call(
                "multiply", {"a": 5, "b": 3}
            )
            assert result.isError is False
            assert "15" in result_text
            print(f"Successfully called multiply tool, result: {result_text}")

        except Exception as e:
            pytest.fail(f"Failed to connect to stdio server: {str(e)}")
        finally:
            # Clean up client
            if client:
                print("Cleaning up MCP client...")
                await client.cleanup()
            print("stdio server connection test completed")

    @pytest.mark.asyncio
    async def test_initialize_with_invalid_config_type(self):
        """Test initializing MCP client with an invalid config type"""
        try:
            # Try to initialize with an invalid config type (integer)
            await initialize_mcp_client(123, "claude-3-7-sonnet-20250219")
            pytest.fail("Should have raised ValueError for invalid config type")
        except ValueError as e:
            # Ensure the error message mentions the invalid type
            assert "must be a string path or dictionary" in str(e)
            print(f"Correctly raised ValueError: {str(e)}")

    @pytest.mark.asyncio
    async def test_initialize_with_invalid_config_dict(self):
        """Test initializing MCP client with an invalid config dictionary (missing mcpServers)"""
        try:
            # Create a dictionary without the mcpServers key
            invalid_config = {"someOtherKey": "value"}

            # Try to initialize with an invalid config dictionary
            await initialize_mcp_client(invalid_config, "claude-3-7-sonnet-20250219")
            pytest.fail("Should have raised ValueError for missing mcpServers key")
        except ValueError as e:
            # Ensure the error message mentions the missing key
            assert "missing required 'mcpServers' key" in str(e)
            print(f"Correctly raised ValueError: {str(e)}")

    @pytest.mark.asyncio
    async def test_initialize_with_empty_mcpservers(self):
        """Test initializing MCP client with empty mcpServers dictionary"""
        try:
            # Create a dictionary with empty mcpServers
            invalid_config = {"mcpServers": {}}

            # Try to initialize with empty mcpServers
            await initialize_mcp_client(invalid_config, "claude-3-7-sonnet-20250219")
            pytest.fail("Should have raised ValueError for empty mcpServers")
        except ValueError as e:
            # Ensure the error message mentions empty mcpServers
            assert "No MCP servers defined in config" in str(e)
            print(f"Correctly raised ValueError: {str(e)}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    async def test_end_to_end_with_tool_use_anthropic(self, arithmetic_stdio_server):
        """End-to-end test with tool use using Anthropic (requires API key and server)"""
        client = None
        try:
            print("Starting end-to-end tool use test with Anthropic...")
            # Initialize client
            client = await initialize_mcp_client(
                arithmetic_stdio_server, "claude-3-7-sonnet-20250219"
            )

            # Query that should use the multiply tool
            query = "I need to multiply 12 and 34. Can you use the multiply tool to help me?"

            # Use mcp_chat to test the full client API
            result, tool_outputs = await client.mcp_chat(query=query)

            # Verify result
            assert result is not None
            assert isinstance(result, str)
            print(f"End-to-end response from Anthropic: {result}")

            # Check if the tool was used (using the returned tool_outputs)
            assert len(tool_outputs) > 0
            tool_used = False
            for tool in tool_outputs:
                print(f"Tool used: {tool['name']}")
                print(f"Tool args: {tool['args']}")
                print(f"Tool result: {tool['result']}")
                if tool["name"] == "multiply":
                    tool_used = True

            assert tool_used, "Multiply tool was not used"

            # Basic check for correct answer (408)
            assert "408" in result

        except Exception as e:
            pytest.fail(f"End-to-end test with Anthropic failed: {str(e)}")
        finally:
            # Clean up client
            if client:
                print("Cleaning up MCP client...")
                await client.cleanup()
            print("End-to-end tool use test with Anthropic completed")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    async def test_end_to_end_with_tool_use_gemini(self, arithmetic_stdio_server):
        """End-to-end test with tool use using Gemini (requires API key and server)"""
        client = None
        try:
            print("Starting end-to-end tool use test with Gemini...")
            # Initialize client
            client = await initialize_mcp_client(
                arithmetic_stdio_server, "gemini-1.5-pro"
            )

            # Query that should use the multiply tool
            query = "I need to multiply 12 and 34. Can you use the multiply tool to help me?"

            # Use mcp_chat to test the full client API
            result, tool_outputs = await client.mcp_chat(query=query)

            # Verify result
            assert result is not None
            assert isinstance(result, str)
            print(f"End-to-end response from Gemini: {result}")

            # Check if the tool was used (using the returned tool_outputs)
            assert len(tool_outputs) > 0
            tool_used = False
            for tool in tool_outputs:
                print(f"Tool used: {tool['name']}")
                print(f"Tool args: {tool['args']}")
                print(f"Tool result: {tool['result']}")
                if tool["name"] == "multiply":
                    tool_used = True

            assert tool_used, "Multiply tool was not used"

            # Basic check for correct answer (408)
            assert "408" in result

        except Exception as e:
            pytest.fail(f"End-to-end test with Gemini failed: {str(e)}")
        finally:
            # Clean up client
            if client:
                print("Cleaning up MCP client...")
                await client.cleanup()
            print("End-to-end tool use test with Gemini completed")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    async def test_end_to_end_with_tool_use_openai(self, arithmetic_stdio_server):
        """End-to-end test with tool use using OpenAI (requires API key and server)"""
        client = None
        try:
            print("Starting end-to-end tool use test with OpenAI...")
            # Initialize client with GPT-4
            client = await initialize_mcp_client(arithmetic_stdio_server, "o3-mini")

            # Query that should use the multiply tool
            query = "I need to multiply 12 and 34. Can you use the multiply tool to help me?"

            # Use mcp_chat to test the full client API
            result, tool_outputs = await client.mcp_chat(query=query)

            # Verify result
            assert result is not None
            assert isinstance(result, str)
            print(f"End-to-end response from OpenAI: {result}")

            # Check if the tool was used (using the returned tool_outputs)
            assert len(tool_outputs) > 0
            tool_used = False
            for tool in tool_outputs:
                print(f"Tool used: {tool['name']}")
                print(f"Tool args: {tool['args']}")
                print(f"Tool result: {tool['result']}")
                if tool["name"] == "multiply":
                    tool_used = True

            assert tool_used, "Multiply tool was not used"

            # Basic check for correct answer (408)
            assert "408" in result

        except Exception as e:
            pytest.fail(f"End-to-end test with OpenAI failed: {str(e)}")
        finally:
            # Clean up client
            if client:
                print("Cleaning up MCP client...")
                await client.cleanup()
            print("End-to-end tool use test with OpenAI completed")
