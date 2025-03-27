from mcp.server.fastmcp import FastMCP

host = "0.0.0.0"
port = 8001
mcp = FastMCP(
    name="arithmetic_sse",
    host=host,
    port=port,
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


if __name__ == "__main__":
    try:
        print(f"Starting FastMCP server {mcp.name} with host {host} and port {port}")
        mcp.run(transport="sse")
    except Exception as e:
        print(f"Error running FastMCP server {mcp.name}: {e}")
