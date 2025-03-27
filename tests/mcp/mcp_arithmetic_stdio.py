from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    name="arithmetic_sse",
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
        print(f"Starting FastMCP server {mcp.name} with transport stdio")
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Error running FastMCP server {mcp.name}: {e}")
