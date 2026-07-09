"""Toy MCP server — Phase 0 learning exercise.

Not part of the shipped kbmcp server (see src/kbmcp/, built starting Phase 4).
Exists to exercise the FastMCP tool/resource/transport model end-to-end.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kb-toy")


@mcp.tool()
def ping() -> str:
    """Liveness check — always returns 'pong'."""
    return "pong"


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text unchanged, to verify argument round-tripping."""
    return text


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Return a personalized greeting for the given name."""
    return f"Hello, {name}!"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
