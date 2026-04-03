"""Tools package for ReAct-style tool implementations.

This package provides:
- BaseTool: Abstract base class for tools
- Registry functions: register_tool, get_tool, get_available_tool_names, execute_tool
- Tool implementations: VectorSearchTool, WebSearchTool, QueryExpandTool

Usage:
    from app.tools import execute_tool, get_available_tool_names
    from app.tools import VectorSearchTool, register_tool

    # Execute a tool
    result = await execute_tool(session, "vector_search", {"query": "AI news"})

    # Get available tools
    tool_names = get_available_tool_names()

    # Create and register a custom tool
    class MyTool(BaseTool):
        name = "my_tool"
        description = "My custom tool"
        async def execute(self, session, **kwargs) -> str:
            return "result"

    register_tool(MyTool())
"""

from app.tools.base import BaseTool
from app.tools.registry import (
    execute_tool,
    get_available_tool_names,
    get_tool,
    register_tool,
)
from app.tools.vector_search import VectorSearchTool
from app.tools.web_search import WebSearchTool
from app.tools.query_expand import QueryExpandTool

__all__ = [
    "BaseTool",
    "execute_tool",
    "get_available_tool_names",
    "get_tool",
    "register_tool",
    "VectorSearchTool",
    "WebSearchTool",
    "QueryExpandTool",
]