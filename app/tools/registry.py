"""Tool registry for managing and executing tools."""

import warnings
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)

# Internal registry dict
_TOOLS: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """Register a tool instance.

    Args:
        tool: Tool instance to register
    """
    if tool.name in _TOOLS:
        logger.warning("Tool already registered, overwriting", tool_name=tool.name)
    _TOOLS[tool.name] = tool
    logger.debug("Tool registered", tool_name=tool.name)


def get_tool(tool_name: str) -> BaseTool | None:
    """Get a tool by name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool instance or None if not found
    """
    return _TOOLS.get(tool_name)


def get_available_tool_names() -> list[str]:
    """Get list of all registered tool names.

    Returns:
        List of tool names
    """
    return list(_TOOLS.keys())


async def execute_tool(
    session: AsyncSession,
    tool_name: str,
    tool_input: dict[str, Any],
) -> str:
    """Execute a tool by name.

    Args:
        session: Database session
        tool_name: Name of the tool
        tool_input: Tool input arguments

    Returns:
        Tool output as string
    """
    tool = get_tool(tool_name)
    if not tool:
        return f"Unknown tool: {tool_name}"

    try:
        return await tool.execute(session, **tool_input)
    except Exception as e:
        logger.error("Tool execution failed", tool=tool_name, error=str(e))
        return f"Tool execution failed: {str(e)}"


# Backwards compatibility property for TOOLS dict
def __getattr__(name: str) -> Any:
    """Provide backwards compatibility for TOOLS dict access."""
    if name == "TOOLS":
        warnings.warn(
            "TOOLS dict is deprecated, use get_tool() or get_available_tool_names() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _TOOLS.copy()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")