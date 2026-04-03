"""Deprecated: Use app.tools instead.

This module is kept for backwards compatibility only.
All tool functionality has been moved to app.tools package.
"""

import warnings

from app.tools import (
    BaseTool,
    QueryExpandTool,
    VectorSearchTool,
    WebSearchTool,
    execute_tool,
    get_tool,
)

# Emit deprecation warning on import
warnings.warn(
    "app.deep_search.tools is deprecated, use app.tools instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backwards compatibility
__all__ = [
    "BaseTool",
    "VectorSearchTool",
    "WebSearchTool",
    "QueryExpandTool",
    "TOOLS",
    "execute_tool",
    "get_tool",
]

# Provide TOOLS dict via registry's __getattr__ for backwards compatibility
from app.tools.registry import _TOOLS

TOOLS = _TOOLS.copy()