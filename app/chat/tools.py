"""Chat-specific tool configuration.

This module defines which tools are available to chat agents.
All tool implementations are now in app/tools/ package.
"""

# Tool list for chat agents
CHAT_TOOLS = [
    "vector_search",
    "web_search",
    "article_lookup",
    "citation_lookup",
    "conversation_history",
    "related_articles",
]


def get_chat_tool_names() -> list[str]:
    """Get list of tools available for chat agents.

    Returns:
        List of tool names
    """
    return CHAT_TOOLS