"""Chat module for multi-turn conversations about news articles.

This module provides:
- Multi-agent chat with supervisor routing
- Tool integration for research and fact-checking
- Memory management with two-layer compression
- Redis caching for active sessions

Main entry point: run_chat()
"""

from app.chat.graph import create_chat_graph, get_chat_graph, run_chat
from app.chat.state import (
    ChatState,
    ChatMessageContent,
    CompactBoundaryMessage,
    ExtractedMemory,
    ToolCallRecord,
    SpecialistResponse,
    RoutingDecision,
    create_initial_chat_state,
    create_user_message,
    create_assistant_message,
)
from app.chat.context import ChatContext

__all__ = [
    # Graph
    "create_chat_graph",
    "get_chat_graph",
    "run_chat",
    # State
    "ChatState",
    "ChatMessageContent",
    "CompactBoundaryMessage",
    "ExtractedMemory",
    "ToolCallRecord",
    "SpecialistResponse",
    "RoutingDecision",
    "create_initial_chat_state",
    "create_user_message",
    "create_assistant_message",
    # Context
    "ChatContext",
]