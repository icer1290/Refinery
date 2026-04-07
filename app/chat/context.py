"""Context definition for the chat workflow using LangGraph context_schema."""

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ChatContext:
    """Context passed to chat nodes during execution.

    Provides access to shared resources and session state.
    Follows the pattern used in DeepSearchContext.
    """

    session: AsyncSession  # Database session
    conversation_id: str  # Conversation UUID
    article_id: str  # Article UUID
    user_id: int  # User ID from api-server
    redis_client: Optional[Any] = None  # Redis client for caching
    session_cache: Optional[dict[str, Any]] = None  # Active session data from Redis