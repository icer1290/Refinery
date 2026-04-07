"""Chat cache service for session management."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.models.orm_models import ChatMemory
from app.services.redis_service import get_redis_service

logger = get_logger(__name__)
settings = get_settings()

# Cache TTLs (in seconds)
SESSION_TTL = settings.chat_session_ttl  # 30 minutes for active session
MEMORY_TTL = settings.chat_memory_ttl  # 24 hours for extracted memories
HISTORY_TTL = settings.chat_history_ttl  # 1 hour for cached history
ARTICLE_CONTEXT_TTL = settings.chat_history_ttl  # 1 hour for article context


class ChatCacheService:
    """Caching service for chat sessions.

    Provides caching for:
    - Active sessions (messages, state, tokens)
    - Extracted memories (user profile, conversation state, citations)
    - Article context (title, summary, deepsearch)
    - Message history
    """

    def __init__(self):
        self.redis = get_redis_service()

    # === Key Generation ===

    def _session_key(self, conversation_id: str) -> str:
        """Generate session cache key."""
        return f"chat:session:{conversation_id}"

    def _memory_key(self, conversation_id: str) -> str:
        """Generate memory cache key."""
        return f"chat:memory:{conversation_id}"

    def _history_key(self, conversation_id: str) -> str:
        """Generate history cache key."""
        return f"chat:history:{conversation_id}"

    def _context_key(self, article_id: str) -> str:
        """Generate article context cache key."""
        return f"chat:article_context:{article_id}"

    # === Session Operations ===

    async def get_session(self, conversation_id: str) -> Optional[dict[str, Any]]:
        """Get active session data from cache.

        Args:
            conversation_id: Conversation UUID

        Returns:
            Session data dict or None
        """
        return await self.redis.get_json(self._session_key(conversation_id))

    async def set_session(
        self,
        conversation_id: str,
        session_data: dict[str, Any],
    ) -> bool:
        """Cache active session data.

        Args:
            conversation_id: Conversation UUID
            session_data: Session data (messages, state, tokens)

        Returns:
            True if cached successfully
        """
        success = await self.redis.set_json(
            self._session_key(conversation_id),
            session_data,
            ttl=SESSION_TTL,
        )
        if success:
            logger.debug("Session cached", conversation_id=conversation_id)
        return success

    async def invalidate_session(self, conversation_id: str) -> bool:
        """Remove session from cache.

        Args:
            conversation_id: Conversation UUID

        Returns:
            True if deleted
        """
        return await self.redis.delete(self._session_key(conversation_id))

    async def refresh_session_ttl(self, conversation_id: str) -> bool:
        """Refresh session TTL to keep it alive.

        Args:
            conversation_id: Conversation UUID

        Returns:
            True if TTL refreshed
        """
        return await self.redis.expire(self._session_key(conversation_id), SESSION_TTL)

    # === Memory Cache ===

    async def get_memories(self, conversation_id: str) -> Optional[dict[str, Any]]:
        """Get cached extracted memories.

        Args:
            conversation_id: Conversation UUID

        Returns:
            Memories dict or None
        """
        return await self.redis.get_json(self._memory_key(conversation_id))

    async def set_memories(
        self,
        conversation_id: str,
        memories: dict[str, Any],
    ) -> bool:
        """Cache extracted memories.

        Args:
            conversation_id: Conversation UUID
            memories: Memory data (user_profile, conversation_state, key_citations)

        Returns:
            True if cached
        """
        success = await self.redis.set_json(
            self._memory_key(conversation_id),
            memories,
            ttl=MEMORY_TTL,
        )
        if success:
            logger.debug("Memories cached", conversation_id=conversation_id)
        return success

    async def update_memory(
        self,
        conversation_id: str,
        memory_type: str,
        content: dict[str, Any],
    ) -> bool:
        """Update specific memory type in cache.

        Args:
            conversation_id: Conversation UUID
            memory_type: Type of memory (user_profile, conversation_state, key_citations)
            content: Memory content

        Returns:
            True if updated
        """
        memories = await self.get_memories(conversation_id) or {}
        memories[memory_type] = {
            "content": content,
            "updated_at": datetime.now().isoformat(),
        }
        return await self.set_memories(conversation_id, memories)

    # === PostgreSQL Persistence ===

    async def save_memories_to_db(
        self,
        session: AsyncSession,
        conversation_id: str,
        memories: dict[str, Any],
    ) -> bool:
        """Save memories to PostgreSQL (upsert by conversation_id + memory_type).

        Args:
            session: Database session
            conversation_id: Conversation UUID
            memories: Memory data dict {user_profile: {...}, conversation_state: {...}}

        Returns:
            True if saved successfully
        """
        try:
            conv_uuid = UUID(conversation_id)

            for memory_type, content in memories.items():
                if not content:
                    continue

                # Check if memory already exists
                stmt = select(ChatMemory).where(
                    ChatMemory.conversation_id == conv_uuid,
                    ChatMemory.memory_type == memory_type,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing memory
                    existing.content = content
                    existing.version += 1
                    existing.extracted_at = datetime.now(timezone.utc)
                else:
                    # Create new memory
                    new_memory = ChatMemory(
                        conversation_id=conv_uuid,
                        memory_type=memory_type,
                        content=content,
                        version=1,
                        extracted_at=datetime.now(timezone.utc),
                    )
                    session.add(new_memory)

            await session.commit()
            logger.debug("Memories saved to PostgreSQL", conversation_id=conversation_id)
            return True

        except Exception as e:
            logger.error("Failed to save memories to PostgreSQL", error=str(e))
            await session.rollback()
            return False

    async def load_memories_from_db(
        self,
        session: AsyncSession,
        conversation_id: str,
        memory_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Load memories from PostgreSQL by conversation_id.

        Args:
            session: Database session
            conversation_id: Conversation UUID
            memory_type: Optional filter for specific memory type (e.g., "session", "user_profile")

        Returns:
            Memory dict. If memory_type specified: returns that type's content dict.
            If not specified: returns combined dict {user_profile: {...}, conversation_state: {...}}
        """
        try:
            conv_uuid = UUID(conversation_id)

            if memory_type:
                # Load specific memory type
                stmt = select(ChatMemory).where(
                    ChatMemory.conversation_id == conv_uuid,
                    ChatMemory.memory_type == memory_type,
                )
                result = await session.execute(stmt)
                mem = result.scalar_one_or_none()

                if mem:
                    logger.debug(
                        f"Loaded {memory_type} from PostgreSQL",
                        conversation_id=conversation_id,
                    )
                    return mem.content
                return None
            else:
                # Load all memory types
                stmt = select(ChatMemory).where(
                    ChatMemory.conversation_id == conv_uuid,
                )
                result = await session.execute(stmt)
                memories_db = result.scalars().all()

                if not memories_db:
                    return None

                # Combine all memory types into single dict
                memories = {}
                for mem in memories_db:
                    memories[mem.memory_type] = mem.content

                logger.debug("Memories loaded from PostgreSQL", conversation_id=conversation_id)
                return memories

        except Exception as e:
            logger.error("Failed to load memories from PostgreSQL", error=str(e))
            return None

    # === Article Context Cache ===

    async def get_article_context(self, article_id: str) -> Optional[dict[str, Any]]:
        """Get cached article context.

        Args:
            article_id: Article UUID

        Returns:
            Article context dict or None
        """
        return await self.redis.get_json(self._context_key(article_id))

    async def set_article_context(
        self,
        article_id: str,
        context: dict[str, Any],
    ) -> bool:
        """Cache article context for quick loading.

        Args:
            article_id: Article UUID
            context: Article context (title, summary, content, deepsearch)

        Returns:
            True if cached
        """
        success = await self.redis.set_json(
            self._context_key(article_id),
            context,
            ttl=ARTICLE_CONTEXT_TTL,
        )
        if success:
            logger.debug("Article context cached", article_id=article_id)
        return success

    # === History Cache ===

    async def get_cached_history(
        self,
        conversation_id: str,
    ) -> Optional[list[dict[str, Any]]]:
        """Get cached message history.

        Args:
            conversation_id: Conversation UUID

        Returns:
            List of messages or None
        """
        data = await self.redis.get_json(self._history_key(conversation_id))
        if data:
            return data.get("messages", [])
        return None

    async def set_cached_history(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Cache message history.

        Args:
            conversation_id: Conversation UUID
            messages: List of message dicts

        Returns:
            True if cached
        """
        return await self.redis.set_json(
            self._history_key(conversation_id),
            {"messages": messages, "cached_at": datetime.now().isoformat()},
            ttl=HISTORY_TTL,
        )

    # === Incremental Operations ===

    async def append_message(
        self,
        conversation_id: str,
        message: dict[str, Any],
    ) -> bool:
        """Append message to cached session.

        Args:
            conversation_id: Conversation UUID
            message: Message to append

        Returns:
            True if appended
        """
        session = await self.get_session(conversation_id)
        if session:
            session["messages"] = session.get("messages", []) + [message]
            session["message_count"] = session.get("message_count", 0) + 1
            session["last_message_at"] = datetime.now().isoformat()
            return await self.set_session(conversation_id, session)
        return False

    async def update_tokens(
        self,
        conversation_id: str,
        tokens_used: int,
    ) -> bool:
        """Update token count in cached session.

        Args:
            conversation_id: Conversation UUID
            tokens_used: Tokens to add

        Returns:
            True if updated
        """
        session = await self.get_session(conversation_id)
        if session:
            session["current_tokens"] = session.get("current_tokens", 0) + tokens_used
            return await self.set_session(conversation_id, session)
        return False

    # === Token Threshold Check ===

    async def check_context_threshold(
        self,
        conversation_id: str,
    ) -> bool:
        """Check if context threshold is exceeded.

        Args:
            conversation_id: Conversation UUID

        Returns:
            True if threshold exceeded (should compress)
        """
        session = await self.get_session(conversation_id)
        if session:
            current_tokens = session.get("current_tokens", 0)
            threshold = settings.chat_context_threshold
            max_tokens = settings.chat_max_tokens
            return current_tokens >= max_tokens * threshold
        return False

    # === Utility Methods ===

    async def clear_all_chat_caches(self, conversation_id: str) -> bool:
        """Clear all caches for a conversation.

        Args:
            conversation_id: Conversation UUID

        Returns:
            True if all cleared
        """
        await self.invalidate_session(conversation_id)
        await self.redis.delete(self._memory_key(conversation_id))
        await self.redis.delete(self._history_key(conversation_id))
        return True

    @property
    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        return self.redis.is_connected


# Singleton instance
_chat_cache: Optional[ChatCacheService] = None


def get_chat_cache_service() -> ChatCacheService:
    """Get chat cache service instance.

    Returns:
        ChatCacheService singleton
    """
    global _chat_cache
    if _chat_cache is None:
        _chat_cache = ChatCacheService()
    return _chat_cache