"""Tool for retrieving conversation history."""

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.models.orm_models import ChatMessage
from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = get_logger(__name__)


class ConversationHistoryTool(BaseTool):
    """Tool for retrieving conversation history.

    Fetches recent messages from a conversation for context.
    """

    name = "conversation_history"
    description = """Retrieve previous messages from this conversation thread.
Use when context of the conversation is needed.

Input:
- conversation_id: UUID of the conversation (required)
- limit: Maximum number of messages to retrieve (default: 10)
"""

    async def execute(
        self,
        session: AsyncSession,
        conversation_id: str,
        limit: int = 10,
    ) -> str:
        """Fetch conversation history.

        Args:
            session: Database session
            conversation_id: Conversation UUID
            limit: Max messages to return

        Returns:
            JSON string with message history
        """
        try:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.conversation_id == UUID(conversation_id))
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            messages = result.scalars().all()

            if not messages:
                return json.dumps({
                    "conversation_id": conversation_id,
                    "messages": [],
                    "count": 0,
                })

            # Reverse to chronological order
            messages_list = []
            for msg in reversed(messages):
                messages_list.append({
                    "role": msg.role,
                    "content": msg.content[:500] if len(msg.content) > 500 else msg.content,
                    "agent_name": msg.agent_name,
                    "created_at": str(msg.created_at),
                })

            return json.dumps({
                "conversation_id": conversation_id,
                "messages": messages_list,
                "count": len(messages_list),
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Conversation history fetch failed", error=str(e))
            return json.dumps({"error": str(e)})


# Register tool at module load time
register_tool(ConversationHistoryTool())