"""Chat API endpoints for multi-turn conversations."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.chat.graph import run_chat
from app.config import get_settings
from app.core import get_logger
from app.models.orm_models import ChatConversation, ChatMessage, NewsArticle
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatMessageResponse,
    ConversationCreateRequest,
    ConversationResponse,
    ConversationListResponse,
    ConversationDetailResponse,
)

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Create a new conversation thread for an article.

    Args:
        request: Conversation creation request
        db: Database session

    Returns:
        Created conversation details
    """
    try:
        # Verify article exists
        stmt = select(NewsArticle).where(NewsArticle.id == uuid.UUID(request.article_id))
        result = await db.execute(stmt)
        article = result.scalar_one_or_none()

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        # Check if conversation already exists for this article/user
        existing_stmt = select(ChatConversation).where(
            ChatConversation.article_id == uuid.UUID(request.article_id),
            ChatConversation.user_id == request.user_id,
            ChatConversation.status == "active",
        )
        existing_result = await db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            return ConversationResponse.model_validate(existing)

        # Create new conversation
        conversation = ChatConversation(
            article_id=uuid.UUID(request.article_id),
            user_id=request.user_id,
            title=request.title,
            status="active",
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        logger.info(
            "Conversation created",
            conversation_id=str(conversation.id),
            article_id=request.article_id,
            user_id=request.user_id,
        )

        return ConversationResponse.model_validate(conversation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create conversation", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
) -> ConversationDetailResponse:
    """Get conversation details with article context.

    Args:
        conversation_id: Conversation UUID
        db: Database session

    Returns:
        Conversation details with article info
    """
    try:
        stmt = select(ChatConversation).where(
            ChatConversation.id == uuid.UUID(conversation_id)
        )
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get article info
        article_stmt = select(NewsArticle).where(
            NewsArticle.id == conversation.article_id
        )
        article_result = await db.execute(article_stmt)
        article = article_result.scalar_one_or_none()

        response_dict = {
            "id": conversation.id,
            "article_id": conversation.article_id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "status": conversation.status,
            "message_count": conversation.message_count,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "last_message_at": conversation.last_message_at,
            "article_title": article.chinese_title if article else None,
            "article_summary": article.chinese_summary if article else None,
            "has_deepsearch": bool(article.deepsearch_report) if article else False,
        }

        return ConversationDetailResponse(**response_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get conversation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    user_id: int = Query(..., description="User ID"),
    article_id: str | None = Query(None, description="Filter by article ID"),
    status: str = Query("active", description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> ConversationListResponse:
    """List conversations for a user.

    Args:
        user_id: User ID (required)
        article_id: Optional article filter
        status: Status filter (default: active)
        page: Page number
        page_size: Items per page
        db: Database session

    Returns:
        Paginated list of conversations
    """
    try:
        # Build query
        stmt = select(ChatConversation).where(
            ChatConversation.user_id == user_id,
            ChatConversation.status == status,
        )

        if article_id:
            stmt = stmt.where(ChatConversation.article_id == uuid.UUID(article_id))

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await db.scalar(count_stmt) or 0

        # Apply pagination
        offset = (page - 1) * page_size
        stmt = stmt.order_by(ChatConversation.updated_at.desc()).offset(offset).limit(page_size)

        result = await db.execute(stmt)
        conversations = result.scalars().all()

        return ConversationListResponse(
            conversations=[ConversationResponse.model_validate(c) for c in conversations],
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Send a message in a conversation and get AI response.

    This endpoint:
    1. Loads conversation context (article + history)
    2. Runs the chat graph (supervisor -> specialist -> memory)
    3. Returns response with citations

    Args:
        request: Chat request with message
        db: Database session

    Returns:
        Chat response with AI reply
    """
    try:
        # Get conversation
        stmt = select(ChatConversation).where(
            ChatConversation.id == uuid.UUID(request.conversation_id)
        )
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.status != "active":
            raise HTTPException(status_code=400, detail="Conversation is not active")

        # Run chat workflow
        state = await run_chat(
            session=db,
            conversation_id=request.conversation_id,
            article_id=str(conversation.article_id),
            user_id=conversation.user_id,
            user_message=request.message,
        )

        # Get the last assistant message
        msg_stmt = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == uuid.UUID(request.conversation_id))
            .where(ChatMessage.role == "assistant")
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
        msg_result = await db.execute(msg_stmt)
        assistant_msg = msg_result.scalar_one_or_none()

        if not assistant_msg:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        return ChatResponse(
            conversation_id=request.conversation_id,
            message_id=str(assistant_msg.id),
            response=assistant_msg.content,
            agent_used=assistant_msg.agent_name or "explainer",
            citations=assistant_msg.citations or [],
            tool_calls=assistant_msg.tool_calls or [],
            tokens_used=assistant_msg.tokens_used,
            created_at=assistant_msg.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=200),
    before: str | None = Query(None, description="Message ID for cursor pagination"),
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    """Get chat message history with pagination.

    Args:
        conversation_id: Conversation UUID
        limit: Maximum messages to return
        before: Cursor for pagination (message ID)
        db: Database session

    Returns:
        Message history
    """
    try:
        # Build query
        stmt = select(ChatMessage).where(
            ChatMessage.conversation_id == uuid.UUID(conversation_id)
        )

        if before:
            # Get messages before this ID
            stmt = stmt.where(ChatMessage.id < uuid.UUID(before))

        # Get total count
        count_stmt = select(func.count()).where(
            ChatMessage.conversation_id == uuid.UUID(conversation_id)
        )
        total = await db.scalar(count_stmt) or 0

        # Apply limit and order
        stmt = stmt.order_by(ChatMessage.created_at.desc()).limit(limit + 1)

        result = await db.execute(stmt)
        messages = result.scalars().all()

        # Check if there are more messages
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        # Convert to response format (reverse to chronological order)
        message_responses = [
            ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                agent_name=msg.agent_name,
                citations=msg.citations,
                tokens_used=msg.tokens_used,
                created_at=msg.created_at,
            )
            for msg in reversed(messages)
        ]

        return ChatHistoryResponse(
            conversation_id=conversation_id,
            messages=message_responses,
            total=total,
            has_more=has_more,
        )

    except Exception as e:
        logger.error("Failed to get history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def archive_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Archive a conversation (soft delete).

    Args:
        conversation_id: Conversation UUID
        db: Database session

    Returns:
        Success message
    """
    try:
        stmt = select(ChatConversation).where(
            ChatConversation.id == uuid.UUID(conversation_id)
        )
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.status = "archived"
        await db.commit()

        logger.info("Conversation archived", conversation_id=conversation_id)

        return {"status": "archived", "conversation_id": conversation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to archive conversation", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))