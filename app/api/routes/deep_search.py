"""Deep search API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core import get_logger
from app.deep_search.graph import run_deep_search
from app.models.schemas import (
    CollectedInfoResponse,
    DeepSearchRequest,
    DeepSearchResponse,
    ToolCallInfo,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post("/deepSearch", response_model=DeepSearchResponse)
async def execute_deep_search(
    request: DeepSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> DeepSearchResponse:
    """Execute deep search for an article.

    This endpoint triggers a ReAct-style workflow to collect background
    information and generate a comprehensive tracking report.

    Args:
        request: Deep search request with article_id and max_iterations
        db: Database session

    Returns:
        Deep search response with final report and tool history
    """
    logger.info(
        "Deep search requested",
        article_id=request.article_id,
        max_iterations=request.max_iterations,
    )

    try:
        # Validate UUID format
        import uuid
        try:
            uuid.UUID(request.article_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid article_id format. Must be a valid UUID.",
            )

        # Execute deep search
        state = await run_deep_search(
            session=db,
            article_id=request.article_id,
            max_iterations=request.max_iterations,
        )

        # Check if article was found
        if not state.get("article"):
            raise HTTPException(
                status_code=404,
                detail="Article not found",
            )

        # Build response
        article = state.get("article", {})
        tool_calls = [
            ToolCallInfo(
                tool_name=tc["tool_name"],
                tool_input=tc["tool_input"],
                tool_output=tc["tool_output"],
                iteration=tc["iteration"],
            )
            for tc in state.get("tool_history", [])
        ]

        collected_info = [
            CollectedInfoResponse(
                source=ci["source"],
                content=ci["content"],
                relevance=ci["relevance"],
                metadata=ci["metadata"],
            )
            for ci in state.get("collected_info", [])
        ]

        return DeepSearchResponse(
            article_id=request.article_id,
            article_title=article.get("title", ""),
            final_report=state.get("final_report", ""),
            tools_used=tool_calls,
            collected_info=collected_info,
            iterations=state.get("current_iteration", 0),
            is_complete=state.get("is_complete", False),
            errors=state.get("errors", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Deep search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Deep search failed: {str(e)}")