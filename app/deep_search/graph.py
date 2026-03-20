"""Graph construction and execution for deep search workflow."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.deep_search.nodes import (
    conclude_node,
    fetch_article_node,
    reasoning_node,
    tools_node,
)
from app.deep_search.state import DeepSearchState, create_initial_deep_search_state
from app.models.orm_models import NewsArticle

logger = get_logger(__name__)


async def run_deep_search(
    session: AsyncSession,
    article_id: str,
    max_iterations: int = 5,
) -> DeepSearchState:
    """Execute the deep search workflow.

    This function implements a manual ReAct loop following the pattern
    used in the existing workflow system.

    Args:
        session: Database session
        article_id: ID of the article to analyze
        max_iterations: Maximum number of ReAct iterations

    Returns:
        Final deep search state with report
    """
    logger.info(
        "Starting deep search",
        article_id=article_id,
        max_iterations=max_iterations,
    )

    # Create initial state
    state = create_initial_deep_search_state(
        article_id=article_id,
        max_iterations=max_iterations,
    )

    # Phase 1: Fetch article
    state.update(await fetch_article_node(state, session))

    if not state.get("article"):
        logger.error("Article not found, aborting deep search")
        return state

    # Phase 2: ReAct Loop
    while state["should_continue"] and state["current_iteration"] < state["max_iterations"]:
        # Reasoning step
        state.update(await reasoning_node(state))

        # Check if should continue
        if not state["should_continue"]:
            break

        # Tools execution step
        if state.get("_pending_action"):
            state.update(await tools_node(state, session))

    # Phase 3: Generate report
    state.update(await conclude_node(state))

    # Save deepsearch results to database
    if state.get("is_complete") and state.get("final_report"):
        try:
            # Rollback to reset transaction state if any previous operation failed
            # This is safe because we haven't made any writes yet
            await session.rollback()

            stmt = select(NewsArticle).where(NewsArticle.id == UUID(article_id))
            result = await session.execute(stmt)
            article = result.scalar_one_or_none()

            if article:
                article.deepsearch_report = state["final_report"]
                article.deepsearch_performed_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info("DeepSearch results saved to database", article_id=article_id)
            else:
                logger.warning("Article not found for saving deepsearch results", article_id=article_id)
        except Exception as e:
            logger.error("Failed to save deepsearch results", error=str(e), article_id=article_id)

    logger.info(
        "Deep search completed",
        article_id=article_id,
        iterations=state["current_iteration"],
        tools_used=len(state.get("tool_history", [])),
        errors=len(state.get("errors", [])),
    )

    return state