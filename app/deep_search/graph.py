"""Graph construction and execution for deep search workflow using LangGraph StateGraph."""

from datetime import datetime, timezone
from uuid import UUID

from langsmith import traceable
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_search.context import DeepSearchContext
from app.deep_search.nodes import (
    conclude_node,
    fetch_article_node,
    reasoning_node,
    tools_node,
)
from app.deep_search.state import DeepSearchState, create_initial_deep_search_state
from app.models.orm_models import NewsArticle

logger = get_logger(__name__)
settings = get_settings()


def _route_after_reasoning(state: DeepSearchState) -> str:
    """Route after reasoning node.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    if not state.get("should_continue"):
        return "conclude"
    if state.get("pending_action"):
        return "tools"
    return "conclude"


def _route_after_tools(state: DeepSearchState) -> str:
    """Route after tools node.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    if state["current_iteration"] >= state["max_iterations"]:
        return "conclude"
    return "reasoning"


def create_deep_search_graph():
    """Create the deep search workflow graph.

    Returns:
        Compiled LangGraph workflow with context_schema for dependency injection.
    """
    # Create the graph with context_schema for dependency injection
    graph = StateGraph(DeepSearchState, context_schema=DeepSearchContext)

    # Add nodes
    graph.add_node("fetch_article", fetch_article_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tools_node)
    graph.add_node("conclude", conclude_node)

    # Set entry point
    graph.set_entry_point("fetch_article")

    # Linear edge from fetch to reasoning
    graph.add_edge("fetch_article", "reasoning")

    # Conditional edges from reasoning
    graph.add_conditional_edges(
        "reasoning",
        _route_after_reasoning,
        {
            "tools": "tools",
            "conclude": "conclude",
        },
    )

    # Conditional edges from tools
    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "reasoning": "reasoning",
            "conclude": "conclude",
        },
    )

    # End after conclude
    graph.add_edge("conclude", END)

    return graph.compile()


@traceable(name="DeepSearch", project_name=settings.langsmith_project)
async def run_deep_search(
    session: AsyncSession,
    article_id: str,
    max_iterations: int = 5,
) -> DeepSearchState:
    """Execute the deep search workflow.

    This function uses LangGraph StateGraph with conditional edges for
    the ReAct loop pattern.

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
    initial_state = create_initial_deep_search_state(
        article_id=article_id,
        max_iterations=max_iterations,
    )

    # Create the graph
    graph = create_deep_search_graph()

    try:
        # Execute the graph with context for dependency injection
        context = DeepSearchContext(session=session, article_id=article_id)
        result = await graph.ainvoke(initial_state, context=context)

        # Save deepsearch results to database
        if result.get("is_complete") and result.get("final_report"):
            try:
                # Rollback to reset transaction state if any previous operation failed
                await session.rollback()

                stmt = select(NewsArticle).where(NewsArticle.id == UUID(article_id))
                db_result = await session.execute(stmt)
                article = db_result.scalar_one_or_none()

                if article:
                    article.deepsearch_report = result["final_report"]
                    article.deepsearch_performed_at = datetime.now(timezone.utc)
                    await session.commit()
                    logger.info("DeepSearch results saved to database", article_id=article_id)
                else:
                    logger.warning(
                        "Article not found for saving deepsearch results", article_id=article_id
                    )
            except Exception as e:
                logger.error("Failed to save deepsearch results", error=str(e), article_id=article_id)

        logger.info(
            "Deep search completed",
            article_id=article_id,
            iterations=result.get("current_iteration", 0),
            tools_used=len(result.get("tool_history", [])),
            errors=len(result.get("errors", [])),
        )

        return result

    except Exception as e:
        logger.error("Deep search failed", error=str(e), article_id=article_id)
        initial_state["errors"] = initial_state.get("errors", []) + [
            {"phase": "orchestration", "message": str(e)}
        ]
        initial_state["is_complete"] = True
        return initial_state