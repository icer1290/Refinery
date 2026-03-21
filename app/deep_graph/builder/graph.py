"""GraphRAG Builder graph construction using LangGraph StateGraph."""

import uuid
from datetime import datetime, timezone

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langsmith import traceable
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.context import DeepGraphBuilderContext
from app.deep_graph.graph_store import graph_store
from app.deep_graph.state import GraphBuilderState, create_initial_builder_state
from app.deep_graph.tracing import DEEPGRAPH_TAGS, get_builder_metadata

logger = get_logger(__name__)
settings = get_settings()


def create_builder_graph():
    """Create the GraphRAG Builder workflow graph.

    Linear workflow:
    fetch_articles → extract_entities → extract_relationships →
    resolve_entities → store_graph → detect_communities → END

    Returns:
        Compiled LangGraph workflow with context_schema for dependency injection.
    """
    from app.deep_graph.builder.nodes import (
        detect_communities_node,
        extract_entities_node,
        extract_relationships_node,
        fetch_articles_node,
        resolve_entities_node,
        store_graph_node,
    )

    # Create the graph with context_schema for dependency injection
    graph = StateGraph(GraphBuilderState, context_schema=DeepGraphBuilderContext)

    # Add nodes
    graph.add_node("fetch_articles", fetch_articles_node)
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("extract_relationships", extract_relationships_node)
    graph.add_node("resolve_entities", resolve_entities_node)
    graph.add_node("store_graph", store_graph_node)
    graph.add_node("detect_communities", detect_communities_node)

    # Set entry point
    graph.set_entry_point("fetch_articles")

    # Linear edges
    graph.add_edge("fetch_articles", "extract_entities")
    graph.add_edge("extract_entities", "extract_relationships")
    graph.add_edge("extract_relationships", "resolve_entities")
    graph.add_edge("resolve_entities", "store_graph")
    graph.add_edge("store_graph", "detect_communities")
    graph.add_edge("detect_communities", END)

    return graph.compile()


def _get_builder_metadata_wrapper(args, kwargs):
    """Wrapper to extract metadata from function arguments."""
    return get_builder_metadata(kwargs.get("article_ids", []))


@traceable(
    name="GraphBuilder_Workflow",
    project_name=settings.langsmith_project,
    tags=DEEPGRAPH_TAGS + ["orchestration"],
    metadata_getter=_get_builder_metadata_wrapper,
)
async def run_graph_builder(
    session: AsyncSession,
    article_ids: list[str],
) -> GraphBuilderState:
    """Execute the GraphRAG Builder workflow.

    This function uses LangGraph StateGraph with linear edges.

    Args:
        session: Database session
        article_ids: IDs of articles to process

    Returns:
        Final builder state with results
    """
    logger.info(
        "Starting GraphRAG Builder",
        article_count=len(article_ids),
    )

    # Create initial state
    initial_state = create_initial_builder_state(article_ids=article_ids)

    try:
        # Create builder run record
        run = await graph_store.create_builder_run(
            session=session,
            article_ids=[uuid.UUID(aid) for aid in article_ids],
        )
        initial_state["run_id"] = str(run.id)
        await session.commit()

        # Create the graph
        graph = create_builder_graph()

        # Execute the graph with context for dependency injection
        context = DeepGraphBuilderContext(
            session=session,
            run_id=str(run.id),
            article_ids=article_ids,
        )
        result = await graph.ainvoke(initial_state, context=context)

        # Mark as complete
        result["current_phase"] = "complete"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _complete_run(session, run.id, result, "completed")

        logger.info(
            "GraphRAG Builder completed",
            run_id=run.id,
            entities=result.get("entities_count", 0),
            relationships=result.get("relationships_count", 0),
            communities=result.get("communities_count", 0),
        )

        return result

    except Exception as e:
        logger.error(
            "GraphRAG Builder failed",
            run_id=initial_state.get("run_id"),
            error=str(e),
        )
        # Rollback to clean state
        await session.rollback()
        initial_state["errors"] = initial_state.get("errors", []) + [
            {"phase": "orchestration", "message": str(e)}
        ]
        initial_state["current_phase"] = "failed"

        # Try to update the run record
        if initial_state.get("run_id"):
            try:
                await _complete_run(
                    session, uuid.UUID(initial_state["run_id"]), initial_state, "failed"
                )
            except Exception:
                pass

        return initial_state


async def _complete_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    state: GraphBuilderState,
    status: str,
) -> None:
    """Update builder run record with final status."""
    await graph_store.update_builder_run(
        session=session,
        run_id=run_id,
        status=status,
        entities_count=state.get("entities_count", 0),
        relationships_count=state.get("relationships_count", 0),
        communities_count=state.get("communities_count", 0),
        errors=state.get("errors"),
    )
    await session.commit()


async def run_graph_builder_background(article_ids: list[str]) -> None:
    """Run graph builder in background task.

    Creates its own database session and runs the builder
    independently of any request context.

    Args:
        article_ids: IDs of articles to process
    """
    from app.models.database import get_async_session

    logger.info(
        "Starting background GraphRAG Builder",
        article_count=len(article_ids),
    )

    try:
        async for session in get_async_session():
            await run_graph_builder(session, article_ids)
            break
    except Exception as e:
        logger.error(
            "Background GraphRAG Builder failed",
            error=str(e),
        )