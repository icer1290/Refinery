"""DeepGraph Analyst graph construction using LangGraph StateGraph."""

from langsmith import traceable
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.context import DeepGraphAnalystContext
from app.deep_graph.state import DeepGraphAnalystState, create_initial_analyst_state
from app.deep_graph.tracing import DEEPGRAPH_TAGS, get_analyst_metadata

logger = get_logger(__name__)
settings = get_settings()


def create_analyst_graph():
    """Create the DeepGraph Analyst workflow graph.

    Linear workflow:
    fetch_articles → fetch_seed_subgraph → expand_subgraph →
    build_visualization → generate_report → END

    Returns:
        Compiled LangGraph workflow with context_schema for dependency injection.
    """
    from langgraph.graph import END, StateGraph

    from app.deep_graph.analyst.nodes import (
        build_visualization_node,
        expand_subgraph_node,
        fetch_articles_node,
        fetch_seed_subgraph_node,
        generate_report_node,
    )

    # Create the graph with context_schema for dependency injection
    graph = StateGraph(DeepGraphAnalystState, context_schema=DeepGraphAnalystContext)

    # Add nodes
    graph.add_node("fetch_articles", fetch_articles_node)
    graph.add_node("fetch_seed_subgraph", fetch_seed_subgraph_node)
    graph.add_node("expand_subgraph", expand_subgraph_node)
    graph.add_node("build_visualization", build_visualization_node)
    graph.add_node("generate_report", generate_report_node)

    # Set entry point
    graph.set_entry_point("fetch_articles")

    # Linear edges
    graph.add_edge("fetch_articles", "fetch_seed_subgraph")
    graph.add_edge("fetch_seed_subgraph", "expand_subgraph")
    graph.add_edge("expand_subgraph", "build_visualization")
    graph.add_edge("build_visualization", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def _get_analyst_metadata_wrapper(args, kwargs):
    """Wrapper to extract metadata from function arguments."""
    return get_analyst_metadata(
        kwargs.get("article_ids", []),
        kwargs.get("max_hops", 2),
        kwargs.get("expansion_limit", 50),
    )


@traceable(
    name="DeepGraphAnalyst_Workflow",
    project_name=settings.langsmith_project,
    tags=DEEPGRAPH_TAGS + ["on-demand"],
    metadata_getter=_get_analyst_metadata_wrapper,
)
async def run_deep_graph_analyst(
    session: AsyncSession,
    article_ids: list[str],
    max_hops: int = 2,
    expansion_limit: int = 50,
) -> DeepGraphAnalystState:
    """Execute the DeepGraph Analyst workflow.

    This function uses LangGraph StateGraph with linear edges.

    Args:
        session: Database session
        article_ids: IDs of selected articles
        max_hops: Maximum hops for graph expansion
        expansion_limit: Maximum entities to add through expansion

    Returns:
        Final analyst state with report and visualization data
    """
    logger.info(
        "Starting DeepGraph Analyst",
        article_count=len(article_ids),
        max_hops=max_hops,
        expansion_limit=expansion_limit,
    )

    # Create initial state
    initial_state = create_initial_analyst_state(
        article_ids=article_ids,
        max_hops=max_hops,
        expansion_limit=expansion_limit,
    )

    try:
        # Create the graph
        graph = create_analyst_graph()

        # Execute the graph with context for dependency injection
        context = DeepGraphAnalystContext(
            session=session,
            article_ids=article_ids,
            max_hops=max_hops,
            expansion_limit=expansion_limit,
        )
        result = await graph.ainvoke(initial_state, context=context)

        result["current_phase"] = "complete"

        logger.info(
            "DeepGraph Analyst completed",
            entities=len(result.get("graph_nodes", [])),
            relationships=len(result.get("graph_edges", [])),
            communities=len(result.get("communities", [])),
            report_length=len(result.get("final_report", "")),
        )

        return result

    except Exception as e:
        logger.error(
            "DeepGraph Analyst failed",
            error=str(e),
        )
        initial_state["errors"] = initial_state.get("errors", []) + [
            {"phase": "orchestration", "message": str(e)}
        ]
        initial_state["current_phase"] = "failed"
        initial_state["final_report"] = f"分析失败: {str(e)}"
        return initial_state