"""DeepGraph Analyst orchestration for on-demand graph analysis.

This workflow runs on-demand when user selects articles:
1. Fetch articles
2. Fetch seed subgraph (entities/relationships from selected articles)
3. Expand subgraph (1-hop neighbors with relevance scoring)
4. Build visualization data
5. Generate comprehensive analysis report
"""

from langsmith import traceable
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.state import DeepGraphAnalystState, create_initial_analyst_state
from app.deep_graph.nodes_analyst import (
    fetch_articles_for_analyst,
    fetch_seed_subgraph_node,
    expand_subgraph_node,
    build_visualization_node,
    generate_report_node,
)
from app.deep_graph.tracing import DEEPGRAPH_TAGS, get_analyst_metadata

logger = get_logger(__name__)
settings = get_settings()


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

    This function implements a manual orchestration following
    the pattern used in deep_search/graph.py.

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
    state = create_initial_analyst_state(
        article_ids=article_ids,
        max_hops=max_hops,
        expansion_limit=expansion_limit,
    )

    try:
        # Phase 1: Fetch articles
        state.update(await fetch_articles_for_analyst(state, session))

        if not state.get("_articles"):
            logger.warning("No articles found for analysis")
            return state

        # Phase 2: Fetch seed subgraph
        state.update(await fetch_seed_subgraph_node(state, session))

        if not state.get("seed_entities"):
            logger.warning("No entities found for selected articles")
            # Continue anyway - report will mention no entities

        # Phase 3: Expand subgraph
        state.update(await expand_subgraph_node(state, session))

        # Phase 4: Build visualization data
        state.update(await build_visualization_node(state, session))

        # Phase 5: Generate report
        state.update(await generate_report_node(state))

        state["current_phase"] = "complete"

        logger.info(
            "DeepGraph Analyst completed",
            entities=len(state.get("graph_nodes", [])),
            relationships=len(state.get("graph_edges", [])),
            communities=len(state.get("communities", [])),
            report_length=len(state.get("final_report", "")),
        )

    except Exception as e:
        logger.error(
            "DeepGraph Analyst failed",
            error=str(e),
        )
        state["errors"] = state.get("errors", []) + [{"phase": "orchestration", "message": str(e)}]
        state["current_phase"] = "failed"
        state["final_report"] = f"分析失败: {str(e)}"

    return state