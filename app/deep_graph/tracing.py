"""LangSmith tracing utilities for DeepGraph module."""

from app.config import get_settings

settings = get_settings()

# Common tags for DeepGraph traces
DEEPGRAPH_TAGS = ["deepgraph", "graphrag"]


def get_builder_metadata(article_ids: list[str]) -> dict:
    """Get metadata for GraphBuilder workflow trace."""
    return {
        "article_count": len(article_ids),
        "workflow_type": "graph_builder",
    }


def get_analyst_metadata(article_ids: list[str], max_hops: int, expansion_limit: int) -> dict:
    """Get metadata for DeepGraph Analyst workflow trace."""
    return {
        "article_count": len(article_ids),
        "max_hops": max_hops,
        "expansion_limit": expansion_limit,
        "workflow_type": "graph_analyst",
    }