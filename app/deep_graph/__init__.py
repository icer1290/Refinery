"""DeepGraph module for GraphRAG integration.

This module provides:
- Background GraphRAG Builder: Extracts entities, relationships, and communities from articles
- On-demand DeepGraph Analyst: Analyzes selected articles with graph expansion
"""

# Import from new module locations
from app.deep_graph.builder import run_graph_builder, create_builder_graph
from app.deep_graph.builder.graph import run_graph_builder_background
from app.deep_graph.analyst import run_deep_graph_analyst, create_analyst_graph
from app.deep_graph.state import (
    GraphBuilderState,
    DeepGraphAnalystState,
    ExtractedEntity,
    ExtractedRelationship,
    ResolvedEntity,
    Community,
    GraphNode,
    GraphEdge,
    CommunityData,
    ExpandedContext,
    create_initial_builder_state,
    create_initial_analyst_state,
)

__all__ = [
    # Main entry points
    "run_graph_builder",
    "run_graph_builder_background",
    "run_deep_graph_analyst",
    "create_builder_graph",
    "create_analyst_graph",
    # State types
    "GraphBuilderState",
    "DeepGraphAnalystState",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ResolvedEntity",
    "Community",
    "GraphNode",
    "GraphEdge",
    "CommunityData",
    "ExpandedContext",
    # State factories
    "create_initial_builder_state",
    "create_initial_analyst_state",
]