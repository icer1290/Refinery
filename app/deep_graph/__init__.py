"""DeepGraph module for GraphRAG integration.

This module provides:
- Background GraphRAG Builder: Extracts entities, relationships, and communities from articles
- On-demand DeepGraph Analyst: Analyzes selected articles with graph expansion
"""

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
    "create_initial_builder_state",
    "create_initial_analyst_state",
]