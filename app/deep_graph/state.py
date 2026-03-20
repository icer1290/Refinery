"""State definitions for DeepGraph workflows.

Defines TypedDict states for:
- GraphBuilder: Background knowledge graph construction
- DeepGraphAnalyst: On-demand graph analysis with expansion
"""

import operator
import uuid
from datetime import datetime
from typing import Annotated, Any

from typing_extensions import TypedDict


# === Entity Types ===

class ExtractedEntity(TypedDict):
    """Entity extracted from an article by LLM."""

    name: str
    type: str  # PERSON, ORGANIZATION, TECHNOLOGY, PRODUCT, EVENT, LOCATION, CONCEPT
    description: str
    mentions: list[str]  # Text snippets where entity is mentioned
    confidence: float  # 0.0 - 1.0


class ExtractedRelationship(TypedDict):
    """Relationship extracted from an article by LLM."""

    source_entity: str  # Entity name
    target_entity: str  # Entity name
    relation_type: str  # e.g., "develops", "acquires", "competes_with"
    description: str
    evidence: str  # Text snippet supporting the relationship


class ResolvedEntity(TypedDict):
    """Entity after resolution/deduplication."""

    canonical_name: str
    canonical_type: str
    description: str
    source_entity_names: list[str]  # Original names that were merged
    article_ids: list[str]
    mention_count: int
    embedding: list[float] | None


class Community(TypedDict):
    """Community detected by Leiden algorithm."""

    id: str
    name: str
    summary: str
    entity_ids: list[str]
    hub_entity_id: str | None
    article_ids: list[str]
    level: int


# === Visualization Types ===

class GraphNode(TypedDict):
    """Node for graph visualization."""

    id: str
    label: str
    type: str
    description: str
    mention_count: int
    article_count: int
    is_expanded: bool  # True if added through expansion


class GraphEdge(TypedDict):
    """Edge for graph visualization."""

    id: str
    source: str
    target: str
    relation_type: str
    description: str
    weight: float
    article_count: int
    is_expanded: bool  # True if added through expansion


class CommunityData(TypedDict):
    """Community data for API response."""

    id: str
    name: str
    summary: str
    entity_count: int
    hub_entity: str | None
    article_ids: list[str]


class ExpandedContext(TypedDict):
    """Context for an entity added through graph expansion."""

    entity_id: str
    relevance_score: float  # Combined score for expansion
    similarity_score: float  # Embedding similarity to seed entities
    relationship_weight: float  # Weight of connecting relationships
    community_overlap: float  # Community overlap with seed entities
    hop_distance: int  # Number of hops from seed entities


# === GraphBuilder State ===

class GraphBuilderState(TypedDict):
    """State for the GraphRAG Builder workflow.

    This workflow runs in the background after article storage:
    1. Extract entities from articles
    2. Extract relationships between entities
    3. Resolve/deduplicate entities by vector similarity
    4. Detect communities using Leiden algorithm
    5. Store to database

    Fields:
        run_id: Unique identifier for this builder run
        article_ids: IDs of articles to process

        extracted_entities: Entities extracted from articles (article_id, entity)
        extracted_relationships: Relationships extracted from articles (article_id, relationship)
        resolved_entities: Entities after resolution/deduplication
        detected_communities: Communities detected by Leiden

        current_phase: Current processing phase
        errors: Errors encountered during processing

        entities_count: Total entities extracted
        relationships_count: Total relationships extracted
        communities_count: Total communities detected
    """

    # Input
    run_id: str
    article_ids: list[str]

    # Processing state - use Annotated with operator.add for accumulation
    extracted_entities: Annotated[list[tuple[str, ExtractedEntity]], operator.add]
    extracted_relationships: Annotated[list[tuple[str, ExtractedRelationship]], operator.add]
    resolved_entities: list[ResolvedEntity]
    detected_communities: list[Community]

    # Status
    current_phase: str
    errors: Annotated[list[dict[str, Any]], operator.add]

    # Statistics
    entities_count: int
    relationships_count: int
    communities_count: int

    # Metadata
    started_at: str
    completed_at: str | None


def create_initial_builder_state(
    article_ids: list[str],
) -> GraphBuilderState:
    """Create initial GraphBuilder state.

    Args:
        article_ids: IDs of articles to process

    Returns:
        Initial builder state
    """
    return GraphBuilderState(
        run_id=str(uuid.uuid4()),
        article_ids=article_ids,
        extracted_entities=[],
        extracted_relationships=[],
        resolved_entities=[],
        detected_communities=[],
        current_phase="init",
        errors=[],
        entities_count=0,
        relationships_count=0,
        communities_count=0,
        started_at=datetime.now().isoformat(),
        completed_at=None,
    )


# === DeepGraphAnalyst State ===

class DeepGraphAnalystState(TypedDict):
    """State for the DeepGraph Analyst workflow.

    This workflow runs on-demand when user selects articles:
    1. Fetch seed subgraph (entities/relationships from selected articles)
    2. Expand subgraph (1-hop neighbors with relevance scoring)
    3. Build visualization data (nodes/edges/communities)
    4. Generate comprehensive analysis report

    Fields:
        article_ids: IDs of selected articles
        max_hops: Maximum hops for expansion
        expansion_limit: Maximum entities to add through expansion

        seed_entities: Entity IDs from selected articles
        seed_relationships: Relationship IDs from selected articles
        expanded_entities: Entities added through expansion with context
        expanded_relationships: Relationship IDs added through expansion

        graph_nodes: Nodes for visualization
        graph_edges: Edges for visualization
        communities: Community data for visualization

        final_report: Generated analysis report
        visualization_data: Data for frontend visualization

        current_phase: Current processing phase
        errors: Errors encountered during processing
    """

    # Input
    article_ids: list[str]
    max_hops: int
    expansion_limit: int

    # Seed subgraph (from selected articles)
    seed_entities: list[str]  # Entity IDs
    seed_relationships: list[str]  # Relationship IDs

    # Expanded subgraph
    expanded_entities: list[ExpandedContext]
    expanded_relationships: list[str]

    # Visualization data
    graph_nodes: list[GraphNode]
    graph_edges: list[GraphEdge]
    communities: list[CommunityData]

    # Output
    final_report: str
    visualization_data: dict[str, Any]

    # Status
    current_phase: str
    errors: Annotated[list[dict[str, Any]], operator.add]


def create_initial_analyst_state(
    article_ids: list[str],
    max_hops: int = 2,
    expansion_limit: int = 50,
) -> DeepGraphAnalystState:
    """Create initial DeepGraph Analyst state.

    Args:
        article_ids: IDs of selected articles
        max_hops: Maximum hops for expansion
        expansion_limit: Maximum entities to add through expansion

    Returns:
        Initial analyst state
    """
    return DeepGraphAnalystState(
        article_ids=article_ids,
        max_hops=max_hops,
        expansion_limit=expansion_limit,
        seed_entities=[],
        seed_relationships=[],
        expanded_entities=[],
        expanded_relationships=[],
        graph_nodes=[],
        graph_edges=[],
        communities=[],
        final_report="",
        visualization_data={},
        current_phase="init",
        errors=[],
    )