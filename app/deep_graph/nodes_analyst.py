"""Node implementations for DeepGraph Analyst workflow.

Nodes for on-demand graph analysis:
1. fetch_seed_subgraph_node: Get entities/relationships from selected articles
2. expand_subgraph_node: Find 1-hop neighbors with relevance scoring
3. build_visualization_node: Create graph nodes/edges for visualization
4. generate_report_node: LLM generates comprehensive analysis report
"""

import uuid
from typing import Any
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langsmith import traceable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.graph_store import graph_store
from app.deep_graph.state import (
    DeepGraphAnalystState,
    GraphNode,
    GraphEdge,
    CommunityData,
    ExpandedContext,
)
from app.deep_graph.prompts import (
    DEEP_GRAPH_REPORT_PROMPT,
    format_graph_for_report,
    format_articles_for_report,
)
from app.models.orm_models import NewsArticle, GraphEntity, GraphRelationship, GraphCommunity
from app.services.embedding import get_embedding_service

logger = get_logger(__name__)
settings = get_settings()


@traceable(name="Fetch_Articles_For_Analyst", run_type="tool")
async def fetch_articles_for_analyst(
    state: DeepGraphAnalystState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Fetch article details for analysis.

    Args:
        state: Current analyst state
        session: Database session

    Returns:
        Updated state fields
    """
    logger.info(
        "Fetching articles for DeepGraph analysis",
        article_count=len(state["article_ids"]),
    )

    try:
        article_uuids = [uuid.UUID(aid) for aid in state["article_ids"]]
        stmt = select(NewsArticle).where(NewsArticle.id.in_(article_uuids))
        result = await session.execute(stmt)
        articles = list(result.scalars().all())

        article_dicts = []
        for article in articles:
            article_dicts.append({
                "id": str(article.id),
                "title": article.chinese_title or article.original_title,
                "original_title": article.original_title,
                "content": article.full_content,
                "summary": article.chinese_summary or article.original_description,
                "source": article.source_name,
                "published_at": str(article.published_at) if article.published_at else "Unknown",
            })

        logger.info("Articles fetched", count=len(article_dicts))

        return {
            "_articles": article_dicts,
            "current_phase": "fetch_articles_complete",
        }

    except Exception as e:
        logger.error("Failed to fetch articles", error=str(e))
        return {
            "errors": [{"phase": "fetch_articles", "message": str(e)}],
            "current_phase": "fetch_articles_failed",
        }


@traceable(name="Fetch_Seed_Subgraph", run_type="tool")
async def fetch_seed_subgraph_node(
    state: DeepGraphAnalystState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Fetch seed entities and relationships from selected articles.

    Args:
        state: Current analyst state
        session: Database session

    Returns:
        Updated state fields with seed entity/relationship IDs
    """
    logger.info("Fetching seed subgraph")

    article_uuids = [uuid.UUID(aid) for aid in state["article_ids"]]

    try:
        # Get entities associated with these articles
        seed_entities = await graph_store.get_entities_by_articles(
            session=session,
            article_ids=article_uuids,
        )

        # Get relationships associated with these articles
        seed_relationships = await graph_store.get_relationships_by_articles(
            session=session,
            article_ids=article_uuids,
        )

        # Store IDs in state
        seed_entity_ids = [str(e.id) for e in seed_entities]
        seed_rel_ids = [str(r.id) for r in seed_relationships]

        logger.info(
            "Seed subgraph fetched",
            entity_count=len(seed_entity_ids),
            relationship_count=len(seed_rel_ids),
        )

        return {
            "seed_entities": seed_entity_ids,
            "seed_relationships": seed_rel_ids,
            "_seed_entity_objects": [
                {
                    "id": str(e.id),
                    "name": e.name,
                    "canonical_name": e.canonical_name,
                    "type": e.type,
                    "description": e.description,
                    "article_ids": [str(aid) for aid in (e.article_ids or [])],
                    "mention_count": e.mention_count,
                    "embedding": e.embedding,
                }
                for e in seed_entities
            ],
            "_seed_rel_objects": seed_relationships,
            "current_phase": "fetch_seed_complete",
        }

    except Exception as e:
        logger.error("Failed to fetch seed subgraph", error=str(e))
        return {
            "errors": [{"phase": "fetch_seed", "message": str(e)}],
            "current_phase": "fetch_seed_failed",
        }


@traceable(name="Expand_Subgraph", run_type="tool")
async def expand_subgraph_node(
    state: DeepGraphAnalystState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Expand subgraph by finding 1-hop neighbors with relevance scoring.

    Expansion algorithm:
    1. Get seed entities
    2. For each hop:
       a. Get 1-hop neighbors
       b. Calculate relevance score
       c. Filter by threshold, add top-N to expanded set
    3. Return expanded entities with context

    Args:
        state: Current analyst state
        session: Database session

    Returns:
        Updated state fields with expanded entities
    """
    logger.info(
        "Expanding subgraph",
        max_hops=state["max_hops"],
        expansion_limit=state["expansion_limit"],
    )

    seed_entity_ids = [uuid.UUID(eid) for eid in state["seed_entities"]]
    if not seed_entity_ids:
        return {
            "expanded_entities": [],
            "current_phase": "expand_complete",
        }

    try:
        embedding_service = get_embedding_service()

        # Get seed entities with embeddings
        seed_entity_objects = state.get("_seed_entity_objects", [])
        seed_embeddings = []
        for e in seed_entity_objects:
            if e.get("embedding"):
                seed_embeddings.append(e["embedding"])

        # Calculate average seed embedding for similarity comparison
        if seed_embeddings:
            avg_seed_embedding = [
                sum(vals) / len(vals)
                for vals in zip(*seed_embeddings)
            ]
        else:
            avg_seed_embedding = None

        # Track visited entities (seed entities are already visited)
        visited = set(state["seed_entities"])
        frontier = list(seed_entity_ids)
        expanded_entities: list[ExpandedContext] = []
        all_expanded_rels = []

        # Get seed communities for community overlap calculation
        seed_communities = await graph_store.get_communities_by_entities(
            session=session,
            entity_ids=seed_entity_ids,
        )
        seed_community_entity_sets = [
            set(str(eid) for eid in c.entity_ids)
            for c in seed_communities
        ]

        # Hop-by-hop expansion
        for hop in range(state["max_hops"]):
            if not frontier:
                break

            logger.debug(f"Hop {hop + 1}: exploring {len(frontier)} entities")

            # Get 1-hop neighbors
            neighbors, relationships = await graph_store.get_1hop_neighbors(
                session=session,
                entity_ids=frontier,
                limit=state["expansion_limit"] * 2,
            )

            if not neighbors:
                break

            all_expanded_rels.extend(relationships)

            # Score and filter neighbors
            scored_neighbors = []
            for neighbor in neighbors:
                neighbor_id = str(neighbor.id)

                if neighbor_id in visited:
                    continue

                # Calculate relevance score
                # 0.4 × embedding similarity + 0.3 × relationship weight + 0.3 × community overlap
                similarity_score = 0.0
                if avg_seed_embedding is not None and neighbor.embedding is not None:
                    try:
                        # Ensure embeddings are lists of floats
                        emb1 = list(avg_seed_embedding) if not isinstance(avg_seed_embedding, list) else avg_seed_embedding
                        emb2 = list(neighbor.embedding) if not isinstance(neighbor.embedding, list) else neighbor.embedding

                        if len(emb1) > 0 and len(emb2) > 0:
                            # Cosine similarity
                            dot_product = sum(a * b for a, b in zip(emb1, emb2))
                            norm_a = sum(a * a for a in emb1) ** 0.5
                            norm_b = sum(b * b for b in emb2) ** 0.5
                            if norm_a > 0 and norm_b > 0:
                                similarity_score = dot_product / (norm_a * norm_b)
                    except Exception as e:
                        logger.debug(f"Failed to compute similarity: {e}")

                # Average relationship weight
                rel_weight = 0.0
                neighbor_rels = [
                    r for r in relationships
                    if r.source_entity_id == neighbor.id or r.target_entity_id == neighbor.id
                ]
                if len(neighbor_rels) > 0:
                    rel_weight = sum(float(r.weight) if r.weight else 1.0 for r in neighbor_rels) / len(neighbor_rels)

                # Community overlap
                community_overlap = 0.0
                neighbor_community_set = {neighbor_id}
                for comm_entities in seed_community_entity_sets:
                    if neighbor_id in comm_entities:
                        community_overlap = 1.0
                        break

                # Combined relevance score
                relevance_score = (
                    0.4 * similarity_score +
                    0.3 * min(rel_weight / 5.0, 1.0) +
                    0.3 * community_overlap
                )

                scored_neighbors.append({
                    "neighbor": neighbor,
                    "relevance_score": relevance_score,
                    "similarity_score": similarity_score,
                    "relationship_weight": rel_weight,
                    "community_overlap": community_overlap,
                    "hop_distance": hop + 1,
                })

            # Sort by relevance and take top-N
            scored_neighbors.sort(key=lambda x: x["relevance_score"], reverse=True)

            remaining_limit = state["expansion_limit"] - len(expanded_entities)
            for item in scored_neighbors[:remaining_limit]:
                neighbor = item["neighbor"]
                neighbor_id = str(neighbor.id)
                visited.add(neighbor_id)

                expanded_entities.append(ExpandedContext(
                    entity_id=neighbor_id,
                    relevance_score=item["relevance_score"],
                    similarity_score=item["similarity_score"],
                    relationship_weight=item["relationship_weight"],
                    community_overlap=item["community_overlap"],
                    hop_distance=item["hop_distance"],
                ))

            # Update frontier for next hop
            frontier = [uuid.UUID(e["entity_id"]) for e in expanded_entities[-len(scored_neighbors[:remaining_limit]):]]

        logger.info(
            "Subgraph expansion complete",
            expanded_count=len(expanded_entities),
            total_rels=len(all_expanded_rels),
        )

        return {
            "expanded_entities": expanded_entities,
            "expanded_relationships": [str(r.id) for r in all_expanded_rels],
            "_expanded_entity_objects": [
                {
                    "id": str(n["neighbor"].id),
                    "name": n["neighbor"].name,
                    "canonical_name": n["neighbor"].canonical_name,
                    "type": n["neighbor"].type,
                    "description": n["neighbor"].description,
                    "mention_count": n["neighbor"].mention_count,
                    "article_ids": [str(aid) for aid in (n["neighbor"].article_ids or [])],
                }
                for n in scored_neighbors[:len(expanded_entities)]
            ] if "scored_neighbors" in dir() else [],
            "_expanded_rel_objects": all_expanded_rels,
            "current_phase": "expand_complete",
        }

    except Exception as e:
        logger.error("Subgraph expansion failed", error=str(e))
        return {
            "errors": [{"phase": "expand", "message": str(e)}],
            "current_phase": "expand_failed",
        }


@traceable(name="Build_Visualization", run_type="tool")
async def build_visualization_node(
    state: DeepGraphAnalystState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Build visualization data (nodes, edges, communities).

    Args:
        state: Current analyst state
        session: Database session

    Returns:
        Updated state fields with visualization data
    """
    logger.info("Building visualization data")

    seed_entity_objects = state.get("_seed_entity_objects", [])
    expanded_entities = state.get("expanded_entities", [])
    expanded_rel_objects = state.get("_expanded_rel_objects", [])

    try:
        # Build nodes
        nodes: list[GraphNode] = []

        # Seed entities
        for e in seed_entity_objects:
            nodes.append(GraphNode(
                id=e["id"],
                label=e["name"],
                type=e["type"],
                description=e.get("description", ""),
                mention_count=e.get("mention_count", 1),
                article_count=len(e.get("article_ids", [])),
                is_expanded=False,
            ))

        # Expanded entities
        expanded_ids = set()
        for exp in expanded_entities:
            expanded_ids.add(exp["entity_id"])

        # Get expanded entity details
        if expanded_ids:
            entity_uuids = [uuid.UUID(eid) for eid in expanded_ids]
            stmt = select(GraphEntity).where(GraphEntity.id.in_(entity_uuids))
            result = await session.execute(stmt)
            expanded_entity_objects = {
                str(e.id): e for e in result.scalars().all()
            }

            for exp in expanded_entities:
                e = expanded_entity_objects.get(exp["entity_id"])
                if e:
                    nodes.append(GraphNode(
                        id=str(e.id),
                        label=e.name,
                        type=e.type,
                        description=e.description or "",
                        mention_count=e.mention_count or 1,
                        article_count=len(e.article_ids or []),
                        is_expanded=True,
                    ))

        # Build edges
        edges: list[GraphEdge] = []

        # Seed relationships
        seed_rel_ids = [uuid.UUID(rid) for rid in state.get("seed_relationships", [])]
        if seed_rel_ids:
            stmt = select(GraphRelationship).where(GraphRelationship.id.in_(seed_rel_ids))
            result = await session.execute(stmt)
            for r in result.scalars().all():
                edges.append(GraphEdge(
                    id=str(r.id),
                    source=str(r.source_entity_id),
                    target=str(r.target_entity_id),
                    relation_type=r.relation_type,
                    description=r.description or "",
                    weight=r.weight or 1.0,
                    article_count=len(r.article_ids or []),
                    is_expanded=False,
                ))

        # Expanded relationships
        for r in expanded_rel_objects:
            # Avoid duplicates
            if str(r.id) in [e["id"] for e in edges]:
                continue

            is_expanded = (
                str(r.source_entity_id) in expanded_ids or
                str(r.target_entity_id) in expanded_ids
            )

            edges.append(GraphEdge(
                id=str(r.id),
                source=str(r.source_entity_id),
                target=str(r.target_entity_id),
                relation_type=r.relation_type,
                description=r.description or "",
                weight=r.weight or 1.0,
                article_count=len(r.article_ids or []),
                is_expanded=is_expanded,
            ))

        # Build community data
        all_entity_ids = [uuid.UUID(n["id"]) for n in nodes]
        communities_db = await graph_store.get_communities_by_entities(
            session=session,
            entity_ids=all_entity_ids,
        )

        communities: list[CommunityData] = []
        for c in communities_db:
            # Find hub entity name
            hub_entity_name = None
            if c.hub_entity_id:
                hub_entity = next(
                    (n for n in nodes if n["id"] == str(c.hub_entity_id)),
                    None
                )
                hub_entity_name = hub_entity["label"] if hub_entity else None

            communities.append(CommunityData(
                id=str(c.id),
                name=c.name,
                summary=c.summary or "",
                entity_count=len(c.entity_ids or []),
                hub_entity=hub_entity_name,
                article_ids=[str(aid) for aid in (c.article_ids or [])],
            ))

        logger.info(
            "Visualization data built",
            node_count=len(nodes),
            edge_count=len(edges),
            community_count=len(communities),
        )

        # Build visualization data dict for frontend
        visualization_data = {
            "nodes": nodes,
            "edges": edges,
            "communities": communities,
            "stats": {
                "total_entities": len(nodes),
                "seed_entities": len([n for n in nodes if not n["is_expanded"]]),
                "expanded_entities": len([n for n in nodes if n["is_expanded"]]),
                "total_relationships": len(edges),
                "total_communities": len(communities),
            },
        }

        return {
            "graph_nodes": nodes,
            "graph_edges": edges,
            "communities": communities,
            "visualization_data": visualization_data,
            "current_phase": "build_visualization_complete",
        }

    except Exception as e:
        logger.error("Failed to build visualization", error=str(e))
        return {
            "errors": [{"phase": "build_visualization", "message": str(e)}],
            "current_phase": "build_visualization_failed",
        }


@traceable(name="Generate_Report", run_type="chain")
async def generate_report_node(
    state: DeepGraphAnalystState,
) -> dict[str, Any]:
    """Generate comprehensive analysis report using LLM.

    Args:
        state: Current analyst state

    Returns:
        Updated state fields with final report
    """
    logger.info("Generating DeepGraph report")

    articles = state.get("_articles", [])
    nodes = state.get("graph_nodes", [])
    edges = state.get("graph_edges", [])
    communities = state.get("communities", [])
    expanded_entities = state.get("expanded_entities", [])

    try:
        # Initialize LLM
        llm_kwargs = {
            "model": settings.openai_chat_model,
            "api_key": settings.openai_api_key,
            "temperature": 0.6,
            "extra_body": {"enable_thinking": True},
        }
        if settings.openai_base_url:
            llm_kwargs["base_url"] = settings.openai_base_url
        llm = ChatOpenAI(**llm_kwargs)

        # Format data for prompt
        from datetime import datetime
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

        articles_str = format_articles_for_report(articles)
        graph_data = format_graph_for_report(
            entities=nodes,
            relationships=edges,
            communities=communities,
            expanded_entities=expanded_entities,
        )

        prompt = DEEP_GRAPH_REPORT_PROMPT.format(
            current_time=current_time,
            articles_info=articles_str,
            entity_count=graph_data["entity_count"],
            relationship_count=graph_data["relationship_count"],
            community_count=graph_data["community_count"],
            entities_info=graph_data["entities_info"],
            relationships_info=graph_data["relationships_info"],
            communities_info=graph_data["communities_info"],
            expanded_entities_info=graph_data["expanded_entities_info"],
        )

        # Generate report
        response = await llm.ainvoke(prompt)
        report = response.content if isinstance(response.content, str) else str(response.content)

        logger.info(
            "DeepGraph report generated",
            length=len(report),
        )

        return {
            "final_report": report,
            "current_phase": "report_complete",
        }

    except Exception as e:
        logger.error("Failed to generate report", error=str(e))
        return {
            "final_report": f"报告生成失败: {str(e)}",
            "errors": [{"phase": "generate_report", "message": str(e)}],
            "current_phase": "report_failed",
        }