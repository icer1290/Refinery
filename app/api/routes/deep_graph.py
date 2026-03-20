"""DeepGraph API endpoints.

Provides endpoints for:
- /deep-graph/analyze: Run DeepGraph analysis on selected articles
- /deep-graph/entities: Search entities
- /deep-graph/entities/{id}: Get entity details
- /deep-graph/communities: List communities
- /deep-graph/builder/run: Trigger background graph builder
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core import get_logger
from app.deep_graph.graph_analyst import run_deep_graph_analyst
from app.deep_graph.graph_builder import run_graph_builder
from app.deep_graph.graph_store import graph_store
from app.models.orm_models import DeepGraphAnalysis
from app.models.schemas import (
    DeepGraphRequest,
    DeepGraphResponse,
    GraphNodeResponse,
    GraphEdgeResponse,
    CommunityResponse,
    VisualizationData,
    VisualizationStats,
    EntitySearchRequest,
    EntityResponse,
    EntityListResponse,
    GraphBuilderRunRequest,
    GraphBuilderRunResponse,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post("/deep-graph/analyze", response_model=DeepGraphResponse)
async def analyze_deep_graph(
    request: DeepGraphRequest,
    db: AsyncSession = Depends(get_db),
) -> DeepGraphResponse:
    """Run DeepGraph analysis on selected articles.

    This endpoint:
    1. Fetches the knowledge graph subgraph for selected articles
    2. Expands the subgraph with relevant neighboring entities
    3. Builds visualization data
    4. Generates a comprehensive analysis report

    Args:
        request: DeepGraph request with article_ids and expansion parameters
        db: Database session

    Returns:
        DeepGraph response with report and visualization data
    """
    logger.info(
        "DeepGraph analysis requested",
        article_count=len(request.article_ids),
        max_hops=request.max_hops,
        expansion_limit=request.expansion_limit,
    )

    try:
        # Validate UUIDs
        for aid in request.article_ids:
            try:
                uuid.UUID(aid)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid article_id format: {aid}",
                )

        # Run DeepGraph Analyst
        state = await run_deep_graph_analyst(
            session=db,
            article_ids=request.article_ids,
            max_hops=request.max_hops,
            expansion_limit=request.expansion_limit,
        )

        # Build response
        nodes = [
            GraphNodeResponse(
                id=n["id"],
                label=n["label"],
                type=n["type"],
                description=n.get("description"),
                mention_count=n.get("mention_count", 1),
                article_count=n.get("article_count", 1),
                is_expanded=n.get("is_expanded", False),
            )
            for n in state.get("graph_nodes", [])
        ]

        edges = [
            GraphEdgeResponse(
                id=e["id"],
                source=e["source"],
                target=e["target"],
                relation_type=e["relation_type"],
                description=e.get("description"),
                weight=e.get("weight", 1.0),
                article_count=e.get("article_count", 1),
                is_expanded=e.get("is_expanded", False),
            )
            for e in state.get("graph_edges", [])
        ]

        communities = [
            CommunityResponse(
                id=c["id"],
                name=c["name"],
                summary=c.get("summary"),
                entity_count=c.get("entity_count", 0),
                hub_entity=c.get("hub_entity"),
                article_ids=c.get("article_ids", []),
            )
            for c in state.get("communities", [])
        ]

        viz_data = state.get("visualization_data", {})
        stats = viz_data.get("stats", {})

        visualization_data = VisualizationData(
            nodes=nodes,
            edges=edges,
            communities=communities,
            stats=VisualizationStats(
                total_entities=stats.get("total_entities", len(nodes)),
                seed_entities=stats.get("seed_entities", 0),
                expanded_entities=stats.get("expanded_entities", 0),
                total_relationships=stats.get("total_relationships", len(edges)),
                total_communities=stats.get("total_communities", len(communities)),
            ),
        )

        response = DeepGraphResponse(
            article_ids=request.article_ids,
            graph_nodes=nodes,
            graph_edges=edges,
            communities=communities,
            report=state.get("final_report", ""),
            visualization_data=visualization_data,
            errors=state.get("errors", []),
        )

        # Store analysis if user_id is provided
        if request.user_id is not None:
            analysis = DeepGraphAnalysis(
                user_id=request.user_id,
                article_ids=[uuid.UUID(aid) for aid in request.article_ids],
                report=response.report,
                visualization_data=response.model_dump(),
                max_hops=request.max_hops,
                expansion_limit=request.expansion_limit,
            )
            db.add(analysis)
            await db.commit()
            logger.info(
                "Stored DeepGraph analysis",
                analysis_id=analysis.id,
                user_id=request.user_id,
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("DeepGraph analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"DeepGraph analysis failed: {str(e)}",
        )


@router.get("/deep-graph/entities", response_model=EntityListResponse)
async def search_entities(
    query: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> EntityListResponse:
    """Search entities by name.

    Args:
        query: Search query string
        limit: Maximum number of results
        db: Database session

    Returns:
        List of matching entities
    """
    logger.info("Entity search requested", query=query, limit=limit)

    try:
        entities = await graph_store.search_entities_by_name(
            session=db,
            query=query,
            limit=limit,
        )

        return EntityListResponse(
            entities=[
                EntityResponse(
                    id=e.id,
                    name=e.name,
                    canonical_name=e.canonical_name,
                    type=e.type,
                    description=e.description,
                    mention_count=e.mention_count or 1,
                    aliases=e.aliases or [],
                    article_ids=e.article_ids or [],
                    created_at=e.created_at,
                )
                for e in entities
            ],
            total=len(entities),
        )

    except Exception as e:
        logger.error("Entity search failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Entity search failed: {str(e)}",
        )


@router.get("/deep-graph/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    db: AsyncSession = Depends(get_db),
) -> EntityResponse:
    """Get entity details by ID.

    Args:
        entity_id: Entity UUID
        db: Database session

    Returns:
        Entity details
    """
    logger.info("Entity detail requested", entity_id=entity_id)

    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid entity_id format. Must be a valid UUID.",
        )

    try:
        entity = await graph_store.get_entity_by_id(
            session=db,
            entity_id=entity_uuid,
        )

        if not entity:
            raise HTTPException(
                status_code=404,
                detail="Entity not found",
            )

        return EntityResponse(
            id=entity.id,
            name=entity.name,
            canonical_name=entity.canonical_name,
            type=entity.type,
            description=entity.description,
            mention_count=entity.mention_count or 1,
            aliases=entity.aliases or [],
            article_ids=entity.article_ids or [],
            created_at=entity.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get entity: {str(e)}",
        )


@router.get("/deep-graph/communities")
async def list_communities(
    limit: int = 100,
    level: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[CommunityResponse]:
    """List all communities.

    Args:
        limit: Maximum number of results
        level: Optional level filter
        db: Database session

    Returns:
        List of communities
    """
    logger.info("Communities list requested", limit=limit, level=level)

    try:
        communities = await graph_store.get_all_communities(
            session=db,
            limit=limit,
            level=level,
        )

        return [
            CommunityResponse(
                id=str(c.id),
                name=c.name,
                summary=c.summary,
                entity_count=len(c.entity_ids or []),
                hub_entity=str(c.hub_entity_id) if c.hub_entity_id else None,
                article_ids=[str(aid) for aid in (c.article_ids or [])],
            )
            for c in communities
        ]

    except Exception as e:
        logger.error("Failed to list communities", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list communities: {str(e)}",
        )


@router.post("/deep-graph/builder/run", response_model=GraphBuilderRunResponse)
async def trigger_graph_builder(
    request: GraphBuilderRunRequest,
    db: AsyncSession = Depends(get_db),
) -> GraphBuilderRunResponse:
    """Trigger background graph builder.

    This endpoint starts the GraphRAG builder to:
    1. Extract entities from specified articles
    2. Extract relationships between entities
    3. Resolve/deduplicate entities
    4. Detect communities
    5. Store to database

    Args:
        request: Builder request with article_ids
        db: Database session

    Returns:
        Builder run record
    """
    logger.info(
        "Graph builder triggered",
        article_count=len(request.article_ids),
    )

    try:
        # Validate UUIDs
        for aid in request.article_ids:
            try:
                uuid.UUID(aid)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid article_id format: {aid}",
                )

        # Run builder (synchronously for now)
        state = await run_graph_builder(
            session=db,
            article_ids=request.article_ids,
        )

        # Get the run record
        from app.models.orm_models import GraphBuilderRun
        from sqlalchemy import select

        stmt = select(GraphBuilderRun).where(
            GraphBuilderRun.id == uuid.UUID(state["run_id"])
        )
        result = await db.execute(stmt)
        run = result.scalar_one_or_none()

        if not run:
            raise HTTPException(
                status_code=500,
                detail="Builder run not found after execution",
            )

        return GraphBuilderRunResponse(
            id=run.id,
            started_at=run.started_at,
            completed_at=run.completed_at,
            status=run.status,
            article_ids=run.article_ids or [],
            entities_extracted=run.entities_extracted,
            relationships_extracted=run.relationships_extracted,
            communities_detected=run.communities_detected,
            errors=run.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Graph builder failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Graph builder failed: {str(e)}",
        )


@router.get("/deep-graph/stats")
async def get_graph_stats(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get knowledge graph statistics.

    Args:
        db: Database session

    Returns:
        Statistics about the knowledge graph
    """
    try:
        entity_count = await graph_store.get_entity_count(db)
        relationship_count = await graph_store.get_relationship_count(db)
        community_count = await graph_store.get_community_count(db)

        return {
            "total_entities": entity_count,
            "total_relationships": relationship_count,
            "total_communities": community_count,
        }

    except Exception as e:
        logger.error("Failed to get graph stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph stats: {str(e)}",
        )