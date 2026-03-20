"""GraphRAG Builder orchestration for background knowledge graph construction.

This workflow runs in the background after article storage:
1. Fetch articles
2. Extract entities
3. Extract relationships
4. Resolve/deduplicate entities
5. Detect communities
6. Store to database
"""

import uuid
from datetime import datetime, timezone

from langsmith import traceable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.graph_store import graph_store
from app.deep_graph.state import GraphBuilderState, create_initial_builder_state
from app.deep_graph.nodes_builder import (
    fetch_articles_node,
    extract_entities_node,
    extract_relationships_node,
    resolve_entities_node,
    detect_communities_node,
)
from app.deep_graph.tracing import DEEPGRAPH_TAGS, get_builder_metadata
from app.models.orm_models import GraphEntity

logger = get_logger(__name__)
settings = get_settings()


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

    This function implements a manual orchestration loop following
    the pattern used in deep_search/graph.py.

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
    state = create_initial_builder_state(article_ids=article_ids)

    try:
        # Create builder run record
        run = await graph_store.create_builder_run(
            session=session,
            article_ids=[uuid.UUID(aid) for aid in article_ids],
        )
        state["run_id"] = str(run.id)
        await session.commit()

        # Phase 1: Fetch articles
        state.update(await fetch_articles_node(state, session))

        if state.get("current_phase") == "fetch_failed":
            await session.rollback()
            await _complete_run(session, run.id, state, "failed")
            return state

        # Phase 2: Extract entities
        state.update(await extract_entities_node(state, session))

        # Phase 3: Extract relationships
        state.update(await extract_relationships_node(state, session))

        # Phase 4: Resolve entities
        state.update(await resolve_entities_node(state, session))

        # Phase 5: Store entities (uses ON CONFLICT for upsert)
        if state.get("resolved_entities"):
            entity_data = [
                {
                    "name": e["canonical_name"],
                    "canonical_name": e["canonical_name"],
                    "type": e["canonical_type"],
                    "description": e["description"],
                    "embedding": e.get("embedding"),
                    "article_ids": [uuid.UUID(aid) for aid in e["article_ids"]],
                    "aliases": e["source_entity_names"],
                    "mention_count": e["mention_count"],
                }
                for e in state["resolved_entities"]
            ]
            await graph_store.store_entities(session, entity_data)
            await session.commit()

        # Phase 6: Store relationships (uses ON CONFLICT for upsert)
        if state.get("extracted_relationships") and state.get("resolved_entities"):
            # Build name to ID mapping from stored entities
            canonical_names = [e["canonical_name"] for e in state["resolved_entities"]]
            stmt = select(GraphEntity).where(GraphEntity.canonical_name.in_(canonical_names))
            result = await session.execute(stmt)
            entities = result.scalars().all()
            name_to_id = {e.canonical_name: e.id for e in entities}

            # Also build case-insensitive mapping
            name_to_id_lower = {name.lower(): eid for name, eid in name_to_id.items()}

            relationship_data = []
            for article_id, rel in state["extracted_relationships"]:
                source_name = rel["source_entity"]
                target_name = rel["target_entity"]

                # Try exact match first, then case-insensitive
                source_id = name_to_id.get(source_name) or name_to_id_lower.get(source_name.lower())
                target_id = name_to_id.get(target_name) or name_to_id_lower.get(target_name.lower())

                if source_id and target_id:
                    relationship_data.append({
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                        "relation_type": rel["relation_type"],
                        "description": rel["description"],
                        "weight": 1.0,
                        "article_ids": [uuid.UUID(article_id)],
                        "evidence_texts": [rel["evidence"]] if rel.get("evidence") else [],
                    })

            if relationship_data:
                await graph_store.store_relationships(session, relationship_data)
                await session.commit()

        # Phase 7: Detect and store communities
        state.update(await detect_communities_node(state, session))

        if state.get("detected_communities"):
            # Get entity IDs
            canonical_names = [e["canonical_name"] for e in state.get("resolved_entities", [])]
            stmt = select(GraphEntity).where(GraphEntity.canonical_name.in_(canonical_names))
            result = await session.execute(stmt)
            entities = list(result.scalars().all())
            name_to_id = {e.canonical_name: e.id for e in entities}
            id_to_entity = {str(e.id): e for e in entities}

            comm_data = []
            for comm in state["detected_communities"]:
                entity_ids = []
                for eid_str in comm["entity_ids"]:
                    try:
                        entity_uuid = uuid.UUID(eid_str)
                        entity_ids.append(entity_uuid)
                    except ValueError:
                        # Try to resolve by entity name
                        if eid_str in name_to_id:
                            entity_ids.append(name_to_id[eid_str])

                hub_entity_id = None
                if comm.get("hub_entity_id"):
                    try:
                        hub_entity_id = uuid.UUID(comm["hub_entity_id"])
                    except ValueError:
                        pass

                comm_data.append({
                    "name": comm["name"],
                    "summary": comm["summary"],
                    "entity_ids": entity_ids,
                    "hub_entity_id": hub_entity_id,
                    "article_ids": [uuid.UUID(aid) for aid in comm.get("article_ids", [])],
                    "level": comm.get("level", 0),
                })

            if comm_data:
                await graph_store.store_communities(session, comm_data)
                await session.commit()

        # Mark as complete
        state["current_phase"] = "complete"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _complete_run(session, run.id, state, "completed")

        logger.info(
            "GraphRAG Builder completed",
            run_id=run.id,
            entities=state.get("entities_count", 0),
            relationships=state.get("relationships_count", 0),
            communities=state.get("communities_count", 0),
        )

    except Exception as e:
        logger.error(
            "GraphRAG Builder failed",
            run_id=state.get("run_id"),
            error=str(e),
        )
        # Rollback to clean state
        await session.rollback()
        state["errors"] = state.get("errors", []) + [{"phase": "orchestration", "message": str(e)}]
        state["current_phase"] = "failed"

        # Try to update the run record
        if state.get("run_id"):
            try:
                await _complete_run(session, uuid.UUID(state["run_id"]), state, "failed")
            except Exception:
                pass

    return state


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