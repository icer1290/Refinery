"""Node implementations for GraphRAG Builder workflow.

Nodes for background knowledge graph construction:
1. extract_entities_node: LLM extracts entities from articles
2. extract_relationships_node: LLM extracts relationships between entities
3. resolve_entities_node: Vector similarity to deduplicate entities
4. detect_communities_node: Leiden algorithm to find entity communities
5. store_graph_node: Persist entities, relationships, and communities
"""

import asyncio
import json
import re
import uuid
from typing import Any

from langchain_openai import ChatOpenAI
from langsmith import traceable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.graph_store import graph_store
from app.deep_graph.state import (
    GraphBuilderState,
    ExtractedEntity,
    ExtractedRelationship,
)
from app.deep_graph.prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT,
    RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT,
    RELATIONSHIP_EXTRACTION_USER_PROMPT,
    format_entity_types,
    format_entities_for_prompt,
)
from app.deep_graph.entity_resolver import entity_resolver
from app.deep_graph.community_detector import community_detector
from app.models.orm_models import NewsArticle, GraphEntity, GraphBuilderRun

logger = get_logger(__name__)
settings = get_settings()

# Concurrency control
MAX_CONCURRENT_EXTRACTIONS = 5


def _extract_json(text: str) -> str:
    """Extract JSON from text that might contain markdown code blocks."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


def _repair_partial_json(text: str) -> str | None:
    """Attempt lightweight JSON repair for truncated LLM responses."""
    extracted = _extract_json(text).strip()
    if not extracted:
        return None

    quote_count = extracted.count('"')
    if quote_count % 2 == 1:
        extracted += '"'

    open_braces = extracted.count("{")
    close_braces = extracted.count("}")
    if open_braces > close_braces:
        extracted += "}" * (open_braces - close_braces)

    open_brackets = extracted.count("[")
    close_brackets = extracted.count("]")
    if open_brackets > close_brackets:
        extracted += "]" * (open_brackets - close_brackets)

    extracted = re.sub(r",\s*([}\]])", r"\1", extracted)
    return extracted


@traceable(name="Fetch_Articles", run_type="tool")
async def fetch_articles_node(
    state: GraphBuilderState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Fetch articles from database by IDs.

    Args:
        state: Current builder state
        session: Database session

    Returns:
        Updated state fields
    """
    logger.info(
        "Fetching articles for graph building",
        article_count=len(state["article_ids"]),
    )

    try:
        article_uuids = [uuid.UUID(aid) for aid in state["article_ids"]]
        stmt = select(NewsArticle).where(NewsArticle.id.in_(article_uuids))
        result = await session.execute(stmt)
        articles = list(result.scalars().all())

        if not articles:
            return {
                "errors": [{"phase": "fetch", "message": "No articles found"}],
                "current_phase": "fetch_failed",
            }

        # Store articles in state
        article_dicts = []
        for article in articles:
            article_dicts.append({
                "id": str(article.id),
                "title": article.chinese_title or article.original_title,
                "original_title": article.original_title,
                "content": article.full_content or article.chinese_summary or article.original_description,
                "summary": article.chinese_summary or article.original_description,
                "source": article.source_name,
                "published_at": str(article.published_at) if article.published_at else "Unknown",
            })

        logger.info(
            "Articles fetched",
            count=len(article_dicts),
        )

        # Store in metadata for use by other nodes
        return {
            "_articles": article_dicts,
            "current_phase": "fetch_complete",
        }

    except Exception as e:
        logger.error("Failed to fetch articles", error=str(e))
        return {
            "errors": [{"phase": "fetch", "message": str(e)}],
            "current_phase": "fetch_failed",
        }


@traceable(name="Extract_Entities_From_Article", run_type="llm")
async def extract_entities_from_article(
    article: dict,
    llm: ChatOpenAI,
) -> list[ExtractedEntity]:
    """Extract entities from a single article using LLM.

    Args:
        article: Article dict with title, content, etc.
        llm: LangChain ChatOpenAI instance

    Returns:
        List of extracted entities
    """
    try:
        system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
            entity_types_desc=format_entity_types()
        )

        user_prompt = ENTITY_EXTRACTION_USER_PROMPT.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            published_at=article.get("published_at", "Unknown"),
            content=(article.get("content") or article.get("summary", ""))[:4000],
        )

        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)

        # Parse JSON response
        json_str = _extract_json(response_text)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try repair
            repaired = _repair_partial_json(response_text)
            if repaired:
                data = json.loads(repaired)
            else:
                logger.warning(
                    "Failed to parse entity extraction response",
                    article_id=article.get("id"),
                    response_preview=response_text[:200],
                )
                return []

        entities = data.get("entities", [])
        return [
            ExtractedEntity(
                name=e.get("name", ""),
                type=e.get("type", "CONCEPT"),
                description=e.get("description", ""),
                mentions=e.get("mentions", []),
                confidence=e.get("confidence", 0.5),
            )
            for e in entities
            if e.get("name")
        ]

    except Exception as e:
        logger.error(
            "Entity extraction failed",
            article_id=article.get("id"),
            error=str(e),
        )
        return []


@traceable(name="Extract_Entities", run_type="chain")
async def extract_entities_node(
    state: GraphBuilderState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Extract entities from all articles using LLM.

    Args:
        state: Current builder state
        session: Database session

    Returns:
        Updated state fields with extracted entities
    """
    logger.info("Starting entity extraction")

    articles = state.get("_articles", [])
    if not articles:
        return {
            "errors": [{"phase": "extract_entities", "message": "No articles to process"}],
            "current_phase": "extract_entities_failed",
        }

    # Initialize LLM
    llm_kwargs = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key,
        "temperature": 0.3,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    llm = ChatOpenAI(**llm_kwargs)

    # Extract entities from each article with concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

    async def extract_with_semaphore(article: dict) -> list[tuple[str, ExtractedEntity]]:
        async with semaphore:
            entities = await extract_entities_from_article(article, llm)
            return [(article["id"], e) for e in entities]

    # Run extractions in parallel
    tasks = [extract_with_semaphore(article) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    all_entities: list[tuple[str, ExtractedEntity]] = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append({
                "phase": "extract_entities",
                "article_id": articles[i].get("id"),
                "message": str(result),
            })
        else:
            all_entities.extend(result)

    entity_count = len(set(e["name"] for _, e in all_entities))

    logger.info(
        "Entity extraction complete",
        total_mentions=len(all_entities),
        unique_entities=entity_count,
        errors=len(errors),
    )

    return {
        "extracted_entities": all_entities,
        "entities_count": entity_count,
        "current_phase": "extract_entities_complete",
        "errors": errors,
    }


@traceable(name="Extract_Relationships_From_Article", run_type="llm")
async def extract_relationships_from_article(
    article: dict,
    entities: list[ExtractedEntity],
    llm: ChatOpenAI,
) -> list[ExtractedRelationship]:
    """Extract relationships from a single article using LLM.

    Args:
        article: Article dict
        entities: Already extracted entities from this article
        llm: LangChain ChatOpenAI instance

    Returns:
        List of extracted relationships
    """
    if not entities:
        return []

    try:
        system_prompt = RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT
        user_prompt = RELATIONSHIP_EXTRACTION_USER_PROMPT.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            content=(article.get("content") or article.get("summary", ""))[:4000],
            entities=format_entities_for_prompt(entities),
        )

        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)

        # Parse JSON response
        json_str = _extract_json(response_text)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = _repair_partial_json(response_text)
            if repaired:
                data = json.loads(repaired)
            else:
                logger.warning(
                    "Failed to parse relationship extraction response",
                    article_id=article.get("id"),
                )
                return []

        relationships = data.get("relationships", [])
        return [
            ExtractedRelationship(
                source_entity=r.get("source_entity", ""),
                target_entity=r.get("target_entity", ""),
                relation_type=r.get("relation_type", "related_to"),
                description=r.get("description", ""),
                evidence=r.get("evidence", ""),
            )
            for r in relationships
            if r.get("source_entity") and r.get("target_entity")
        ]

    except Exception as e:
        logger.error(
            "Relationship extraction failed",
            article_id=article.get("id"),
            error=str(e),
        )
        return []


@traceable(name="Extract_Relationships", run_type="chain")
async def extract_relationships_node(
    state: GraphBuilderState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Extract relationships from all articles using LLM.

    Args:
        state: Current builder state
        session: Database session

    Returns:
        Updated state fields with extracted relationships
    """
    logger.info("Starting relationship extraction")

    articles = state.get("_articles", [])
    extracted_entities = state.get("extracted_entities", [])

    if not articles:
        return {
            "current_phase": "extract_relationships_complete",
            "relationships_count": 0,
        }

    # Group entities by article
    entities_by_article: dict[str, list[ExtractedEntity]] = {}
    for article_id, entity in extracted_entities:
        if article_id not in entities_by_article:
            entities_by_article[article_id] = []
        entities_by_article[article_id].append(entity)

    # Initialize LLM
    llm_kwargs = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key,
        "temperature": 0.3,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    llm = ChatOpenAI(**llm_kwargs)

    # Extract relationships from each article
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

    async def extract_with_semaphore(article: dict) -> list[tuple[str, ExtractedRelationship]]:
        async with semaphore:
            article_entities = entities_by_article.get(article["id"], [])
            relationships = await extract_relationships_from_article(
                article, article_entities, llm
            )
            return [(article["id"], r) for r in relationships]

    tasks = [extract_with_semaphore(article) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    all_relationships: list[tuple[str, ExtractedRelationship]] = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append({
                "phase": "extract_relationships",
                "article_id": articles[i].get("id"),
                "message": str(result),
            })
        else:
            all_relationships.extend(result)

    relationship_count = len(all_relationships)

    logger.info(
        "Relationship extraction complete",
        total_relationships=relationship_count,
        errors=len(errors),
    )

    return {
        "extracted_relationships": all_relationships,
        "relationships_count": relationship_count,
        "current_phase": "extract_relationships_complete",
        "errors": errors,
    }


@traceable(name="Resolve_Entities", run_type="tool")
async def resolve_entities_node(
    state: GraphBuilderState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Resolve and deduplicate entities using vector similarity.

    Args:
        state: Current builder state
        session: Database session

    Returns:
        Updated state fields with resolved entities
    """
    logger.info("Starting entity resolution")

    extracted_entities = state.get("extracted_entities", [])
    if not extracted_entities:
        return {
            "resolved_entities": [],
            "current_phase": "resolve_entities_complete",
        }

    try:
        resolved = await entity_resolver.resolve_entities(
            session=session,
            extracted_entities=extracted_entities,
        )

        logger.info(
            "Entity resolution complete",
            input_count=len(extracted_entities),
            resolved_count=len(resolved),
        )

        return {
            "resolved_entities": resolved,
            "current_phase": "resolve_entities_complete",
        }

    except Exception as e:
        logger.error("Entity resolution failed", error=str(e))
        return {
            "errors": [{"phase": "resolve_entities", "message": str(e)}],
            "current_phase": "resolve_entities_failed",
        }


@traceable(name="Detect_Communities", run_type="tool")
async def detect_communities_node(
    state: GraphBuilderState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Detect communities using Leiden algorithm.

    Args:
        state: Current builder state
        session: Database session

    Returns:
        Updated state fields with detected communities
    """
    logger.info("Starting community detection")

    # Get stored entities and relationships
    resolved_entities = state.get("resolved_entities", [])
    extracted_relationships = state.get("extracted_relationships", [])

    if not resolved_entities:
        return {
            "detected_communities": [],
            "communities_count": 0,
            "current_phase": "detect_communities_complete",
        }

    try:
        # We need entity IDs from the database for community detection
        # Get entities by canonical names
        canonical_names = [e["canonical_name"] for e in resolved_entities]
        stmt = select(GraphEntity).where(GraphEntity.canonical_name.in_(canonical_names))
        result = await session.execute(stmt)
        db_entities = list(result.scalars().all())

        # Get relationships for these entities
        entity_ids = [e.id for e in db_entities]
        db_relationships = await graph_store.get_relationships_by_entities(
            session, entity_ids
        )

        # Run community detection
        communities = await community_detector.detect_communities(
            session=session,
            entities=db_entities,
            relationships=db_relationships,
        )

        logger.info(
            "Community detection complete",
            community_count=len(communities),
        )

        return {
            "detected_communities": communities,
            "communities_count": len(communities),
            "current_phase": "detect_communities_complete",
        }

    except Exception as e:
        logger.error("Community detection failed", error=str(e))
        return {
            "errors": [{"phase": "detect_communities", "message": str(e)}],
            "current_phase": "detect_communities_failed",
        }


async def store_graph_node(
    state: GraphBuilderState,
    session: AsyncSession,
) -> dict[str, Any]:
    """Store resolved entities, relationships, and communities to database.

    Args:
        state: Current builder state
        session: Database session

    Returns:
        Updated state fields
    """
    logger.info("Starting graph storage")

    resolved_entities = state.get("resolved_entities", [])
    extracted_relationships = state.get("extracted_relationships", [])
    detected_communities = state.get("detected_communities", [])

    try:
        # Build entity name to ID mapping
        entity_name_to_id: dict[str, uuid.UUID] = {}

        # Store entities
        if resolved_entities:
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
                for e in resolved_entities
            ]
            stored_entities = await graph_store.store_entities(session, entity_data)
            for e in stored_entities:
                entity_name_to_id[e.canonical_name] = e.id

        # Store relationships
        if extracted_relationships and entity_name_to_id:
            # Map extracted relationships to entity IDs
            relationship_data = []
            for article_id, rel in extracted_relationships:
                source_name = rel["source_entity"]
                target_name = rel["target_entity"]

                # Try to find entity IDs (use resolution map)
                source_id = entity_name_to_id.get(source_name)
                target_id = entity_name_to_id.get(target_name)

                # If not found, try case-insensitive lookup
                if not source_id:
                    for name, eid in entity_name_to_id.items():
                        if name.lower() == source_name.lower():
                            source_id = eid
                            break
                if not target_id:
                    for name, eid in entity_name_to_id.items():
                        if name.lower() == target_name.lower():
                            target_id = eid
                            break

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

            await graph_store.store_relationships(session, relationship_data)

        # Store communities
        if detected_communities:
            # Update entity IDs in communities
            comm_data = []
            for comm in detected_communities:
                # Resolve entity IDs
                resolved_entity_ids = []
                for eid_str in comm["entity_ids"]:
                    try:
                        resolved_entity_ids.append(uuid.UUID(eid_str))
                    except ValueError:
                        # Try to find by canonical name
                        for name, eid in entity_name_to_id.items():
                            if str(eid) == eid_str:
                                resolved_entity_ids.append(eid)
                                break

                comm_data.append({
                    "name": comm["name"],
                    "summary": comm["summary"],
                    "entity_ids": resolved_entity_ids,
                    "hub_entity_id": uuid.UUID(comm["hub_entity_id"]) if comm.get("hub_entity_id") else None,
                    "article_ids": [uuid.UUID(aid) for aid in comm.get("article_ids", [])],
                    "level": comm.get("level", 0),
                })

            await graph_store.store_communities(session, comm_data)

        logger.info(
            "Graph storage complete",
            entities=len(resolved_entities),
            relationships=len(extracted_relationships),
            communities=len(detected_communities),
        )

        return {
            "current_phase": "storage_complete",
        }

    except Exception as e:
        logger.error("Graph storage failed", error=str(e))
        return {
            "errors": [{"phase": "store_graph", "message": str(e)}],
            "current_phase": "storage_failed",
        }