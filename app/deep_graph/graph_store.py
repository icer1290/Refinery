"""Graph store operations for GraphRAG.

Provides database CRUD operations for:
- graph_entities: Entity nodes
- graph_relationships: Directed edges between entities
- graph_communities: Clusters of related entities
- graph_builder_runs: Builder execution history
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, text, func, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import DatabaseError
from app.models.orm_models import (
    GraphEntity,
    GraphRelationship,
    GraphCommunity,
    GraphBuilderRun,
)

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class EntitySearchResult:
    """Result from entity search."""

    entity_id: uuid.UUID
    name: str
    canonical_name: str
    type: str
    description: str
    similarity: float


class GraphStore:
    """PostgreSQL operations for graph data.

    Supports:
    - Entity storage with upsert and embedding search
    - Relationship storage with merge
    - Community storage and retrieval
    - Graph expansion (1-hop neighbors)
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    # === Entity Operations ===

    async def store_entities(
        self,
        session: AsyncSession,
        entities: List[dict],
    ) -> List[GraphEntity]:
        """Store entities with upsert (merge on canonical_name).

        Uses INSERT ... ON CONFLICT for atomic upsert.

        Args:
            session: Database session
            entities: List of entity dicts with keys:
                - canonical_name: str
                - type: str
                - description: str
                - embedding: list[float] | None
                - article_ids: list[uuid.UUID]
                - aliases: list[str]
                - mention_count: int

        Returns:
            List of stored GraphEntity objects
        """
        if not entities:
            return []

        try:
            stored = []

            # Group entities by canonical_name to merge duplicates in batch
            grouped: dict[str, dict] = {}
            for entity_data in entities:
                key = entity_data["canonical_name"]
                if key not in grouped:
                    grouped[key] = {
                        "name": entity_data.get("name", key),
                        "canonical_name": key,
                        "type": entity_data["type"],
                        "description": entity_data.get("description", ""),
                        "embedding": entity_data.get("embedding"),
                        "article_ids": list(entity_data.get("article_ids", [])),
                        "aliases": list(entity_data.get("aliases", [])),
                        "mention_count": entity_data.get("mention_count", 1),
                    }
                else:
                    # Merge duplicates in batch
                    existing = grouped[key]
                    existing["article_ids"] = list(set(existing["article_ids"]) | set(entity_data.get("article_ids", [])))
                    existing["aliases"] = list(set(existing["aliases"]) | set(entity_data.get("aliases", [])))
                    existing["mention_count"] += entity_data.get("mention_count", 1)
                    if entity_data.get("description"):
                        existing["description"] = entity_data["description"]
                    if entity_data.get("embedding"):
                        existing["embedding"] = entity_data["embedding"]

            for entity_data in grouped.values():
                canonical_name = entity_data["canonical_name"]

                # Use separate queries for with/without embedding to avoid type ambiguity
                if entity_data.get("embedding"):
                    embedding_str = "[" + ",".join(map(str, entity_data["embedding"])) + "]"

                    query = text("""
                        INSERT INTO graph_entities
                            (id, name, canonical_name, type, description, embedding, article_ids, mention_count, aliases)
                        VALUES
                            (gen_random_uuid(), :name, :canonical_name, :type, :description,
                             CAST(:embedding AS vector),
                             :article_ids, :mention_count, :aliases)
                        ON CONFLICT (canonical_name) DO UPDATE SET
                            article_ids = (SELECT ARRAY(SELECT DISTINCT unnest(array_cat(graph_entities.article_ids, EXCLUDED.article_ids)))),
                            aliases = (SELECT ARRAY(SELECT DISTINCT unnest(array_cat(graph_entities.aliases, EXCLUDED.aliases)))),
                            mention_count = graph_entities.mention_count + EXCLUDED.mention_count,
                            description = COALESCE(EXCLUDED.description, graph_entities.description),
                            embedding = COALESCE(EXCLUDED.embedding, graph_entities.embedding),
                            updated_at = NOW()
                        RETURNING id, name, canonical_name, type, description, embedding, article_ids, mention_count, aliases, created_at, updated_at
                    """)

                    result = await session.execute(
                        query,
                        {
                            "name": entity_data["name"],
                            "canonical_name": canonical_name,
                            "type": entity_data["type"],
                            "description": entity_data["description"],
                            "embedding": embedding_str,
                            "article_ids": [str(aid) for aid in entity_data["article_ids"]],
                            "mention_count": entity_data["mention_count"],
                            "aliases": entity_data["aliases"],
                        },
                    )
                else:
                    query = text("""
                        INSERT INTO graph_entities
                            (id, name, canonical_name, type, description, embedding, article_ids, mention_count, aliases)
                        VALUES
                            (gen_random_uuid(), :name, :canonical_name, :type, :description, NULL,
                             :article_ids, :mention_count, :aliases)
                        ON CONFLICT (canonical_name) DO UPDATE SET
                            article_ids = (SELECT ARRAY(SELECT DISTINCT unnest(array_cat(graph_entities.article_ids, EXCLUDED.article_ids)))),
                            aliases = (SELECT ARRAY(SELECT DISTINCT unnest(array_cat(graph_entities.aliases, EXCLUDED.aliases)))),
                            mention_count = graph_entities.mention_count + EXCLUDED.mention_count,
                            description = COALESCE(EXCLUDED.description, graph_entities.description),
                            updated_at = NOW()
                        RETURNING id, name, canonical_name, type, description, embedding, article_ids, mention_count, aliases, created_at, updated_at
                    """)

                    result = await session.execute(
                        query,
                        {
                            "name": entity_data["name"],
                            "canonical_name": canonical_name,
                            "type": entity_data["type"],
                            "description": entity_data["description"],
                            "article_ids": [str(aid) for aid in entity_data["article_ids"]],
                            "mention_count": entity_data["mention_count"],
                            "aliases": entity_data["aliases"],
                        },
                    )

                row = result.fetchone()
                if row:
                    stored.append(GraphEntity(
                        id=row.id,
                        name=row.name,
                        canonical_name=row.canonical_name,
                        type=row.type,
                        description=row.description,
                        embedding=row.embedding,
                        article_ids=row.article_ids,
                        mention_count=row.mention_count,
                        aliases=row.aliases,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    ))

            await session.flush()

            logger.info(
                "Stored entities",
                count=len(stored),
            )

            return stored

        except Exception as e:
            raise DatabaseError(
                f"Failed to store entities: {str(e)}",
                {"error": str(e)},
            )

    async def get_entities_by_articles(
        self,
        session: AsyncSession,
        article_ids: List[uuid.UUID],
    ) -> List[GraphEntity]:
        """Get all entities associated with given articles.

        Args:
            session: Database session
            article_ids: List of article IDs

        Returns:
            List of GraphEntity objects
        """
        try:
            if not article_ids:
                return []

            # Use ANY query for array column
            query = text("""
                SELECT * FROM graph_entities
                WHERE article_ids && :article_ids
            """)

            result = await session.execute(
                query,
                {"article_ids": [str(aid) for aid in article_ids]},
            )

            entities = []
            for row in result:
                # Convert embedding from string/list of strings to list of floats
                embedding = None
                if row.embedding:
                    if isinstance(row.embedding, str):
                        # Parse string representation like "[0.1,0.2,0.3]"
                        embedding = [float(x) for x in row.embedding.strip('[]').split(',') if x.strip()]
                    elif isinstance(row.embedding, list):
                        embedding = [float(x) if isinstance(x, str) else x for x in row.embedding]

                entity = GraphEntity(
                    id=row.id,
                    name=row.name,
                    canonical_name=row.canonical_name,
                    type=row.type,
                    description=row.description,
                    embedding=embedding,
                    article_ids=row.article_ids,
                    mention_count=int(row.mention_count) if row.mention_count else 1,
                    aliases=row.aliases,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                entities.append(entity)

            logger.info(
                "Retrieved entities by articles",
                article_count=len(article_ids),
                entity_count=len(entities),
            )

            return entities

        except Exception as e:
            raise DatabaseError(
                f"Failed to get entities by articles: {str(e)}",
                {"error": str(e)},
            )

    async def search_similar_entities(
        self,
        session: AsyncSession,
        embedding: List[float],
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        exclude_ids: Optional[List[uuid.UUID]] = None,
    ) -> List[EntitySearchResult]:
        """Search entities by embedding similarity.

        Args:
            session: Database session
            embedding: Query embedding vector
            limit: Maximum results
            similarity_threshold: Minimum similarity (default: self.similarity_threshold)
            exclude_ids: Entity IDs to exclude

        Returns:
            List of EntitySearchResult objects
        """
        try:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            threshold = similarity_threshold or self.similarity_threshold

            query = text("""
                SELECT
                    id, name, canonical_name, type, description,
                    1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                FROM graph_entities
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """)

            params = {
                "embedding": embedding_str,
                "threshold": threshold,
                "limit": limit,
            }

            if exclude_ids:
                exclude_str = ",".join([f"'{str(eid)}'" for eid in exclude_ids])
                query = text(
                    query.text.replace(
                        "WHERE",
                        f"WHERE id NOT IN ({exclude_str}) AND"
                    )
                )

            result = await session.execute(query, params)

            results = []
            for row in result:
                results.append(EntitySearchResult(
                    entity_id=row.id,
                    name=row.name,
                    canonical_name=row.canonical_name,
                    type=row.type,
                    description=row.description,
                    similarity=row.similarity,
                ))

            return results

        except Exception as e:
            raise DatabaseError(
                f"Failed to search similar entities: {str(e)}",
                {"error": str(e)},
            )

    async def get_entity_by_id(
        self,
        session: AsyncSession,
        entity_id: uuid.UUID,
    ) -> Optional[GraphEntity]:
        """Get entity by ID.

        Args:
            session: Database session
            entity_id: Entity UUID

        Returns:
            GraphEntity or None
        """
        try:
            stmt = select(GraphEntity).where(GraphEntity.id == entity_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get entity: {str(e)}",
                {"entity_id": str(entity_id), "error": str(e)},
            )

    # === Relationship Operations ===

    async def store_relationships(
        self,
        session: AsyncSession,
        relationships: List[dict],
    ) -> List[GraphRelationship]:
        """Store relationships with merge on (source, target, type).

        Uses INSERT ... ON CONFLICT for atomic upsert.

        Args:
            session: Database session
            relationships: List of relationship dicts with keys:
                - source_entity_id: uuid.UUID
                - target_entity_id: uuid.UUID
                - relation_type: str
                - description: str
                - weight: float
                - article_ids: list[uuid.UUID]
                - evidence_texts: list[str]

        Returns:
            List of stored GraphRelationship objects
        """
        if not relationships:
            return []

        try:
            stored = []

            # Group relationships by (source, target, type) to merge duplicates in batch
            grouped: dict[tuple, dict] = {}
            for rel_data in relationships:
                key = (str(rel_data["source_entity_id"]), str(rel_data["target_entity_id"]), rel_data["relation_type"])
                if key not in grouped:
                    grouped[key] = {
                        "source_entity_id": rel_data["source_entity_id"],
                        "target_entity_id": rel_data["target_entity_id"],
                        "relation_type": rel_data["relation_type"],
                        "description": rel_data.get("description", ""),
                        "weight": rel_data.get("weight", 1.0),
                        "article_ids": list(rel_data.get("article_ids", [])),
                        "evidence_texts": list(rel_data.get("evidence_texts", [])),
                    }
                else:
                    # Merge duplicates in batch
                    existing = grouped[key]
                    existing["article_ids"] = list(set(existing["article_ids"]) | set(rel_data.get("article_ids", [])))
                    existing["evidence_texts"] = list(set(existing["evidence_texts"]) | set(rel_data.get("evidence_texts", [])))
                    existing["weight"] += rel_data.get("weight", 1.0)
                    if rel_data.get("description"):
                        existing["description"] = rel_data["description"]

            # Use raw SQL with ON CONFLICT for atomic upsert
            for rel_data in grouped.values():
                source_id = str(rel_data["source_entity_id"])
                target_id = str(rel_data["target_entity_id"])
                rel_type = rel_data["relation_type"]

                # Use ON CONFLICT to handle duplicates atomically
                query = text("""
                    INSERT INTO graph_relationships
                        (id, source_entity_id, target_entity_id, relation_type, description, weight, article_ids, evidence_texts)
                    VALUES
                        (gen_random_uuid(), :source_id, :target_id, :rel_type, :description, :weight, :article_ids, :evidence_texts)
                    ON CONFLICT (source_entity_id, target_entity_id, relation_type) DO UPDATE SET
                        article_ids = (SELECT ARRAY(SELECT DISTINCT unnest(array_cat(graph_relationships.article_ids, EXCLUDED.article_ids)))),
                        evidence_texts = (SELECT ARRAY(SELECT DISTINCT unnest(array_cat(graph_relationships.evidence_texts, EXCLUDED.evidence_texts)))),
                        weight = graph_relationships.weight + EXCLUDED.weight,
                        description = COALESCE(EXCLUDED.description, graph_relationships.description),
                        updated_at = NOW()
                    RETURNING id, source_entity_id, target_entity_id, relation_type, description, weight, article_ids, evidence_texts, created_at, updated_at
                """)

                # Convert lists to PostgreSQL array format
                article_ids_str = [str(aid) for aid in rel_data["article_ids"]]
                evidence_texts = rel_data["evidence_texts"]

                result = await session.execute(
                    query,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type": rel_type,
                        "description": rel_data["description"],
                        "weight": rel_data["weight"],
                        "article_ids": article_ids_str,
                        "evidence_texts": evidence_texts,
                    },
                )

                row = result.fetchone()
                if row:
                    stored.append(GraphRelationship(
                        id=row.id,
                        source_entity_id=row.source_entity_id,
                        target_entity_id=row.target_entity_id,
                        relation_type=row.relation_type,
                        description=row.description,
                        weight=float(row.weight) if row.weight else 1.0,
                        article_ids=row.article_ids,
                        evidence_texts=row.evidence_texts,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    ))

            await session.flush()

            logger.info(
                "Stored relationships",
                count=len(stored),
            )

            return stored

        except Exception as e:
            raise DatabaseError(
                f"Failed to store relationships: {str(e)}",
                {"error": str(e)},
            )

    async def get_relationships_by_entities(
        self,
        session: AsyncSession,
        entity_ids: List[uuid.UUID],
    ) -> List[GraphRelationship]:
        """Get relationships where source or target is in entity_ids.

        Args:
            session: Database session
            entity_ids: List of entity IDs

        Returns:
            List of GraphRelationship objects
        """
        try:
            if not entity_ids:
                return []

            entity_id_strs = [str(eid) for eid in entity_ids]

            query = text("""
                SELECT * FROM graph_relationships
                WHERE source_entity_id = ANY(:entity_ids)
                   OR target_entity_id = ANY(:entity_ids)
            """)

            result = await session.execute(
                query,
                {"entity_ids": entity_id_strs},
            )

            relationships = []
            for row in result:
                rel = GraphRelationship(
                    id=row.id,
                    source_entity_id=row.source_entity_id,
                    target_entity_id=row.target_entity_id,
                    relation_type=row.relation_type,
                    description=row.description,
                    weight=float(row.weight) if row.weight else 1.0,
                    article_ids=row.article_ids,
                    evidence_texts=row.evidence_texts,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                relationships.append(rel)

            return relationships

        except Exception as e:
            raise DatabaseError(
                f"Failed to get relationships: {str(e)}",
                {"error": str(e)},
            )

    async def get_relationships_by_articles(
        self,
        session: AsyncSession,
        article_ids: List[uuid.UUID],
    ) -> List[GraphRelationship]:
        """Get relationships associated with given articles.

        Args:
            session: Database session
            article_ids: List of article IDs

        Returns:
            List of GraphRelationship objects
        """
        try:
            if not article_ids:
                return []

            query = text("""
                SELECT * FROM graph_relationships
                WHERE article_ids && :article_ids
            """)

            result = await session.execute(
                query,
                {"article_ids": [str(aid) for aid in article_ids]},
            )

            relationships = []
            for row in result:
                rel = GraphRelationship(
                    id=row.id,
                    source_entity_id=row.source_entity_id,
                    target_entity_id=row.target_entity_id,
                    relation_type=row.relation_type,
                    description=row.description,
                    weight=float(row.weight) if row.weight else 1.0,
                    article_ids=row.article_ids,
                    evidence_texts=row.evidence_texts,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                relationships.append(rel)

            return relationships

        except Exception as e:
            raise DatabaseError(
                f"Failed to get relationships by articles: {str(e)}",
                {"error": str(e)},
            )

    # === Expansion Operations ===

    async def get_1hop_neighbors(
        self,
        session: AsyncSession,
        entity_ids: List[uuid.UUID],
        limit: int = 50,
    ) -> tuple[List[GraphEntity], List[GraphRelationship]]:
        """Get 1-hop neighbors of given entities.

        Args:
            session: Database session
            entity_ids: Seed entity IDs
            limit: Maximum neighbors to return

        Returns:
            Tuple of (neighbor_entities, connecting_relationships)
        """
        try:
            if not entity_ids:
                return [], []

            entity_id_strs = [str(eid) for eid in entity_ids]

            # Get relationships connecting to neighbors
            rel_query = text("""
                SELECT * FROM graph_relationships
                WHERE source_entity_id = ANY(:entity_ids)
                   OR target_entity_id = ANY(:entity_ids)
                LIMIT :limit
            """)

            rel_result = await session.execute(
                rel_query,
                {"entity_ids": entity_id_strs, "limit": limit * 2},
            )

            neighbor_ids = set()
            relationships = []

            for row in rel_result:
                rel = GraphRelationship(
                    id=row.id,
                    source_entity_id=row.source_entity_id,
                    target_entity_id=row.target_entity_id,
                    relation_type=row.relation_type,
                    description=row.description,
                    weight=float(row.weight) if row.weight else 1.0,
                    article_ids=row.article_ids,
                    evidence_texts=row.evidence_texts,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                relationships.append(rel)

                # Collect neighbor IDs (excluding seed entities)
                if row.source_entity_id not in [uuid.UUID(e) for e in entity_id_strs]:
                    neighbor_ids.add(row.source_entity_id)
                if row.target_entity_id not in [uuid.UUID(e) for e in entity_id_strs]:
                    neighbor_ids.add(row.target_entity_id)

            if not neighbor_ids:
                return [], relationships

            # Get neighbor entities
            neighbor_list = list(neighbor_ids)[:limit]
            stmt = select(GraphEntity).where(GraphEntity.id.in_(neighbor_list))
            result = await session.execute(stmt)
            neighbors = list(result.scalars().all())

            logger.info(
                "Retrieved 1-hop neighbors",
                seed_count=len(entity_ids),
                neighbor_count=len(neighbors),
                relationship_count=len(relationships),
            )

            return neighbors, relationships

        except Exception as e:
            raise DatabaseError(
                f"Failed to get 1-hop neighbors: {str(e)}",
                {"error": str(e)},
            )

    # === Community Operations ===

    async def store_communities(
        self,
        session: AsyncSession,
        communities: List[dict],
    ) -> List[GraphCommunity]:
        """Store communities.

        Args:
            session: Database session
            communities: List of community dicts with keys:
                - name: str
                - summary: str
                - entity_ids: list[uuid.UUID]
                - hub_entity_id: uuid.UUID | None
                - article_ids: list[uuid.UUID]
                - level: int

        Returns:
            List of stored GraphCommunity objects
        """
        try:
            stored = []
            for comm_data in communities:
                community = GraphCommunity(
                    name=comm_data["name"],
                    summary=comm_data.get("summary"),
                    entity_ids=comm_data.get("entity_ids", []),
                    hub_entity_id=comm_data.get("hub_entity_id"),
                    article_ids=comm_data.get("article_ids", []),
                    level=comm_data.get("level", 0),
                )
                session.add(community)
                stored.append(community)

            await session.flush()

            logger.info(
                "Stored communities",
                count=len(stored),
            )

            return stored

        except Exception as e:
            raise DatabaseError(
                f"Failed to store communities: {str(e)}",
                {"error": str(e)},
            )

    async def get_communities_by_entities(
        self,
        session: AsyncSession,
        entity_ids: List[uuid.UUID],
    ) -> List[GraphCommunity]:
        """Get communities that contain any of the given entities.

        Args:
            session: Database session
            entity_ids: List of entity IDs

        Returns:
            List of GraphCommunity objects
        """
        try:
            if not entity_ids:
                return []

            query = text("""
                SELECT * FROM graph_communities
                WHERE entity_ids && :entity_ids
            """)

            result = await session.execute(
                query,
                {"entity_ids": [str(eid) for eid in entity_ids]},
            )

            communities = []
            for row in result:
                comm = GraphCommunity(
                    id=row.id,
                    name=row.name,
                    summary=row.summary,
                    entity_ids=row.entity_ids,
                    hub_entity_id=row.hub_entity_id,
                    article_ids=row.article_ids,
                    level=row.level,
                    created_at=row.created_at,
                )
                communities.append(comm)

            return communities

        except Exception as e:
            raise DatabaseError(
                f"Failed to get communities: {str(e)}",
                {"error": str(e)},
            )

    async def get_all_communities(
        self,
        session: AsyncSession,
        limit: int = 100,
        level: Optional[int] = None,
    ) -> List[GraphCommunity]:
        """Get all communities.

        Args:
            session: Database session
            limit: Maximum results
            level: Optional level filter

        Returns:
            List of GraphCommunity objects
        """
        try:
            stmt = select(GraphCommunity)
            if level is not None:
                stmt = stmt.where(GraphCommunity.level == level)
            stmt = stmt.order_by(GraphCommunity.created_at.desc()).limit(limit)

            result = await session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            raise DatabaseError(
                f"Failed to get communities: {str(e)}",
                {"error": str(e)},
            )

    # === Builder Run Operations ===

    async def create_builder_run(
        self,
        session: AsyncSession,
        article_ids: List[uuid.UUID],
    ) -> GraphBuilderRun:
        """Create a new builder run record.

        Args:
            session: Database session
            article_ids: IDs of articles to process

        Returns:
            Created GraphBuilderRun
        """
        try:
            run = GraphBuilderRun(
                article_ids=article_ids,
                status="running",
            )
            session.add(run)
            await session.flush()

            logger.info(
                "Created builder run",
                run_id=str(run.id),
                article_count=len(article_ids),
            )

            return run

        except Exception as e:
            raise DatabaseError(
                f"Failed to create builder run: {str(e)}",
                {"error": str(e)},
            )

    async def update_builder_run(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        status: str,
        entities_count: int = 0,
        relationships_count: int = 0,
        communities_count: int = 0,
        errors: Optional[List[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[GraphBuilderRun]:
        """Update a builder run record.

        Args:
            session: Database session
            run_id: Run UUID
            status: New status
            entities_count: Total entities extracted
            relationships_count: Total relationships extracted
            communities_count: Total communities detected
            errors: List of errors
            metadata: Additional metadata

        Returns:
            Updated GraphBuilderRun or None
        """
        try:
            stmt = select(GraphBuilderRun).where(GraphBuilderRun.id == run_id)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()

            if run:
                run.status = status
                run.completed_at = datetime.now(timezone.utc)
                run.entities_extracted = entities_count
                run.relationships_extracted = relationships_count
                run.communities_detected = communities_count
                run.errors = errors
                run.metadata_ = metadata
                await session.flush()

            return run

        except Exception as e:
            raise DatabaseError(
                f"Failed to update builder run: {str(e)}",
                {"run_id": str(run_id), "error": str(e)},
            )

    # === Utility Operations ===

    async def get_entity_count(self, session: AsyncSession) -> int:
        """Get total entity count."""
        result = await session.execute(select(func.count(GraphEntity.id)))
        return result.scalar() or 0

    async def get_relationship_count(self, session: AsyncSession) -> int:
        """Get total relationship count."""
        result = await session.execute(select(func.count(GraphRelationship.id)))
        return result.scalar() or 0

    async def get_community_count(self, session: AsyncSession) -> int:
        """Get total community count."""
        result = await session.execute(select(func.count(GraphCommunity.id)))
        return result.scalar() or 0

    async def search_entities_by_name(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 10,
    ) -> List[GraphEntity]:
        """Search entities by name (case-insensitive).

        Args:
            session: Database session
            query: Search query
            limit: Maximum results

        Returns:
            List of matching GraphEntity objects
        """
        try:
            stmt = select(GraphEntity).where(
                GraphEntity.name.ilike(f"%{query}%")
            ).limit(limit)

            result = await session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            raise DatabaseError(
                f"Failed to search entities: {str(e)}",
                {"query": query, "error": str(e)},
            )


# Singleton instance
graph_store = GraphStore()