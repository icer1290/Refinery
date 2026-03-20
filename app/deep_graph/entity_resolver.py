"""Entity resolver for deduplicating and merging entities.

Uses vector similarity to identify and merge duplicate entities.
"""

import uuid
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.deep_graph.graph_store import graph_store
from app.deep_graph.state import ExtractedEntity, ResolvedEntity
from app.services.embedding import get_embedding_service

logger = get_logger(__name__)
settings = get_settings()


class EntityResolver:
    """Resolves and deduplicates entities using vector similarity.

    Process:
    1. Generate embeddings for entity names/descriptions
    2. Search for similar existing entities
    3. Merge if similarity exceeds threshold
    4. Track resolution history
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embedding_service = get_embedding_service()
        self._resolution_map: Dict[str, str] = {}  # original_name -> canonical_name

    async def resolve_entities(
        self,
        session: AsyncSession,
        extracted_entities: List[tuple[str, ExtractedEntity]],  # (article_id, entity)
    ) -> List[ResolvedEntity]:
        """Resolve extracted entities against existing entities.

        Args:
            session: Database session
            extracted_entities: List of (article_id, entity) tuples

        Returns:
            List of resolved entities
        """
        logger.info(
            "Starting entity resolution",
            extracted_count=len(extracted_entities),
        )

        # Reset resolution map for this run
        self._resolution_map = {}

        # Group entities by name (case-insensitive) for initial grouping
        entity_groups: Dict[str, List[tuple[str, ExtractedEntity]]] = {}
        for article_id, entity in extracted_entities:
            key = entity["name"].lower().strip()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append((article_id, entity))

        # Resolve each unique entity
        resolved = []

        for name_key, group in entity_groups.items():
            # Take the first entity as representative
            first_article_id, first_entity = group[0]

            # Combine information from all instances
            all_article_ids = list(set(article_id for article_id, _ in group))
            all_mentions = []
            all_aliases = set()

            for _, entity in group:
                all_mentions.extend(entity.get("mentions", []))
                # Small variations in naming are aliases
                if entity["name"] != first_entity["name"]:
                    all_aliases.add(entity["name"])

            all_mentions = all_mentions[:5]  # Keep max 5 mentions
            all_aliases = list(all_aliases)

            # Generate embedding for entity
            embedding = None
            try:
                entity_text = f"{first_entity['name']}: {first_entity.get('description', '')}"
                embedding = await self.embedding_service.embed_text(entity_text)
            except Exception as e:
                logger.warning(
                    "Failed to generate entity embedding",
                    entity=first_entity["name"],
                    error=str(e),
                )

            # Search for similar existing entities
            canonical_name = first_entity["name"]
            existing_entity_id = None

            if embedding:
                similar = await graph_store.search_similar_entities(
                    session=session,
                    embedding=embedding,
                    limit=5,
                    similarity_threshold=self.similarity_threshold,
                )

                if similar:
                    # Use the most similar existing entity
                    best_match = similar[0]
                    logger.debug(
                        "Found similar entity",
                        name=first_entity["name"],
                        match=best_match.canonical_name,
                        similarity=best_match.similarity,
                    )
                    canonical_name = best_match.canonical_name
                    existing_entity_id = best_match.entity_id

            # Record resolution
            self._resolution_map[first_entity["name"]] = canonical_name

            resolved.append(ResolvedEntity(
                canonical_name=canonical_name,
                canonical_type=first_entity["type"],
                description=first_entity.get("description", ""),
                source_entity_names=[e["name"] for _, e in group],
                article_ids=all_article_ids,
                mention_count=len(group),
                embedding=embedding,
            ))

        logger.info(
            "Entity resolution complete",
            input_count=len(extracted_entities),
            resolved_count=len(resolved),
        )

        return resolved

    def get_resolution_map(self) -> Dict[str, str]:
        """Get the mapping from original names to canonical names."""
        return self._resolution_map


# Singleton instance
entity_resolver = EntityResolver()