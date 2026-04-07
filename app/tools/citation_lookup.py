"""Tool for verifying and enriching citations from knowledge graph."""

import json
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.models.orm_models import GraphEntity
from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = get_logger(__name__)


class CitationLookupTool(BaseTool):
    """Tool for verifying and enriching citations from knowledge graph.

    Searches for entities mentioned in the article to provide
    additional context and verification.
    """

    name = "citation_lookup"
    description = """Verify and enrich citation information from the knowledge graph.
Use to find related entities, companies, or people mentioned in articles.

Input:
- entity_name: Name of the entity to look up (required)
- entity_type: Optional type filter (PERSON, ORGANIZATION, TECHNOLOGY)
"""

    async def execute(
        self,
        session: AsyncSession,
        entity_name: str,
        entity_type: Optional[str] = None,
    ) -> str:
        """Find entity and related information.

        Args:
            session: Database session
            entity_name: Name to search for
            entity_type: Optional type filter

        Returns:
            JSON string with entity details
        """
        try:
            # Build query
            stmt = select(GraphEntity).where(
                GraphEntity.name.ilike(f"%{entity_name}%")
            )

            if entity_type:
                stmt = stmt.where(GraphEntity.type == entity_type.upper())

            stmt = stmt.limit(5)

            result = await session.execute(stmt)
            entities = result.scalars().all()

            if not entities:
                return json.dumps({
                    "found": False,
                    "query": entity_name,
                    "message": "No matching entities found in knowledge graph"
                })

            entity_list = []
            for entity in entities:
                entity_list.append({
                    "id": str(entity.id),
                    "name": entity.name,
                    "canonical_name": entity.canonical_name,
                    "type": entity.type,
                    "description": entity.description,
                    "mention_count": entity.mention_count,
                    "aliases": entity.aliases[:5] if entity.aliases else [],
                })

            return json.dumps({
                "found": True,
                "query": entity_name,
                "entities": entity_list,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Citation lookup failed", error=str(e), entity_name=entity_name)
            return json.dumps({"error": str(e)})


# Register tool at module load time
register_tool(CitationLookupTool())