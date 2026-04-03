"""Query expand tool for expanding queries into multiple related queries."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.services.query_transform import get_query_transform_service
from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = get_logger(__name__)


class QueryExpandTool(BaseTool):
    """Tool for expanding queries into multiple related queries."""

    name = "query_expand"
    description = "Expand a query into multiple related queries to improve search coverage. Use this when initial search results are insufficient."

    async def execute(
        self,
        session: AsyncSession,
        query: str,
        n: int = 3,
    ) -> str:
        """Expand query into multiple related queries.

        Args:
            session: Database session (unused)
            query: Original query
            n: Number of expanded queries

        Returns:
            JSON list of expanded queries
        """
        try:
            transform_service = get_query_transform_service()
            expanded = await transform_service.expand_query(query, n)
            return json.dumps(expanded, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Query expansion failed", error=str(e))
            return json.dumps([query], ensure_ascii=False)


# Auto-register on import
register_tool(QueryExpandTool())