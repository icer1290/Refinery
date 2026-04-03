"""Web search tool for searching the web for information."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.services.web_search import get_web_search_service
from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Tool for searching the web for information."""

    name = "web_search"
    description = "Search the web for information about companies, technologies, and events. Use this to find external background information not available in the local database."

    async def execute(
        self,
        session: AsyncSession,
        query: str,
    ) -> str:
        """Execute web search.

        Args:
            session: Database session (unused but required for interface)
            query: Search query

        Returns:
            Formatted search results
        """
        try:
            web_search = get_web_search_service()
            results = await web_search.search(query, max_results=5)

            if not results:
                return "No results found on the web."

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "snippet": result.snippet,
                    "url": result.url,
                })

            logger.info(
                "Web search completed",
                query=query[:50],
                results_count=len(results),
            )

            return json.dumps(formatted_results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Web search failed", error=str(e), query=query[:50])
            return f"Web search failed: {str(e)}"


# Auto-register on import
register_tool(WebSearchTool())