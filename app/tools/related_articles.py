"""Tool for finding related articles using vector search."""

import json
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.tools import execute_tool
from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = get_logger(__name__)


class RelatedArticlesTool(BaseTool):
    """Tool for finding related articles using vector search.

    Finds articles similar to a given article or query.
    """

    name = "related_articles"
    description = """Find articles related to a topic or article.
Use to discover related news or background information.

Input:
- query: Search query or article topic (required)
- limit: Maximum results (default: 5)
"""

    async def execute(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
    ) -> str:
        """Find related articles.

        This is a wrapper around vector_search tool with
        article-specific formatting.

        Args:
            session: Database session
            query: Search query
            limit: Max results

        Returns:
            JSON string with related articles
        """
        try:
            # Use vector search from tools
            result = await execute_tool(
                session,
                "vector_search",
                {"query": query, "limit": limit}
            )

            # Parse and reformat for chat context
            try:
                articles = json.loads(result)
                if isinstance(articles, list):
                    formatted = []
                    for article in articles:
                        formatted.append({
                            "title": article.get("title", ""),
                            "summary": article.get("summary", "")[:200],
                            "source": article.get("source_name", ""),
                            "url": article.get("url", ""),
                            "score": article.get("score", 0),
                        })
                    return json.dumps({
                        "query": query,
                        "related_articles": formatted,
                    }, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass

            return result

        except Exception as e:
            logger.error("Related articles search failed", error=str(e))
            return json.dumps({"error": str(e)})


# Register tool at module load time
register_tool(RelatedArticlesTool())