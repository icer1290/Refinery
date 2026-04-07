"""Tool for retrieving a specific article by ID."""

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_logger
from app.models.orm_models import NewsArticle
from app.tools.base import BaseTool
from app.tools.registry import register_tool

logger = get_logger(__name__)


class ArticleLookupTool(BaseTool):
    """Tool for retrieving a specific article by ID.

    Fetches article details including title, summary, content,
    and deepsearch report if available.
    """

    name = "article_lookup"
    description = """Retrieve full details of a specific article by its ID.
Use when user references a specific article or needs full content.
Returns article title, summary, content, source, and deepsearch report if available.

Input:
- article_id: UUID of the article (required)
- include_deepsearch: Whether to include deepsearch report (default: true)
"""

    async def execute(
        self,
        session: AsyncSession,
        article_id: str,
        include_deepsearch: bool = True,
    ) -> str:
        """Fetch article details.

        Args:
            session: Database session
            article_id: Article UUID
            include_deepsearch: Include deepsearch report

        Returns:
            JSON string with article details
        """
        # Validate UUID format
        try:
            uuid_obj = UUID(article_id)
        except ValueError:
            return json.dumps({
                "error": f"Invalid article ID format: '{article_id}'. Expected UUID format."
            })

        try:
            stmt = select(NewsArticle).where(NewsArticle.id == uuid_obj)
            result = await session.execute(stmt)
            article = result.scalar_one_or_none()

            if not article:
                return json.dumps({"error": f"Article not found: {article_id}"})

            article_data = {
                "id": str(article.id),
                "title": article.chinese_title or article.original_title,
                "original_title": article.original_title,
                "summary": article.chinese_summary or article.original_description,
                "content": article.full_content[:2000] if article.full_content else None,  # Truncate
                "source_name": article.source_name,
                "source_url": article.source_url,
                "published_at": str(article.published_at) if article.published_at else None,
                "total_score": article.total_score,
            }

            if include_deepsearch and article.deepsearch_report:
                # Include truncated deepsearch report
                article_data["deepsearch_report"] = article.deepsearch_report[:1000] + "..."

            return json.dumps(article_data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Article lookup failed", error=str(e), article_id=article_id)
            return json.dumps({"error": str(e)})


# Register tool at module load time
register_tool(ArticleLookupTool())