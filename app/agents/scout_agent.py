"""Scout Agent - RSS feed fetching and parsing."""

import asyncio
from typing import Any

from app.agents.base import BaseAgent
from app.config import get_settings
from app.core.exceptions import RSSParseError
from app.services.rss_parser import rss_parser

settings = get_settings()


class ScoutAgent(BaseAgent):
    """Agent responsible for fetching and parsing RSS feeds."""

    def __init__(self):
        super().__init__("Scout")
        self.rss_parser = rss_parser

    async def execute(self, feed_urls: list[str] | None = None) -> list[dict[str, Any]]:
        """Fetch and parse RSS feeds.

        Args:
            feed_urls: List of RSS feed URLs to fetch.
                      If None, uses default feeds from config.

        Returns:
            List of parsed articles with standardized fields
        """
        urls = feed_urls or settings.default_rss_feeds

        # Fetch feeds concurrently
        tasks = [self._fetch_feed(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all articles
        all_articles = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(
                    "Feed fetch failed",
                    error=str(result),
                )
                continue
            all_articles.extend(result)

        self.logger.info(
            "Scout phase complete",
            feeds_fetched=len(urls),
            total_articles=len(all_articles),
        )

        return all_articles

    async def _fetch_feed(self, url: str) -> list[dict[str, Any]]:
        """Fetch a single RSS feed.

        Args:
            url: RSS feed URL

        Returns:
            List of parsed articles from this feed
        """
        try:
            # Extract feed name from URL
            feed_name = self._extract_feed_name(url)

            entries = await self.rss_parser.parse_entries(url, feed_name)
            return entries

        except RSSParseError as e:
            self.logger.warning(
                "Failed to parse RSS feed",
                url=url,
                error=e.message,
                details=e.details,
            )
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error fetching RSS feed",
                url=url,
                error=str(e),
            )
            raise RSSParseError(
                f"Unexpected error: {str(e)}",
                {"url": url},
            )

    def _extract_feed_name(self, url: str) -> str:
        """Extract a readable name from RSS feed URL.

        Args:
            url: RSS feed URL

        Returns:
            Feed name for display
        """
        # Try to extract domain name
        if "://" in url:
            domain = url.split("://")[1].split("/")[0]
            # Remove www. prefix
            domain = domain.replace("www.", "")
            return domain
        return url[:50]


# Singleton instance
scout_agent = ScoutAgent()