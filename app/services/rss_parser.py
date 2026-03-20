"""RSS feed parsing service."""

import asyncio
import hashlib
import ssl
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx
from dateutil import parser as date_parser

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import RSSParseError

logger = get_logger(__name__)
settings = get_settings()


class RSSParser:
    """RSS feed parser with async support and time filtering."""

    def __init__(
        self,
        timeout: float = 30.0,
        hours_back: int = 24,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.timeout = timeout
        self.hours_back = hours_back
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "User-Agent": "TechNewsAggregator/0.1.0 (RSS Reader)"
        }

    async def fetch_feed(self, url: str) -> dict[str, Any]:
        """Fetch and parse an RSS feed.

        Args:
            url: URL of the RSS feed

        Returns:
            Parsed feed data

        Raises:
            RSSParseError: If feed parsing fails after all retries
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                # Create explicit SSL context to avoid async client issues in Docker
                ssl_context = ssl.create_default_context()
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True,
                    verify=ssl_context,
                ) as client:
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()

                # Parse the feed content
                feed = feedparser.parse(response.content)

                if feed.bozo and not feed.entries:
                    raise RSSParseError(
                        f"Failed to parse feed: {feed.bozo_exception}",
                        {"url": url, "error": str(feed.bozo_exception)},
                    )

                return feed

            except httpx.HTTPStatusError as e:
                # Don't retry on HTTP status errors (4xx, 5xx)
                raise RSSParseError(
                    f"HTTP error fetching feed: {e.response.status_code}",
                    {"url": url, "status_code": e.response.status_code},
                )
            except httpx.TimeoutException as e:
                # Retry on timeout errors
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "RSS fetch timed out, retrying...",
                        url=url,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        wait_seconds=wait_time,
                        error=f"{type(e).__name__}",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise RSSParseError(
                        f"Timeout fetching feed after {self.max_retries + 1} attempts: {type(e).__name__}",
                        {"url": url, "error": f"{type(e).__name__}: {str(e) or 'timeout'}"},
                    )
            except httpx.RequestError as e:
                # Retry on network errors (ConnectError, ReadError, etc.)
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "RSS fetch failed, retrying...",
                        url=url,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        wait_seconds=wait_time,
                        error=f"{type(e).__name__}",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise RSSParseError(
                        f"Request error fetching feed after {self.max_retries + 1} attempts: {type(e).__name__}",
                        {"url": url, "error": f"{type(e).__name__}: {str(e) or 'unknown error'}"},
                    )
            except RSSParseError:
                # Re-raise RSSParseError without wrapping
                raise
            except Exception as e:
                # Don't retry on unexpected errors
                raise RSSParseError(
                    f"Error fetching feed: {type(e).__name__}: {str(e) or 'unknown error'}",
                    {"url": url, "error": f"{type(e).__name__}: {str(e) or 'unknown error'}"},
                )

        # Should never reach here, but satisfy type checker
        raise RSSParseError(
            f"Unexpected error fetching feed: {type(last_error).__name__ if last_error else 'unknown'}",
            {"url": url, "error": str(last_error) if last_error else 'unknown error'},
        )

    async def parse_entries(
        self, feed_url: str, feed_name: str
    ) -> list[dict[str, Any]]:
        """Parse entries from an RSS feed, filtering by time.

        Only returns entries published within the last `hours_back` hours.

        Args:
            feed_url: URL of the RSS feed
            feed_name: Name of the feed source

        Returns:
            List of parsed entries with standardized fields (last 24h only)
        """
        feed = await self.fetch_feed(feed_url)
        entries = []
        skipped_old = 0

        # Calculate time threshold
        now = datetime.now(timezone.utc)
        time_threshold = now - timedelta(hours=self.hours_back)

        for entry in feed.entries:
            try:
                parsed_entry = self._parse_entry(entry, feed_name)

                # Filter by publish time
                published_at = parsed_entry.get("published_at")

                if published_at:
                    # Ensure timezone-aware comparison
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)

                    if published_at < time_threshold:
                        skipped_old += 1
                        continue

                entries.append(parsed_entry)

            except Exception as e:
                logger.warning(
                    "Failed to parse entry",
                    feed_name=feed_name,
                    entry_link=getattr(entry, "link", None),
                    error=str(e),
                )
                continue

        logger.info(
            "Parsed RSS feed",
            feed_name=feed_name,
            feed_url=feed_url,
            entries_count=len(entries),
            skipped_old=skipped_old,
            hours_back=self.hours_back,
        )

        return entries

    def _parse_entry(self, entry: Any, source_name: str) -> dict[str, Any]:
        """Parse a single RSS entry.

        Args:
            entry: feedparser entry object
            source_name: Name of the source

        Returns:
            Parsed entry dict
        """
        # Extract required fields
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")

        if not title or not link:
            raise ValueError("Entry missing required fields: title or link")

        # Extract description
        description = getattr(entry, "description", "") or getattr(
            entry, "summary", ""
        )

        # Parse published date
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif hasattr(entry, "published"):
            try:
                published_at = date_parser.parse(entry.published)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        # Also try updated date if published date is missing
        if published_at is None:
            if hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated"):
                try:
                    published_at = date_parser.parse(entry.updated)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

        # Generate content hash for deduplication
        content_hash = self._generate_hash(title + description)

        return {
            "source_name": source_name,
            "source_url": link,
            "original_title": title,
            "original_description": description,
            "published_at": published_at,
            "content_hash": content_hash,
        }

    def _generate_hash(self, content: str) -> str:
        """Generate SHA256 hash for content deduplication.

        Args:
            content: Content to hash

        Returns:
            Hexadecimal hash string
        """
        normalized = content.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()


# Singleton instance
rss_parser = RSSParser()