"""Web search service using DuckDuckGo (free, no API key required)."""

import asyncio
import random
import re
import ssl
import time
from typing import Optional
from urllib.parse import quote_plus, unquote

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import ExternalServiceError

logger = get_logger(__name__)
settings = get_settings()

# Try to import the maintained DDGS package first.
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False
        logger.warning("DDGS client not installed, using fallback HTML scraping")

# Rate limiting configuration
MIN_SEARCH_INTERVAL = 2.0  # Minimum seconds between searches
MAX_SEARCH_INTERVAL = 4.0  # Maximum seconds between searches
_last_search_time = 0.0  # Track last search timestamp


class WebSearchResult:
    """Represents a web search result."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class WebSearchService:
    """Service for performing web searches using DuckDuckGo."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or getattr(settings, "web_search_provider", "duckduckgo")
        self.api_key = api_key or getattr(settings, "web_search_api_key", None)
        self.timeout = 30.0

        logger.info(
            "Web search service initialized",
            provider=self.provider,
        )

    async def _wait_for_rate_limit(self) -> None:
        """Wait before making a search to avoid rate limiting.

        Uses random delay to simulate human behavior.
        """
        global _last_search_time

        current_time = time.time()
        time_since_last = current_time - _last_search_time

        # Random delay between MIN and MAX interval
        required_delay = random.uniform(MIN_SEARCH_INTERVAL, MAX_SEARCH_INTERVAL)

        if time_since_last < required_delay:
            wait_time = required_delay - time_since_last
            logger.debug(
                "Rate limiting: waiting before search",
                wait_seconds=round(wait_time, 2),
            )
            await asyncio.sleep(wait_time)

        _last_search_time = time.time()

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Perform a web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results

        Raises:
            ExternalServiceError: If search fails
        """
        if self.provider == "tavily" and self.api_key:
            return await self._search_tavily(query, max_results)
        else:
            return await self._search_duckduckgo(query, max_results)

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search using DuckDuckGo.

        Primary method: duckduckgo-search library (more reliable)
        Fallback: Direct HTML scraping

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        # Wait to avoid rate limiting
        await self._wait_for_rate_limit()

        # Try using duckduckgo-search library first (more reliable)
        if HAS_DDGS:
            try:
                results = await self._search_with_ddgs(query, max_results)
                if results:
                    return results
                logger.info("DDGS returned no results, falling back to HTML scraping")
            except Exception as e:
                logger.warning("DDGS search failed, falling back to HTML scraping", error=str(e))

        # Fallback to HTML scraping
        results = await self._search_duckduckgo_html(query, max_results)
        if results:
            return results

        refined_query = self._build_fallback_query(query)
        if refined_query and refined_query != query:
            logger.info(
                "Retrying DuckDuckGo search with refined query",
                original_query=query[:50],
                refined_query=refined_query[:50],
            )

            if HAS_DDGS:
                try:
                    results = await self._search_with_ddgs(refined_query, max_results)
                    if results:
                        return results
                except Exception as e:
                    logger.warning(
                        "Refined DDGS search failed, falling back to HTML scraping",
                        error=str(e),
                    )

            return await self._search_duckduckgo_html(refined_query, max_results)

        return results

    async def _search_with_ddgs(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search using duckduckgo-search library.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        # DDGS is synchronous, run in executor
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results

        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(None, _search)

        results = []
        for item in search_results:
            results.append(WebSearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
            ))

        logger.info(
            "DuckDuckGo search completed (DDGS library)",
            query=query[:50],
            results_count=len(results),
        )

        return results

    async def _search_duckduckgo_html(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search using DuckDuckGo HTML version.

        Uses session-based approach with cookies for better reliability.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
        }

        try:
            # Use a session with cookies for better reliability
            ssl_context = ssl.create_default_context()
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers,
                verify=ssl_context,
            ) as client:
                # First visit the main page to establish cookies
                try:
                    await client.get("https://duckduckgo.com/")
                    # Small delay to appear more natural
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.debug("Failed to visit DuckDuckGo main page", error=str(e))

                # Then perform the search
                encoded_query = quote_plus(query)
                url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                response = await client.get(url)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Check for empty results or anti-bot detection
            page_text = response.text.lower()
            if "no results" in page_text or "no results found" in page_text:
                logger.info(
                    "DuckDuckGo returned no results for query",
                    query=query[:50],
                )
                return results

            # Find all result divs
            result_divs = soup.find_all("div", class_="result")

            # Alternative parsing: try finding results by link class
            if not result_divs:
                result_links = soup.find_all("a", class_="result__a")
                for link in result_links[:max_results]:
                    try:
                        title = link.get_text(strip=True)
                        href = link.get("href", "")

                        # DuckDuckGo uses redirect URLs, extract actual URL
                        if "uddg=" in href:
                            href = href.split("uddg=")[-1].split("&")[0]
                            href = unquote(href)

                        # Get parent div for snippet
                        parent = link.find_parent("div", class_="result")
                        snippet = ""
                        if parent:
                            snippet_elem = parent.find("a", class_="result__snippet")
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                        if title and href:
                            results.append(WebSearchResult(
                                title=title,
                                url=href,
                                snippet=snippet,
                            ))
                    except Exception as e:
                        logger.debug(f"Failed to parse result: {e}")
                        continue

                logger.info(
                    "DuckDuckGo search completed (alternative parser)",
                    query=query[:50],
                    results_count=len(results),
                )
                return results

            # Standard parsing with result divs
            for div in result_divs[:max_results]:
                try:
                    # Get title and link
                    title_elem = div.find("a", class_="result__a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    href = title_elem.get("href", "")

                    # DuckDuckGo uses redirect URLs, extract actual URL
                    if "uddg=" in href:
                        href = href.split("uddg=")[-1].split("&")[0]
                        href = unquote(href)

                    # Get snippet
                    snippet_elem = div.find("a", class_="result__snippet")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if title and href:
                        results.append(WebSearchResult(
                            title=title,
                            url=href,
                            snippet=snippet,
                        ))

                except Exception as e:
                    logger.debug(f"Failed to parse result: {e}")
                    continue

            # Log warning if no results found but page didn't explicitly say "no results"
            if not results and "did you mean" not in page_text:
                logger.warning(
                    "DuckDuckGo may have failed silently - no results but no 'no results' message",
                    query=query[:50],
                    html_length=len(response.text),
                )

            logger.info(
                "DuckDuckGo search completed",
                query=query[:50],
                results_count=len(results),
            )

            return results

        except httpx.TimeoutException:
            logger.error("DuckDuckGo search timeout", query=query[:50])
            raise ExternalServiceError(
                "Web search timeout",
                {"query": query, "error": "timeout"},
            )
        except Exception as e:
            logger.error("DuckDuckGo search failed", error=str(e), query=query[:50])
            raise ExternalServiceError(
                f"Web search failed: {str(e)}",
                {"query": query, "error": str(e)},
            )

    def _build_fallback_query(self, query: str) -> str:
        """Simplify long or mixed-language queries when the first attempt returns nothing."""
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return normalized

        tokens = normalized.split(" ")
        filtered_tokens = [
            token for token in tokens
            if token and not re.fullmatch(r"\d{4}(?:-\d{4})?", token)
        ]

        if len(filtered_tokens) > 7:
            filtered_tokens = filtered_tokens[:7]

        refined = " ".join(filtered_tokens)

        if re.search(r"[\u4e00-\u9fff]", refined) and re.search(r"[A-Za-z]", refined):
            refined = re.sub(r"\s+", " ", refined)

        return refined

    async def _search_tavily(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        url = "https://api.tavily.com/search"

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }

        headers = {
            "Content-Type": "application/json",
        }

        try:
            ssl_context = ssl.create_default_context()
            async with httpx.AsyncClient(timeout=self.timeout, verify=ssl_context) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("results", [])[:max_results]:
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                ))

            logger.info(
                "Tavily search completed",
                query=query[:50],
                results_count=len(results),
            )

            return results

        except Exception as e:
            logger.error("Tavily search failed", error=str(e), query=query[:50])
            raise ExternalServiceError(
                f"Tavily search failed: {str(e)}",
                {"query": query, "error": str(e)},
            )

    async def fetch_page_content(self, url: str) -> str:
        """Fetch and extract text content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }

        try:
            ssl_context = ssl.create_default_context()
            async with httpx.AsyncClient(timeout=self.timeout, verify=ssl_context) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Limit length
            if len(text) > 2000:
                text = text[:2000] + "..."

            return text

        except Exception as e:
            logger.warning("Failed to fetch page content", url=url, error=str(e))
            return ""


# Singleton instance
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    """Get or create web search service instance."""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service
