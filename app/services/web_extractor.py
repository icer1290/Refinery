"""Web content extraction service using trafilatura."""

import asyncio
import ssl
from urllib.parse import urlparse
from typing import Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup

from app.core import get_logger
from app.core.exceptions import WebExtractionError

logger = get_logger(__name__)


class WebExtractor:
    """Extract main content from web pages."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
        }
        self._host_semaphores: dict[str, asyncio.Semaphore] = {}
        self._default_host_limit = 2
        self._strict_host_limits = {
            "venturebeat.com": 1,
            "www.venturebeat.com": 1,
            "openai.com": 1,
            "www.openai.com": 1,
            "phys.org": 1,
            "www.phys.org": 1,
        }

    async def extract_content(self, url: str) -> str:
        """Extract main content from a web page.

        Args:
            url: URL of the web page

        Returns:
            Extracted text content

        Raises:
            WebExtractionError: If extraction fails
        """
        try:
            # Fetch the page
            html_content = await self._fetch_page(url)

            # Extract text using trafilatura
            content = self._extract_with_trafilatura(html_content, url)

            if not content:
                # Fallback to BeautifulSoup extraction
                content = self._extract_with_beautifulsoup(html_content)

            if not content:
                raise WebExtractionError(
                    "Could not extract content from page",
                    {"url": url},
                )

            logger.info(
                "Extracted web content",
                url=url,
                content_length=len(content),
            )

            return content

        except WebExtractionError:
            raise
        except Exception as e:
            raise WebExtractionError(
                f"Failed to extract content: {type(e).__name__}: {str(e) or 'unknown error'}",
                {"url": url, "error": f"{type(e).__name__}: {str(e) or 'unknown error'}"},
            )

    async def _fetch_page(self, url: str) -> str:
        """Fetch page HTML content with retry logic.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string

        Raises:
            WebExtractionError: If fetch fails after all retries
        """
        last_error: Exception | None = None
        host = urlparse(url).netloc
        semaphore = self._get_host_semaphore(host)

        async with semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    ssl_context = ssl.create_default_context()
                    async with httpx.AsyncClient(
                        timeout=self.timeout, follow_redirects=True, verify=ssl_context
                    ) as client:
                        response = await client.get(
                            url,
                            headers=self._build_headers(url),
                        )
                        response.raise_for_status()
                        return response.text

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    last_error = e
                    if status_code == 429 and attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            "Web fetch rate limited, retrying...",
                            url=url,
                            host=host,
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            wait_seconds=wait_time,
                            status_code=status_code,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    raise WebExtractionError(
                        f"HTTP error fetching page: {status_code}",
                        {
                            "url": url,
                            "status_code": status_code,
                            "host": host,
                            "error": f"{type(e).__name__}: {str(e) or 'http error'}",
                        },
                    )

                except httpx.TimeoutException as e:
                    # Retry on timeout errors
                    last_error = e
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            "Web fetch timed out, retrying...",
                            url=url,
                            host=host,
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            wait_seconds=wait_time,
                            error=f"{type(e).__name__}",
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise WebExtractionError(
                            f"Timeout fetching page after {self.max_retries + 1} attempts: {type(e).__name__}",
                            {
                                "url": url,
                                "host": host,
                                "error": f"{type(e).__name__}: {str(e) or 'timeout'}",
                            },
                        )
                except httpx.RequestError as e:
                    # Retry on network errors (ConnectError, ReadError, etc.)
                    last_error = e
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            "Web fetch failed, retrying...",
                            url=url,
                            host=host,
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            wait_seconds=wait_time,
                            error=f"{type(e).__name__}",
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise WebExtractionError(
                            f"Request error fetching page after {self.max_retries + 1} attempts: {type(e).__name__}",
                            {
                                "url": url,
                                "host": host,
                                "error": f"{type(e).__name__}: {str(e) or 'unknown error'}",
                            },
                        )
                except WebExtractionError:
                    # Re-raise WebExtractionError without wrapping
                    raise
                except Exception as e:
                    # Don't retry on unexpected errors
                    raise WebExtractionError(
                        f"Error fetching page: {type(e).__name__}: {str(e) or 'unknown error'}",
                        {
                            "url": url,
                            "host": host,
                            "error": f"{type(e).__name__}: {str(e) or 'unknown error'}",
                        },
                    )

        # Should never reach here, but satisfy type checker
        raise WebExtractionError(
            f"Unexpected error fetching page: {type(last_error).__name__ if last_error else 'unknown'}",
            {"url": url, "error": str(last_error) if last_error else 'unknown error'},
        )

    def _build_headers(self, url: str) -> dict[str, str]:
        """Build browser-like headers for content fetches."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        headers = dict(self.headers)
        headers["Referer"] = origin
        return headers

    def _get_host_semaphore(self, host: str) -> asyncio.Semaphore:
        """Return a per-host semaphore to reduce rate-limit pressure."""
        if host not in self._host_semaphores:
            limit = self._strict_host_limits.get(host, self._default_host_limit)
            self._host_semaphores[host] = asyncio.Semaphore(limit)
        return self._host_semaphores[host]

    def _extract_with_trafilatura(
        self, html_content: str, url: str
    ) -> Optional[str]:
        """Extract content using trafilatura.

        Args:
            html_content: Raw HTML
            url: Source URL for context

        Returns:
            Extracted text or None
        """
        try:
            content = trafilatura.extract(
                html_content,
                url=url,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                favor_precision=True,
            )
            return content
        except Exception as e:
            logger.debug(
                "Trafilatura extraction failed",
                url=url,
                error=str(e),
            )
            return None

    def _extract_with_beautifulsoup(self, html_content: str) -> str:
        """Fallback extraction using BeautifulSoup.

        Args:
            html_content: Raw HTML

        Returns:
            Extracted text
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Try to find main content
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="content")
            or soup.find("div", class_="article")
            or soup.body
        )

        if not main_content:
            return ""

        # Extract text
        text = main_content.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    async def extract_batch(self, urls: list[str]) -> dict[str, str]:
        """Extract content from multiple URLs concurrently.

        Args:
            urls: List of URLs to extract

        Returns:
            Dict mapping URLs to extracted content
        """
        results = {}

        async def extract_one(url: str) -> tuple[str, str]:
            try:
                content = await self.extract_content(url)
                return url, content
            except WebExtractionError as e:
                logger.warning(
                    "Failed to extract URL",
                    url=url,
                    error=e.message,
                )
                return url, ""

        # Run extractions concurrently
        tasks = [extract_one(url) for url in urls]
        for url, content in await asyncio.gather(*tasks):
            results[url] = content

        return results


# Singleton instance
web_extractor = WebExtractor()
