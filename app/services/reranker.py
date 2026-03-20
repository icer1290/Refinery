"""Reranker service for improving search result quality.

Uses DashScope gte-rerank API to re-rank search results based on relevance
to the query.

Note: DashScope rerank API uses a native endpoint, NOT the OpenAI-compatible mode:
https://dashscope.aliyuncs.com/api/v1/services/rerank
"""

import traceback
from typing import List, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import EmbeddingError
from app.services.vector_store import SearchResult

logger = get_logger(__name__)
settings = get_settings()


class RerankerService:
    """Service for re-ranking search results using DashScope gte-rerank.

    Re-ranking improves the quality of search results by using a
    cross-encoder model that jointly considers the query and each
    document to compute a relevance score.

    Note: Uses DashScope native API endpoint (not OpenAI-compatible mode).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the reranker service.

        Args:
            model: Rerank model name (default from config)
            api_key: API key (default from settings)
            base_url: API base URL (default from settings)
        """
        self.model = model or settings.rag_rerank_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        if not self.api_key:
            logger.warning("No API key configured for reranker")

        logger.info(
            "RerankerService initialized",
            model=self.model,
            base_url=self.base_url or "default",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int | None = None,
    ) -> List[Tuple[SearchResult, float]]:
        """Re-rank search results by relevance to query.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return (default: all)

        Returns:
            List of (SearchResult, rerank_score) tuples sorted by relevance

        Raises:
            EmbeddingError: If reranking fails
        """
        if not results:
            return []

        if not self.api_key:
            logger.warning("No API key for reranker, returning original order")
            return [(r, r.similarity) for r in results[:top_k]]

        top_k = top_k or len(results)

        try:
            # Prepare documents for reranking
            documents = [r.chunk_text for r in results]

            # Call rerank API
            scores = await self._call_rerank_api(query, documents)

            # Pair results with scores and sort
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                "Reranking completed",
                query=query[:50],
                input_count=len(results),
                output_count=min(top_k, len(scored_results)),
            )

            return scored_results[:top_k]

        except Exception as e:
            logger.error(
                "Reranking failed, falling back to original order",
                error=str(e),
                query=query[:50],
            )
            # Fallback to original order on failure
            return [(r, r.similarity) for r in results[:top_k]]

    async def _call_rerank_api(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Call the rerank API.

        Supports DashScope and OpenAI-compatible APIs.
        DashScope format: https://help.aliyun.com/zh/model-studio/developer-reference/text-reranking-api

        Args:
            query: The search query
            documents: List of documents to rank

        Returns:
            List of relevance scores
        """
        is_dashscope = self.base_url and "dashscope" in self.base_url.lower()

        if is_dashscope:
            return await self._call_dashscope_rerank(query, documents)
        else:
            # For other APIs, try a generic rerank endpoint
            return await self._call_generic_rerank(query, documents)

    async def _call_dashscope_rerank(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Call DashScope rerank API.

        DashScope rerank API uses a different endpoint than OpenAI-compatible mode:
        https://dashscope.aliyuncs.com/api/v1/services/rerank

        Args:
            query: The search query
            documents: List of documents to rank

        Returns:
            List of relevance scores
        """
        # DashScope rerank uses native API, not OpenAI-compatible mode
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    "DashScope rerank API error",
                    status_code=response.status_code,
                    response=error_text[:500],
                )
                raise EmbeddingError(
                    f"DashScope rerank API error: {response.status_code}",
                    {"status_code": response.status_code, "response": error_text[:200]},
                )

            data = response.json()

            try:
                # DashScope returns: {"output": {"results": [{"index": 0, "relevance_score": 0.9}, ...]}}
                results = data["output"]["results"]
                # Sort by index to match original order, then extract scores
                results.sort(key=lambda x: x["index"])
                return [r["relevance_score"] for r in results]
            except (KeyError, IndexError) as e:
                logger.error(
                    "Unexpected DashScope rerank response format",
                    response=data,
                    error=str(e),
                )
                raise EmbeddingError(
                    f"Failed to parse DashScope rerank response: {e}",
                    {"response": str(data)[:200]},
                )

    async def _call_generic_rerank(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Call a generic rerank API (Cohere-style).

        Args:
            query: The search query
            documents: List of documents to rank

        Returns:
            List of relevance scores
        """
        url = f"{self.base_url}/rerank"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    "Rerank API error",
                    status_code=response.status_code,
                    response=error_text[:500],
                )
                raise EmbeddingError(
                    f"Rerank API error: {response.status_code}",
                    {"status_code": response.status_code, "response": error_text[:200]},
                )

            data = response.json()

            try:
                # Cohere-style response: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
                results = data["results"]
                results.sort(key=lambda x: x["index"])
                return [r["relevance_score"] for r in results]
            except (KeyError, IndexError) as e:
                logger.error(
                    "Unexpected rerank response format",
                    response=data,
                    error=str(e),
                )
                raise EmbeddingError(
                    f"Failed to parse rerank response: {e}",
                    {"response": str(data)[:200]},
                )


# Singleton instance
_reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    """Get or create reranker service instance."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service