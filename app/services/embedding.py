"""Embedding service using OpenAI-compatible APIs."""

import traceback
from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import EmbeddingError

logger = get_logger(__name__)
settings = get_settings()

DASHSCOPE_MAX_BATCH_SIZE = 10


class EmbeddingService:
    """Service for generating text embeddings using OpenAI-compatible APIs.

    Supports both standard OpenAI API and DashScope API (Alibaba Cloud).
    DashScope's "compatible-mode" doesn't fully support OpenAI's embedding API format,
    so we use direct HTTP calls with DashScope-specific request format.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or settings.openai_embedding_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        if not self.api_key:
            raise EmbeddingError("API key not configured")

        # Determine if we're using DashScope
        self._is_dashscope = self.base_url and "dashscope" in self.base_url.lower()

        logger.info(
            "Embedding service initialized",
            model=self.model,
            base_url=self.base_url or "default",
            is_dashscope=self._is_dashscope,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Truncate text if too long (OpenAI has token limits)
            truncated_text = self._truncate_text(text)

            # Use DashScope-specific API if detected
            if self._is_dashscope:
                embedding = await self._embed_with_dashscope(truncated_text)
            else:
                embedding = await self._embed_with_openai(truncated_text)

            logger.debug(
                "Generated embedding",
                text_length=len(text),
                embedding_dim=len(embedding),
            )

            return embedding

        except Exception as e:
            # 详细错误日志
            error_detail = f"{type(e).__name__}: {str(e)}"
            logger.error(
                "Embedding generation failed",
                error=error_detail,
                model=self.model,
                text_preview=text[:100],
                traceback=traceback.format_exc(),
            )
            raise EmbeddingError(
                f"Failed to generate embedding: {error_detail}",
                {"error": str(e), "model": self.model},
            )

    async def _embed_with_dashscope(self, text: str) -> List[float]:
        """Call DashScope embedding API directly with correct format.

        DashScope's compatible-mode uses standard OpenAI format:
        {"model": "...", "input": "text"} for single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # DashScope API endpoint for embeddings
        url = f"{self.base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Standard OpenAI-compatible format (DashScope compatible-mode expects this)
        payload = {
            "model": self.model,
            "input": text,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    "DashScope embedding API error",
                    status_code=response.status_code,
                    response=error_text[:500],
                )
                raise EmbeddingError(
                    f"DashScope API error: {response.status_code}",
                    {"status_code": response.status_code, "response": error_text[:200]},
                )

            data = response.json()

            # Parse OpenAI-compatible response format
            # Response: {"data": [{"embedding": [...], "index": 0}], "model": "...", ...}
            try:
                return data["data"][0]["embedding"]
            except (KeyError, IndexError) as e:
                logger.error(
                    "Unexpected DashScope response format",
                    response=data,
                    error=str(e),
                )
                raise EmbeddingError(
                    f"Failed to parse DashScope response: {e}",
                    {"response": str(data)[:200]},
                )

    async def _embed_with_openai(self, text: str) -> List[float]:
        """Call OpenAI-compatible embedding API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Import OpenAI SDK only when needed (for non-DashScope endpoints)
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        response = await client.embeddings.create(
            model=self.model,
            input=text,
        )

        return list(response.data[0].embedding)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Truncate texts
            truncated_texts = [self._truncate_text(text) for text in texts]

            # Use DashScope-specific API if detected
            if self._is_dashscope:
                embeddings = []
                batches = list(self._chunk_texts(truncated_texts, DASHSCOPE_MAX_BATCH_SIZE))
                logger.info(
                    "Generating DashScope batch embeddings",
                    total_count=len(truncated_texts),
                    batch_count=len(batches),
                    batch_sizes=[len(batch) for batch in batches],
                    model=self.model,
                )
                for batch in batches:
                    embeddings.extend(await self._embed_batch_with_dashscope(batch))
            else:
                embeddings = await self._embed_batch_with_openai(truncated_texts)

            logger.info(
                "Generated batch embeddings",
                count=len(embeddings),
                model=self.model,
            )

            return embeddings

        except Exception as e:
            error_detail = f"{type(e).__name__}: {str(e)}"
            logger.error(
                "Batch embedding generation failed",
                error=error_detail,
                model=self.model,
                count=len(texts),
                traceback=traceback.format_exc(),
            )
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {error_detail}",
                {"error": str(e), "model": self.model, "count": len(texts)},
            )

    def _chunk_texts(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Split texts into fixed-size batches while preserving order."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    async def _embed_batch_with_dashscope(self, texts: List[str]) -> List[List[float]]:
        """Call DashScope embedding API for batch texts.

        DashScope compatible-mode uses standard OpenAI format:
        {"model": "...", "input": ["text1", "text2"]} for batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Standard OpenAI-compatible format for batch
        payload = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    "DashScope batch embedding API error",
                    status_code=response.status_code,
                    response=error_text[:500],
                )
                raise EmbeddingError(
                    f"DashScope API error: {response.status_code}",
                    {"status_code": response.status_code, "response": error_text[:200]},
                )

            data = response.json()

            try:
                # Parse OpenAI-compatible response format
                # Response: {"data": [{"embedding": [...], "index": 0}, ...], ...}
                embeddings_data = data["data"]
                # Sort by index to ensure correct order
                embeddings_data.sort(key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in embeddings_data]
            except (KeyError, IndexError) as e:
                logger.error(
                    "Unexpected DashScope batch response format",
                    response=data,
                    error=str(e),
                )
                raise EmbeddingError(
                    f"Failed to parse DashScope batch response: {e}",
                    {"response": str(data)[:200]},
                )

    async def _embed_batch_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI-compatible embedding API for batch texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        response = await client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [list(item.embedding) for item in sorted_data]

    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to fit within token limits.

        OpenAI embedding models have token limits. We use a conservative
        character-based approximation.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    async def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Compute magnitudes
        mag1 = sum(a * a for a in embedding1) ** 0.5
        mag2 = sum(b * b for b in embedding2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
