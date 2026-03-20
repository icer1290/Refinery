"""Query transformation service for RAG system.

Provides advanced query processing techniques:
- HyDE (Hypothetical Document Embedding): Generate a hypothetical document
  that would answer the query, then use it for retrieval.
- Multi-query expansion: Generate multiple related queries to improve recall.
"""

import json
from typing import List

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core import get_logger
from app.prompts import get_prompt

logger = get_logger(__name__)
settings = get_settings()


class QueryTransformService:
    """Service for transforming and expanding queries for better retrieval.

    Uses LLM to:
    1. Generate hypothetical documents (HyDE)
    2. Expand queries into multiple related queries
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the query transform service.

        Args:
            model: Chat model name (default from config)
            api_key: API key (default from settings)
            base_url: API base URL (default from settings)
        """
        self.model = model or settings.openai_chat_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.3,
        )

        logger.info(
            "QueryTransformService initialized",
            model=self.model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_hypothetical_document(
        self,
        query: str,
        doc_length: int = 500,
    ) -> str:
        """Generate a hypothetical document that would answer the query.

        HyDE improves retrieval by generating a document that would ideally
        contain the answer, then using that document's embedding for search.

        Args:
            query: The search query
            doc_length: Target length of the hypothetical document

        Returns:
            Generated hypothetical document text
        """
        try:
            prompt = get_prompt("rag.hyde").format(
                query=query,
                doc_length=str(doc_length),
            )

            response = await self.llm.ainvoke(prompt)
            hypothetical_doc = response.content

            logger.debug(
                "Generated hypothetical document",
                query=query[:50],
                doc_length=len(hypothetical_doc),
            )

            return hypothetical_doc

        except Exception as e:
            logger.error(
                "Failed to generate hypothetical document",
                error=str(e),
                query=query[:50],
            )
            # Return original query as fallback
            return query

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def expand_query(
        self,
        query: str,
        n: int = 3,
    ) -> List[str]:
        """Expand a query into multiple related queries.

        Multi-query expansion improves recall by searching for
        different phrasings and aspects of the original query.

        Args:
            query: The original query
            n: Number of expanded queries to generate

        Returns:
            List of expanded queries (including original)
        """
        try:
            prompt = get_prompt("rag.query_expand").format(
                query=query,
                n=str(n),
            )

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

            # Try to find JSON array in the response
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                json_str = content[start:end + 1]
                expanded = json.loads(json_str)
            else:
                # Fallback: split by newlines
                expanded = [line.strip() for line in content.split("\n") if line.strip()]

            # Ensure we have the original query included
            if query not in expanded:
                expanded = [query] + expanded[:n - 1]
            else:
                expanded = expanded[:n]

            logger.debug(
                "Expanded query",
                original=query[:50],
                expanded_count=len(expanded),
            )

            return expanded

        except Exception as e:
            logger.error(
                "Failed to expand query",
                error=str(e),
                query=query[:50],
            )
            # Return original query as fallback
            return [query]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def extract_keywords(
        self,
        query: str,
        n: int = 5,
    ) -> List[str]:
        """Extract key terms from a query.

        Useful for improving full-text search effectiveness.

        Args:
            query: The search query
            n: Number of keywords to extract

        Returns:
            List of extracted keywords
        """
        try:
            prompt = get_prompt("rag.keyword_extract").format(
                query=query,
                n=str(n),
            )

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                json_str = content[start:end + 1]
                keywords = json.loads(json_str)
            else:
                keywords = [query]

            logger.debug(
                "Extracted keywords",
                query=query[:50],
                keywords=keywords,
            )

            return keywords[:n]

        except Exception as e:
            logger.error(
                "Failed to extract keywords",
                error=str(e),
                query=query[:50],
            )
            return [query]


# Singleton instance
_query_transform_service: QueryTransformService | None = None


def get_query_transform_service() -> QueryTransformService:
    """Get or create query transform service instance."""
    global _query_transform_service
    if _query_transform_service is None:
        _query_transform_service = QueryTransformService()
    return _query_transform_service