"""Query transformation service for RAG system.

Provides advanced query processing techniques:
- HyDE (Hypothetical Document Embedding): Generate a hypothetical document
  that would answer the query, then use it for retrieval.
- Multi-query expansion: Generate multiple related queries to improve recall.
"""

from typing import List

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core import get_logger

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
            prompt = f"""请生成一篇假设性的新闻文章，这篇文章能够完美回答以下问题。
文章应该包含详细的技术细节和背景信息。
目标长度约{doc_length}字。

问题: {query}

请直接输出文章内容，不要添加任何解释或标注:"""

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
            prompt = f"""你是一个搜索助手。请将以下查询扩展为{n}个相关但不同的搜索查询。
这些查询应该从不同角度探索原始问题的不同方面。
每个查询应该是一个独立的问题，可以单独用于搜索。

原始查询: {query}

请以JSON数组格式输出，例如:
["查询1", "查询2", "查询3"]

只输出JSON数组，不要添加其他内容:"""

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            # Parse JSON response
            import json
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
            prompt = f"""从以下查询中提取{n}个最重要的关键词。
关键词应该是技术术语、公司名称、产品名称或核心概念。

查询: {query}

请以JSON数组格式输出，例如:
["关键词1", "关键词2", "关键词3"]

只输出JSON数组:"""

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            import json
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