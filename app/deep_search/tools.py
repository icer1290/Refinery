"""Tool implementations for deep search ReAct workflow."""

import json
from typing import Any, List

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.services.compression import get_compression_service
from app.services.embedding import get_embedding_service
from app.services.query_transform import get_query_transform_service
from app.services.reranker import get_reranker_service
from app.services.vector_store import vector_store, SearchResult
from app.services.web_search import get_web_search_service

logger = get_logger(__name__)
settings = get_settings()


class BaseTool:
    """Base class for tools."""

    name: str = "base_tool"
    description: str = "Base tool description"

    async def execute(self, session: AsyncSession, **kwargs) -> str:
        """Execute the tool.

        Args:
            session: Database session
            **kwargs: Tool arguments

        Returns:
            Tool output as string
        """
        raise NotImplementedError


class VectorSearchTool(BaseTool):
    """Tool for searching similar articles in vector store.

    Features:
    - Hybrid search: Combines vector similarity and full-text search
    - Reranking: Uses qwen3-rerank for improved relevance
    - Compression: Optional context compression for long results
    """

    name = "vector_search"
    description = "Search for similar articles in the local database. Use this to find related news and background information from previously processed articles."

    async def execute(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
        use_rerank: bool = True,
        use_compression: bool = False,
        use_hybrid: bool = True,
    ) -> str:
        """Execute vector search with optional reranking and compression.

        Args:
            session: Database session
            query: Search query
            limit: Maximum number of results
            use_rerank: Whether to apply reranking
            use_compression: Whether to compress results
            use_hybrid: Whether to use hybrid search (vector + FTS)

        Returns:
            Formatted search results
        """
        try:
            # Generate embedding for query
            embedding_service = get_embedding_service()
            embedding = await embedding_service.embed_text(query)

            # Determine search strategy
            if use_hybrid:
                # Hybrid search: vector + full-text search
                results = await vector_store.hybrid_search(
                    session=session,
                    query=query,
                    embedding=embedding,
                    limit=settings.rag_rerank_top_k,  # Get more candidates for reranking
                )
            else:
                # Vector-only search
                results = await vector_store.vector_search_chunks(
                    session=session,
                    embedding=embedding,
                    limit=settings.rag_rerank_top_k,
                )

            if not results:
                # Fallback to summary embeddings search
                logger.info(
                    "No chunk results, falling back to summary search",
                    query=query[:50],
                )
                articles = await vector_store.find_similar(
                    session=session,
                    embedding=embedding,
                    limit=limit,
                    similarity_threshold=0.6,
                )

                if not articles:
                    return "No similar articles found in database."

                # Convert to SearchResult format
                results = [
                    SearchResult(
                        article_id=article.id,
                        chunk_text=article.chinese_summary or article.original_description or "",
                        similarity=similarity,
                        article_title=article.chinese_title or article.original_title,
                        article_summary=article.chinese_summary or article.original_description or "",
                        source_name=article.source_name,
                        source_url=article.source_url,
                        chunk_number=0,
                        embedding_type="summary",
                    )
                    for article, similarity in articles
                ]

            # Apply reranking if enabled
            if use_rerank and len(results) > limit:
                reranker = get_reranker_service()
                ranked_results = await reranker.rerank(
                    query=query,
                    results=results,
                    top_k=limit,
                )
                results = [r for r, _ in ranked_results]

            # Apply compression if enabled
            if use_compression and results:
                compression_service = get_compression_service()
                compressed_context = await compression_service.compress_chunks(
                    query=query,
                    results=results[:limit],
                )
                return self._format_compressed_results(query, results[:limit], compressed_context)

            # Format standard results
            formatted_results = self._format_results(results[:limit])

            logger.info(
                "Vector search completed",
                query=query[:50],
                results_count=len(results[:limit]),
                use_rerank=use_rerank,
                use_hybrid=use_hybrid,
            )

            return formatted_results

        except Exception as e:
            logger.error("Vector search failed", error=str(e), query=query[:50])
            return f"Vector search failed: {str(e)}"

    def _format_results(self, results: List[SearchResult]) -> str:
        """Format search results as JSON."""
        formatted = []
        for r in results:
            formatted.append({
                "title": r.article_title,
                "content": r.chunk_text[:500] + "..." if len(r.chunk_text) > 500 else r.chunk_text,
                "source": r.source_name,
                "relevance": round(r.similarity, 3),
                "url": r.source_url,
                "chunk": r.chunk_number,
            })
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    def _format_compressed_results(
        self,
        query: str,
        results: List[SearchResult],
        compressed: str,
    ) -> str:
        """Format compressed results with sources."""
        sources = list(set(r.source_name for r in results))
        return json.dumps({
            "query": query,
            "compressed_context": compressed,
            "sources": sources,
            "total_chunks": len(results),
        }, ensure_ascii=False, indent=2)


class WebSearchTool(BaseTool):
    """Tool for searching the web for information."""

    name = "web_search"
    description = "Search the web for information about companies, technologies, and events. Use this to find external background information not available in the local database."

    async def execute(
        self,
        session: AsyncSession,
        query: str,
    ) -> str:
        """Execute web search.

        Args:
            session: Database session (unused but required for interface)
            query: Search query

        Returns:
            Formatted search results
        """
        try:
            web_search = get_web_search_service()
            results = await web_search.search(query, max_results=5)

            if not results:
                return "No results found on the web."

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "snippet": result.snippet,
                    "url": result.url,
                })

            logger.info(
                "Web search completed",
                query=query[:50],
                results_count=len(results),
            )

            return json.dumps(formatted_results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Web search failed", error=str(e), query=query[:50])
            return f"Web search failed: {str(e)}"


class QueryExpandTool(BaseTool):
    """Tool for expanding queries into multiple related queries."""

    name = "query_expand"
    description = "Expand a query into multiple related queries to improve search coverage. Use this when initial search results are insufficient."

    async def execute(
        self,
        session: AsyncSession,
        query: str,
        n: int = 3,
    ) -> str:
        """Expand query into multiple related queries.

        Args:
            session: Database session (unused)
            query: Original query
            n: Number of expanded queries

        Returns:
            JSON list of expanded queries
        """
        try:
            transform_service = get_query_transform_service()
            expanded = await transform_service.expand_query(query, n)
            return json.dumps(expanded, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Query expansion failed", error=str(e))
            return json.dumps([query], ensure_ascii=False)


# Tool registry
TOOLS: dict[str, BaseTool] = {
    "vector_search": VectorSearchTool(),
    "web_search": WebSearchTool(),
    "query_expand": QueryExpandTool(),
}


def get_tool(tool_name: str) -> BaseTool | None:
    """Get a tool by name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool instance or None if not found
    """
    return TOOLS.get(tool_name)


async def execute_tool(
    session: AsyncSession,
    tool_name: str,
    tool_input: dict[str, Any],
) -> str:
    """Execute a tool by name.

    Args:
        session: Database session
        tool_name: Name of the tool
        tool_input: Tool input arguments

    Returns:
        Tool output as string
    """
    tool = get_tool(tool_name)
    if not tool:
        return f"Unknown tool: {tool_name}"

    try:
        return await tool.execute(session, **tool_input)
    except Exception as e:
        logger.error("Tool execution failed", tool=tool_name, error=str(e))
        return f"Tool execution failed: {str(e)}"