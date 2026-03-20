"""Vector store operations using pgvector.

Supports both summary embeddings (title + description) and
chunk embeddings (sections of full_content) for RAG system.
"""

import uuid
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import DatabaseError
from app.models.orm_models import ArticleEmbedding, NewsArticle

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SearchResult:
    """Represents a search result from hybrid search."""

    article_id: uuid.UUID
    chunk_text: str
    similarity: float
    article_title: str
    article_summary: str
    source_name: str
    source_url: str
    chunk_number: int
    embedding_type: str


class VectorStore:
    """PostgreSQL pgvector operations for article embeddings.

    Supports:
    - Summary embeddings (title + description) for deduplication
    - Chunk embeddings (full_content sections) for RAG retrieval
    - Hybrid search combining vector similarity and full-text search
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    async def store_embedding(
        self,
        session: AsyncSession,
        article_id: uuid.UUID,
        embedding: List[float],
        content_hash: str,
    ) -> ArticleEmbedding:
        """Store summary embedding for an article (backward compatible).

        Args:
            session: Database session
            article_id: UUID of the article
            embedding: Embedding vector
            content_hash: Hash of the content

        Returns:
            Created ArticleEmbedding instance
        """
        try:
            # Delete any existing summary embedding for this article
            await session.execute(
                text(
                    "DELETE FROM article_embeddings WHERE article_id = :article_id AND embedding_type = 'summary'"
                ),
                {"article_id": str(article_id)},
            )

            db_embedding = ArticleEmbedding(
                article_id=article_id,
                embedding=embedding,
                content_hash=content_hash,
                embedding_type="summary",
                chunk_number=-1,  # Use -1 for summary to avoid conflict with chunk_number=0
            )
            session.add(db_embedding)
            await session.flush()

            logger.debug(
                "Stored summary embedding",
                article_id=str(article_id),
                content_hash=content_hash[:8],
            )

            return db_embedding

        except Exception as e:
            raise DatabaseError(
                f"Failed to store embedding: {str(e)}",
                {"article_id": str(article_id), "error": str(e)},
            )

    async def store_chunk_embeddings(
        self,
        session: AsyncSession,
        article_id: uuid.UUID,
        chunks: List[tuple[str, int, int, List[float]]],
        content_hash: str,
    ) -> List[ArticleEmbedding]:
        """Store chunk embeddings for an article.

        Args:
            session: Database session
            article_id: UUID of the article
            chunks: List of (chunk_text, start_char, end_char, embedding) tuples
            content_hash: Hash of the article content

        Returns:
            List of created ArticleEmbedding instances
        """
        try:
            # Delete existing chunk embeddings for this article
            await session.execute(
                text(
                    "DELETE FROM article_embeddings WHERE article_id = :article_id AND embedding_type = 'chunk'"
                ),
                {"article_id": str(article_id)},
            )
            # Flush the delete before inserting new records
            await session.flush()

            embeddings = []
            for i, (chunk_text, start_char, end_char, embedding) in enumerate(chunks):
                # Use truncated hash to fit in 64 chars: first 60 chars + _c + chunk number
                chunk_hash = f"{content_hash[:60]}_c{i}"
                db_embedding = ArticleEmbedding(
                    article_id=article_id,
                    embedding=embedding,
                    content_hash=chunk_hash,
                    embedding_type="chunk",
                    chunk_number=i,
                    chunk_text=chunk_text,
                    chunk_start=start_char,
                    chunk_end=end_char,
                )
                session.add(db_embedding)
                embeddings.append(db_embedding)

            await session.flush()

            logger.info(
                "Stored chunk embeddings",
                article_id=str(article_id),
                num_chunks=len(chunks),
            )

            return embeddings

        except Exception as e:
            raise DatabaseError(
                f"Failed to store chunk embeddings: {str(e)}",
                {"article_id": str(article_id), "error": str(e)},
            )

    async def find_similar(
        self,
        session: AsyncSession,
        embedding: List[float],
        limit: int = 10,
        exclude_ids: Optional[List[uuid.UUID]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[tuple[NewsArticle, float]]:
        """Find similar articles by embedding (using summary embeddings).

        Args:
            session: Database session
            embedding: Query embedding vector
            limit: Maximum number of results
            exclude_ids: Article IDs to exclude

        Returns:
            List of (article, similarity) tuples
        """
        try:
            # Convert embedding to string format for SQL
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Build query using cosine similarity on summary embeddings only
            query = text(
                f"""
                SELECT
                    na.id, na.source_name, na.source_url, na.original_title,
                    na.original_description, na.chinese_title, na.chinese_summary,
                    na.total_score, na.published_at, na.processed_at,
                    ae.embedding <=> CAST(:embedding AS vector) as distance
                FROM news_articles na
                JOIN article_embeddings ae ON na.id = ae.article_id
                WHERE ae.embedding_type = 'summary'
                AND 1 - (ae.embedding <=> CAST(:embedding AS vector)) >= :threshold
                ORDER BY distance
                LIMIT :limit
                """
            )

            threshold = similarity_threshold or self.similarity_threshold
            params = {
                "embedding": embedding_str,
                "threshold": threshold,
                "limit": limit,
            }

            if exclude_ids:
                # Add exclusion filter
                query = text(
                    query.text.replace(
                        "WHERE",
                        f"WHERE na.id NOT IN ({','.join([f"'{str(id)}'" for id in exclude_ids])}) AND "
                    )
                )

            result = await session.execute(query, params)
            rows = result.fetchall()

            articles = []
            for row in rows:
                article = NewsArticle(
                    id=row[0],
                    source_name=row[1],
                    source_url=row[2],
                    original_title=row[3],
                    original_description=row[4],
                    chinese_title=row[5],
                    chinese_summary=row[6],
                    total_score=row[7],
                    published_at=row[8],
                    processed_at=row[9],
                )
                similarity = 1 - row[10]  # Convert distance to similarity
                articles.append((article, similarity))

            logger.info(
                "Vector similarity search completed",
                limit=limit,
                threshold=threshold,
                results_count=len(articles),
            )

            return articles

        except Exception as e:
            raise DatabaseError(
                f"Failed to find similar articles: {str(e)}",
                {"error": str(e)},
            )

    async def hybrid_search(
        self,
        session: AsyncSession,
        query: str,
        embedding: List[float],
        limit: int = 10,
        vector_weight: float | None = None,
        fts_weight: float | None = None,
    ) -> List[SearchResult]:
        """Hybrid search combining vector similarity and full-text search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.

        Args:
            session: Database session
            query: Search query text
            embedding: Query embedding vector
            limit: Maximum number of results
            vector_weight: Weight for vector similarity (default from config)
            fts_weight: Weight for full-text search (default from config)

        Returns:
            List of SearchResult objects
        """
        try:
            vector_weight = vector_weight or settings.rag_vector_weight
            fts_weight = fts_weight or settings.rag_fts_weight

            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Combined hybrid search query with RRF fusion
            # We use a single query that combines vector and FTS scores
            query_sql = text(
                f"""
                WITH vector_results AS (
                    SELECT
                        ae.article_id,
                        ae.chunk_text,
                        ae.chunk_number,
                        ae.embedding_type,
                        1 - (ae.embedding <=> CAST(:embedding AS vector)) as vector_score,
                        ROW_NUMBER() OVER (ORDER BY ae.embedding <=> CAST(:embedding AS vector)) as vector_rank
                    FROM article_embeddings ae
                    WHERE ae.embedding_type = 'chunk'
                    AND ae.chunk_text IS NOT NULL
                    ORDER BY ae.embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit * 2
                ),
                fts_results AS (
                    SELECT
                        ae.article_id,
                        ae.chunk_text,
                        ae.chunk_number,
                        ae.embedding_type,
                        ts_rank(ae.chunk_tsv, plainto_tsquery('english', :query)) as fts_score,
                        ROW_NUMBER() OVER (ORDER BY ts_rank(ae.chunk_tsv, plainto_tsquery('english', :query)) DESC) as fts_rank
                    FROM article_embeddings ae
                    WHERE ae.embedding_type = 'chunk'
                    AND ae.chunk_text IS NOT NULL
                    AND ae.chunk_tsv @@ plainto_tsquery('english', :query)
                    ORDER BY fts_score DESC
                    LIMIT :limit * 2
                ),
                combined AS (
                    SELECT
                        COALESCE(v.article_id, f.article_id) as article_id,
                        COALESCE(v.chunk_text, f.chunk_text) as chunk_text,
                        COALESCE(v.chunk_number, f.chunk_number) as chunk_number,
                        COALESCE(v.embedding_type, f.embedding_type) as embedding_type,
                        COALESCE(v.vector_score, 0) as vector_score,
                        COALESCE(f.fts_score, 0) as fts_score,
                        COALESCE(v.vector_rank, 1000) as vector_rank,
                        COALESCE(f.fts_rank, 1000) as fts_rank
                    FROM vector_results v
                    FULL OUTER JOIN fts_results f ON v.article_id = f.article_id AND v.chunk_number = f.chunk_number
                )
                SELECT
                    c.article_id,
                    c.chunk_text,
                    c.chunk_number,
                    c.embedding_type,
                    c.vector_score,
                    c.fts_score,
                    na.original_title,
                    na.chinese_title,
                    na.original_description,
                    na.chinese_summary,
                    na.source_name,
                    na.source_url
                FROM combined c
                JOIN news_articles na ON c.article_id = na.id
                ORDER BY
                    :vector_weight / (c.vector_rank + 60) +
                    :fts_weight / (c.fts_rank + 60)
                DESC
                LIMIT :limit
                """
            )

            result = await session.execute(
                query_sql,
                {
                    "embedding": embedding_str,
                    "query": query,
                    "limit": limit,
                    "vector_weight": vector_weight,
                    "fts_weight": fts_weight,
                },
            )
            rows = result.fetchall()

            results = []
            for row in rows:
                # Combine original and Chinese titles/summaries
                title = row[7] or row[6]  # chinese_title or original_title
                summary = row[9] or row[8]  # chinese_summary or original_description

                results.append(SearchResult(
                    article_id=row[0],
                    chunk_text=row[1],
                    similarity=row[4],  # vector_score
                    article_title=title,
                    article_summary=summary or "",
                    source_name=row[10],
                    source_url=row[11],
                    chunk_number=row[2],
                    embedding_type=row[3],
                ))

            logger.info(
                "Hybrid search completed",
                query=query[:50],
                limit=limit,
                results_count=len(results),
            )

            return results

        except Exception as e:
            logger.error("Hybrid search failed", error=str(e), query=query[:50])
            raise DatabaseError(
                f"Failed to perform hybrid search: {str(e)}",
                {"error": str(e), "query": query[:100]},
            )

    async def vector_search_chunks(
        self,
        session: AsyncSession,
        embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Search chunks by vector similarity only.

        Args:
            session: Database session
            embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        try:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            query = text(
                f"""
                SELECT
                    ae.article_id,
                    ae.chunk_text,
                    ae.chunk_number,
                    ae.embedding_type,
                    1 - (ae.embedding <=> CAST(:embedding AS vector)) as similarity,
                    na.original_title,
                    na.chinese_title,
                    na.original_description,
                    na.chinese_summary,
                    na.source_name,
                    na.source_url
                FROM article_embeddings ae
                JOIN news_articles na ON ae.article_id = na.id
                WHERE ae.embedding_type = 'chunk'
                AND ae.chunk_text IS NOT NULL
                AND 1 - (ae.embedding <=> CAST(:embedding AS vector)) >= :threshold
                ORDER BY ae.embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
                """
            )

            result = await session.execute(
                query,
                {
                    "embedding": embedding_str,
                    "threshold": similarity_threshold,
                    "limit": limit,
                },
            )
            rows = result.fetchall()

            results = []
            for row in rows:
                title = row[6] or row[5]  # chinese_title or original_title
                summary = row[8] or row[7]  # chinese_summary or original_description

                results.append(SearchResult(
                    article_id=row[0],
                    chunk_text=row[1],
                    similarity=row[4],
                    article_title=title,
                    article_summary=summary or "",
                    source_name=row[9],
                    source_url=row[10],
                    chunk_number=row[2],
                    embedding_type=row[3],
                ))

            return results

        except Exception as e:
            raise DatabaseError(
                f"Failed to search chunks: {str(e)}",
                {"error": str(e)},
            )

    async def check_duplicate_by_hash(
        self,
        session: AsyncSession,
        content_hash: str,
    ) -> Optional[NewsArticle]:
        """Check for duplicate article by content hash.

        Args:
            session: Database session
            content_hash: Hash to check

        Returns:
            Existing article if found, None otherwise
        """
        try:
            stmt = (
                select(NewsArticle)
                .join(ArticleEmbedding)
                .where(
                    ArticleEmbedding.content_hash == content_hash,
                    ArticleEmbedding.embedding_type == "summary",
                )
                .limit(1)
            )

            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            raise DatabaseError(
                f"Failed to check duplicate: {str(e)}",
                {"content_hash": content_hash, "error": str(e)},
            )

    async def check_duplicate_by_url(
        self,
        session: AsyncSession,
        url: str,
    ) -> Optional[NewsArticle]:
        """Check for duplicate article by URL.

        Args:
            session: Database session
            url: URL to check

        Returns:
            Existing article if found, None otherwise
        """
        try:
            stmt = select(NewsArticle).where(NewsArticle.source_url == url).limit(1)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            raise DatabaseError(
                f"Failed to check duplicate URL: {str(e)}",
                {"url": url, "error": str(e)},
            )

    async def delete_embedding(
        self,
        session: AsyncSession,
        article_id: uuid.UUID,
    ) -> bool:
        """Delete all embeddings for an article.

        Args:
            session: Database session
            article_id: UUID of the article

        Returns:
            True if deleted, False if not found
        """
        try:
            result = await session.execute(
                text("DELETE FROM article_embeddings WHERE article_id = :article_id"),
                {"article_id": str(article_id)},
            )
            await session.flush()
            return result.rowcount > 0

        except Exception as e:
            raise DatabaseError(
                f"Failed to delete embedding: {str(e)}",
                {"article_id": str(article_id), "error": str(e)},
            )

    async def get_article_chunks(
        self,
        session: AsyncSession,
        article_id: uuid.UUID,
    ) -> List[ArticleEmbedding]:
        """Get all chunk embeddings for an article.

        Args:
            session: Database session
            article_id: UUID of the article

        Returns:
            List of ArticleEmbedding objects
        """
        try:
            stmt = (
                select(ArticleEmbedding)
                .where(
                    ArticleEmbedding.article_id == article_id,
                    ArticleEmbedding.embedding_type == "chunk",
                )
                .order_by(ArticleEmbedding.chunk_number)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            raise DatabaseError(
                f"Failed to get article chunks: {str(e)}",
                {"article_id": str(article_id), "error": str(e)},
            )


# Singleton instance
vector_store = VectorStore()