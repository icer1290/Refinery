#!/usr/bin/env python3
"""Backfill script for chunk embeddings.

This script processes existing articles that have full_content but no
chunk embeddings, generating and storing their chunk embeddings.

Usage:
    python scripts/backfill_embeddings.py [--batch-size N] [--dry-run]

Options:
    --batch-size N    Number of articles to process per batch (default: 10)
    --dry-run         Show what would be done without making changes
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.core import get_logger
from app.models.orm_models import NewsArticle, ArticleEmbedding
from app.services.chunking import get_chunking_service
from app.services.embedding import get_embedding_service
from app.services.vector_store import vector_store

logger = get_logger(__name__)
settings = get_settings()


async def get_articles_without_chunks(
    session: AsyncSession,
    limit: int = 100,
    offset: int = 0,
) -> list[NewsArticle]:
    """Get articles with full_content but no chunk embeddings.

    Args:
        session: Database session
        limit: Maximum number of articles to fetch
        offset: Offset for pagination

    Returns:
        List of articles without chunk embeddings
    """
    # Subquery to find articles that have chunk embeddings
    chunk_subquery = (
        select(ArticleEmbedding.article_id)
        .where(ArticleEmbedding.embedding_type == "chunk")
        .distinct()
    )

    # Main query: articles with full_content but no chunks
    query = (
        select(NewsArticle)
        .where(
            NewsArticle.full_content.isnot(None),
            NewsArticle.full_content != "",
            NewsArticle.id.not_in(chunk_subquery),
        )
        .order_by(NewsArticle.processed_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await session.execute(query)
    return list(result.scalars().all())


async def count_articles_without_chunks(session: AsyncSession) -> int:
    """Count articles with full_content but no chunk embeddings."""
    chunk_subquery = (
        select(ArticleEmbedding.article_id)
        .where(ArticleEmbedding.embedding_type == "chunk")
        .distinct()
    )

    query = (
        select(func.count(NewsArticle.id))
        .where(
            NewsArticle.full_content.isnot(None),
            NewsArticle.full_content != "",
            NewsArticle.id.not_in(chunk_subquery),
        )
    )

    result = await session.execute(query)
    return result.scalar() or 0


async def process_article(
    session: AsyncSession,
    article: NewsArticle,
    dry_run: bool = False,
) -> dict:
    """Process a single article: chunk and create embeddings.

    Args:
        session: Database session
        article: Article to process
        dry_run: If True, don't actually store embeddings

    Returns:
        Dict with processing results
    """
    result = {
        "article_id": str(article.id),
        "title": article.chinese_title or article.original_title,
        "content_length": len(article.full_content) if article.full_content else 0,
        "chunks_created": 0,
        "status": "pending",
        "error": None,
    }

    try:
        if not article.full_content or len(article.full_content.strip()) < 100:
            result["status"] = "skipped"
            result["error"] = "Content too short"
            return result

        chunking_service = get_chunking_service()
        embedding_service = get_embedding_service()

        # Get Chinese summary for summary-first strategy
        summary = article.chinese_summary or ""

        # Chunk the content
        chunks = chunking_service.chunk_text_with_summary_first(
            text=article.full_content,
            summary=summary,
        )

        if not chunks:
            result["status"] = "skipped"
            result["error"] = "No chunks generated"
            return result

        if dry_run:
            result["chunks_created"] = len(chunks)
            result["status"] = "dry_run"
            return result

        # Generate embeddings for all chunks
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = await embedding_service.embed_batch(chunk_texts)

        # Prepare chunk data
        chunk_data = [
            (c.text, c.start_char, c.end_char, emb)
            for c, emb in zip(chunks, chunk_embeddings)
        ]

        # Store chunk embeddings
        content_hash = hashlib.sha256(article.full_content.encode()).hexdigest()
        await vector_store.store_chunk_embeddings(
            session=session,
            article_id=article.id,
            chunks=chunk_data,
            content_hash=content_hash,
        )

        result["chunks_created"] = len(chunks)
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(
            "Failed to process article",
            article_id=str(article.id),
            error=str(e),
        )

    return result


async def run_backfill(
    batch_size: int = 10,
    dry_run: bool = False,
) -> None:
    """Run the backfill process.

    Args:
        batch_size: Number of articles to process per batch
        dry_run: If True, don't actually store embeddings
    """
    # Create async engine
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )

    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        # Count total articles to process
        total_count = await count_articles_without_chunks(session)

        if total_count == 0:
            logger.info("No articles need chunk embeddings")
            print("✅ All articles already have chunk embeddings!")
            return

        print(f"\n📊 Found {total_count} articles needing chunk embeddings")
        if dry_run:
            print("🔍 DRY RUN - No changes will be made\n")
        else:
            print(f"📦 Processing in batches of {batch_size}\n")

        processed = 0
        success_count = 0
        error_count = 0
        skipped_count = 0

        offset = 0
        while offset < total_count:
            # Fetch batch of articles
            articles = await get_articles_without_chunks(
                session,
                limit=batch_size,
                offset=offset,
            )

            if not articles:
                break

            # Process each article
            for article in articles:
                result = await process_article(session, article, dry_run)

                processed += 1

                if result["status"] == "success":
                    success_count += 1
                    print(f"✅ [{processed}/{total_count}] {result['title'][:50]}... ({result['chunks_created']} chunks)")
                elif result["status"] == "dry_run":
                    print(f"🔍 [{processed}/{total_count}] Would create {result['chunks_created']} chunks for: {result['title'][:50]}...")
                elif result["status"] == "skipped":
                    skipped_count += 1
                    print(f"⏭️  [{processed}/{total_count}] Skipped: {result['title'][:50]}... ({result['error']})")
                else:
                    error_count += 1
                    print(f"❌ [{processed}/{total_count}] Error: {result['title'][:50]}... ({result['error']})")

                # Commit after each successful article
                if not dry_run and result["status"] == "success":
                    await session.commit()

            offset += batch_size

            # Small delay between batches
            await asyncio.sleep(0.5)

        # Summary
        print(f"\n{'='*60}")
        print("📈 Backfill Summary:")
        print(f"   Total processed: {processed}")
        if not dry_run:
            print(f"   ✅ Success: {success_count}")
            print(f"   ⏭️  Skipped: {skipped_count}")
            print(f"   ❌ Errors: {error_count}")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill chunk embeddings for existing articles"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of articles to process per batch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    print("\n🚀 Starting chunk embeddings backfill...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Dry run: {args.dry_run}")

    asyncio.run(run_backfill(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()