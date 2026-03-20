"""Database connection and session management."""

import uuid
from typing import AsyncGenerator

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models.orm_models import Base, RSSFeedSource
from app.utils.constants import DEFAULT_RSS_FEEDS

settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables and seed default RSS feeds."""
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    # Seed default RSS feeds
    await _seed_rss_feeds()


async def _seed_rss_feeds() -> None:
    """Seed default RSS feeds into database."""
    async with async_session_factory() as session:
        try:
            # Check if feeds already exist
            result = await session.execute(select(RSSFeedSource).limit(1))
            if result.scalar_one_or_none():
                return  # Already seeded

            # Insert default feeds
            for feed in DEFAULT_RSS_FEEDS:
                db_feed = RSSFeedSource(
                    name=feed.name,
                    url=feed.url,
                    is_active=True,
                )
                session.add(db_feed)

            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Backward-compatible alias for background tasks."""
    async for session in get_session():
        yield session
