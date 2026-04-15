"""Async database session factory for CLI commands."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def get_cli_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for CLI commands.

    This context manager handles session lifecycle including commit/rollback.
    Use in CLI commands that need database access:

        async with get_cli_session() as session:
            result = await session.execute(query)
            data = result.scalars().all()

    Yields:
        AsyncSession: Database session that auto-commits on success.

    Raises:
        Exception: Re-raises any exception after rollback.
    """
    # Create engine for CLI use (separate from web app engine)
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )

    # Create session factory
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            await engine.dispose()


async def check_database_connection() -> bool:
    """Check if database connection is working.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        engine = create_async_engine(settings.database_url, echo=False)
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False
