"""API dependencies."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async for session in get_session():
        yield session