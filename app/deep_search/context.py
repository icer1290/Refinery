"""Deep search context for LangGraph dependency injection."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class DeepSearchContext:
    """Context passed to deep search nodes during execution.

    Provides access to shared resources like database session.
    """

    session: AsyncSession
    article_id: str