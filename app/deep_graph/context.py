"""DeepGraph context for LangGraph dependency injection."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class DeepGraphBuilderContext:
    """Context passed to GraphBuilder nodes during execution.

    Provides access to shared resources like database session.
    """

    session: AsyncSession
    run_id: str
    article_ids: list[str]


@dataclass
class DeepGraphAnalystContext:
    """Context passed to DeepGraph Analyst nodes during execution.

    Provides access to shared resources like database session.
    """

    session: AsyncSession
    article_ids: list[str]
    max_hops: int
    expansion_limit: int