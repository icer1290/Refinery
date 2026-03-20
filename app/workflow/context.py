"""Workflow context for LangGraph dependency injection."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class WorkflowContext:
    """Context passed to workflow nodes during execution.

    Provides access to shared resources like database session.
    """

    session: AsyncSession
    run_id: str