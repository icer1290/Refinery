"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from app.core import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class BaseAgent(ABC, Generic[T]):
    """Base class for all agents in the news aggregation system."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(agent=name)

    @abstractmethod
    async def execute(self, input_data: Any) -> T:
        """Execute the agent's main task.

        Args:
            input_data: Input data for the agent

        Returns:
            Result of the agent execution
        """
        pass

    async def run(self, input_data: Any) -> T:
        """Run the agent with error handling and logging.

        Args:
            input_data: Input data for the agent

        Returns:
            Result of the agent execution

        Raises:
            Exception: If agent execution fails
        """
        self.logger.info(f"Starting {self.name} agent")
        try:
            result = await self.execute(input_data)
            self.logger.info(f"Completed {self.name} agent")
            return result
        except Exception as e:
            self.logger.error(
                f"Failed {self.name} agent",
                error=str(e),
                input_type=type(input_data).__name__,
            )
            raise