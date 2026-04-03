"""Base class for tools."""

from sqlalchemy.ext.asyncio import AsyncSession


class BaseTool:
    """Base class for tools.

    All tools must inherit from this class and implement the execute method.
    Tools are registered in the registry via register_tool() function.
    """

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