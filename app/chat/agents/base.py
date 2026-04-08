"""Base agent class for chat specialists."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.chat.state import SpecialistResponse
from app.core import get_logger
from app.prompts import get_prompt

logger = get_logger(__name__)


class BaseChatAgent(ABC):
    """Base class for chat specialist agents.

    All specialist agents (researcher, explainer, fact_checker)
    inherit from this base class.
    """

    name: str = "base_agent"
    description: str = "Base agent description"
    available_tools: list[str] = []

    @abstractmethod
    async def execute(
        self,
        query: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        user_profile: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> SpecialistResponse:
        """Execute the agent's primary function.

        Args:
            query: User's question or request
            article_context: Article details (title, summary, content)
            conversation_history: Recent conversation messages
            user_profile: Extracted user preferences
            **kwargs: Additional context

        Returns:
            SpecialistResponse with response text, citations, tool calls
        """
        pass

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            System prompt text
        """
        return f"你是{self.name}代理。{self.description}"

    def format_context(
        self,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
    ) -> str:
        """Format context for LLM input using centralized prompt.

        Args:
            article_context: Article details
            conversation_history: Recent messages

        Returns:
            Formatted context string
        """
        # Format article content (full content for better context)
        content = article_context.get('content', '')
        if content:
            truncated = content
        else:
            truncated = "无内容"

        # Format deepsearch section
        deepsearch_section = ""
        if article_context.get('deepsearch_report'):
            report = article_context['deepsearch_report']
            deepsearch_prompt = get_prompt("chat.deepsearch_section")
            deepsearch_section = deepsearch_prompt.format(deepsearch_report=report)

        # Format conversation history
        history_lines = []
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history_lines.append(f"{role}: {content}")
        history = "\n".join(history_lines) if history_lines else "无历史记录"

        # Use centralized context format template
        context_prompt = get_prompt("chat.context_format")
        return context_prompt.format(
            title=article_context.get('title', '未知'),
            source_name=article_context.get('source_name', '未知'),
            summary=article_context.get('summary', '无摘要'),
            published_at=article_context.get('published_at', '未知'),
            content=truncated,
            deepsearch_section=deepsearch_section,
            history=history,
        )