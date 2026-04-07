"""Explainer agent for concept explanations and article clarifications."""

from typing import Any, Optional

from app.chat.agents.base import BaseChatAgent
from app.chat.state import SpecialistResponse, ToolCallRecord
from app.core import get_logger
from app.prompts import get_prompt
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)


class ExplainerAgent(BaseChatAgent):
    """Agent for explaining concepts and clarifying article content.

    Handles:
    - Concept explanations within article context
    - "What does this mean?" queries
    - Article content clarifications
    - General questions about the article
    """

    name = "explainer"
    description = "Explains concepts and clarifies article content"
    available_tools = []  # Explainer primarily uses article context

    def get_system_prompt(self) -> str:
        """Get explainer-specific system prompt from centralized registry."""
        return get_prompt("chat.explainer_system").template

    async def execute(
        self,
        query: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        user_profile: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> SpecialistResponse:
        """Generate an explanation response.

        Args:
            query: User's question
            article_context: Article details
            conversation_history: Recent messages
            user_profile: User preferences (expertise level)

        Returns:
            SpecialistResponse with explanation
        """
        llm_service = get_llm_service()

        # Format context
        context = self.format_context(article_context, conversation_history)

        # Adjust based on user profile
        expertise = "intermediate"
        if user_profile:
            expertise = user_profile.get("expertise_level", "intermediate")

        # Build prompt with expertise adjustment
        user_prompt = self._build_explanation_prompt(query, context, expertise)

        try:
            # Generate explanation
            response = await llm_service.chat_completion(
                system_prompt=self.get_system_prompt(),
                user_prompt=user_prompt,
                temperature=0.5,  # Slightly higher for natural explanations
                max_tokens=500,
            )

            # Extract citations from response
            citations = self._extract_citations(response, article_context)

            # Estimate tokens
            tokens_used = len(response) // 4 + len(user_prompt) // 4

            logger.info(
                "Explainer response generated",
                query=query[:50],
                tokens_used=tokens_used,
            )

            return SpecialistResponse(
                agent_name=self.name,
                response=response,
                citations=citations,
                tool_calls=[],  # Explainer doesn't use tools
                confidence=0.8,  # High confidence for context-based explanations
            )

        except Exception as e:
            logger.error("Explainer failed", error=str(e), query=query[:50])
            return SpecialistResponse(
                agent_name=self.name,
                response=f"抱歉，我无法处理您的问题。错误：{str(e)}",
                citations=[],
                tool_calls=[],
                confidence=0.0,
            )

    def _build_explanation_prompt(
        self,
        query: str,
        context: str,
        expertise: str,
    ) -> str:
        """Build the explanation prompt using centralized template.

        Args:
            query: User's question
            context: Formatted context
            expertise: User's expertise level

        Returns:
            Formatted prompt
        """
        # Adjust detail level based on expertise
        detail_instruction = ""
        if expertise == "beginner":
            detail_instruction = "请用简单术语解释，避免专业术语，如有帮助可使用类比。"
        elif expertise == "advanced":
            detail_instruction = "请提供详细的技术解释，包含相关细节。"

        prompt = get_prompt("chat.explainer_user")
        return prompt.format(
            context=context,
            expertise=expertise,
            detail_instruction=detail_instruction,
            query=query,
        )

    def _extract_citations(
        self,
        response: str,
        article_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract citations from response.

        Args:
            response: Generated response
            article_context: Article details

        Returns:
            List of citation dicts
        """
        import re

        citations = []

        # Look for citation patterns [文章：标题] or [来源：名称]
        article_pattern = r"\[文章：\s*([^\]]+)\]"
        source_pattern = r"\[来源：\s*([^\]]+)\]"

        article_refs = re.findall(article_pattern, response)
        for ref in article_refs:
            citations.append({
                "source_type": "article",
                "source_name": ref.strip(),
                "content_snippet": article_context.get("title", ""),
            })

        source_refs = re.findall(source_pattern, response)
        for ref in source_refs:
            citations.append({
                "source_type": "source",
                "source_name": ref.strip(),
                "content_snippet": article_context.get("source_name", ""),
            })

        return citations