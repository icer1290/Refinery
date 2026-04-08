"""Supervisor agent for routing user queries to specialists."""

from typing import Any, Optional

from app.chat.agents.base import BaseChatAgent
from app.chat.state import RoutingDecision
from app.core import get_logger
from app.prompts import get_prompt
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)


class SupervisorAgent(BaseChatAgent):
    """Router agent that dispatches queries to specialists.

    Analyzes user queries and routes them to the appropriate specialist:
    - researcher: External information search, background research
    - explainer: Concept explanations, article clarifications
    - fact_checker: Verification requests, claims checking
    """

    name = "supervisor"
    description = "Routes user queries to the appropriate specialist agent"
    available_tools = []  # Supervisor doesn't use tools directly

    async def route(
        self,
        user_message: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        user_profile: Optional[dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Analyze query and determine routing.

        Args:
            user_message: User's question
            article_context: Article details
            conversation_history: Recent messages
            user_profile: User preferences

        Returns:
            RoutingDecision with agent assignment and reasoning
        """
        llm_service = get_llm_service()

        # Build routing prompt
        routing_prompt = self._build_routing_prompt(
            user_message, article_context, conversation_history, user_profile
        )

        try:
            # Use LLM to classify query
            response = await llm_service.chat_completion(
                system_prompt=get_prompt("chat.routing_system").template,
                user_prompt=routing_prompt,
                temperature=0.3,
                max_tokens=200,
            )

            # Parse response
            decision = self._parse_routing_response(response)
            logger.info(
                "Routing decision made",
                agent=decision["agent"],
                query_type=decision["query_type"],
            )
            return decision

        except Exception as e:
            logger.warning("Routing failed, defaulting to explainer", error=str(e))
            # Default to explainer on errors
            return RoutingDecision(
                agent="explainer",
                reasoning=f"默认路由，因错误：{str(e)}",
                query_type="general",
            )

    def _build_routing_prompt(
        self,
        user_message: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        user_profile: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build the routing classification prompt using centralized template."""
        # Article info
        article_title = article_context.get("title", "未知文章")
        article_summary = article_context.get("summary", "")

        # Conversation context
        last_messages = ""
        for msg in conversation_history[-3:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            last_messages += f"{role}: {content}\n"

        # User profile hints
        profile_hint = ""
        if user_profile:
            interests = user_profile.get("interests", [])
            if interests:
                profile_hint = f"用户兴趣：{', '.join(interests[:3])}\n"

        # Use centralized routing user prompt
        prompt = get_prompt("chat.routing_user")
        return prompt.format(
            article_title=article_title,
            article_summary=article_summary,
            last_messages=last_messages,
            profile_hint=profile_hint,
            user_message=user_message,
        )

    def _parse_routing_response(self, response: str) -> RoutingDecision:
        """Parse LLM routing response into RoutingDecision."""
        import json
        import re

        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r"```json\s*|\s*```", "", response.strip())
            parsed = json.loads(cleaned)

            # Validate agent
            agent = parsed.get("agent", "explainer")
            if agent not in ["researcher", "explainer", "fact_checker"]:
                agent = "explainer"

            return RoutingDecision(
                agent=agent,
                reasoning=parsed.get("reasoning", ""),
                query_type=parsed.get("query_type", "general"),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback parsing
            if "research" in response.lower() or "search" in response.lower():
                agent = "researcher"
            elif "verify" in response.lower() or "check" in response.lower():
                agent = "fact_checker"
            else:
                agent = "explainer"

            return RoutingDecision(
                agent=agent,
                reasoning="备用解析",
                query_type="general",
            )

    async def execute(
        self,
        query: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        user_profile: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Supervisor doesn't execute directly, only routes."""
        raise NotImplementedError("Supervisor only routes, use route() method")