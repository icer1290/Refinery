"""Researcher agent for conducting research using tools."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.agents.base import BaseChatAgent
from app.chat.state import SpecialistResponse, ToolCallRecord
from app.core import get_logger
from app.prompts import get_prompt
from app.services.llm_service import get_llm_service
from app.tools import execute_tool, get_available_tool_names

logger = get_logger(__name__)


class ResearcherAgent(BaseChatAgent):
    """Agent for conducting research using external tools.

    Handles:
    - External information search
    - Background research on topics
    - Finding related articles
    - Web search for additional context

    Available tools:
    - vector_search: Search local article database
    - web_search: Search external web sources
    - article_lookup: Fetch specific article details
    """

    name = "researcher"
    description = "Conducts research using web and vector search tools"
    available_tools = ["vector_search", "web_search", "article_lookup"]

    def get_system_prompt(self) -> str:
        """Get researcher-specific system prompt from centralized registry."""
        return get_prompt("chat.researcher_system").template

    async def execute(
        self,
        query: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        session: Optional[AsyncSession] = None,
        user_profile: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> SpecialistResponse:
        """Execute research with tool calls.

        Args:
            query: User's question
            article_context: Article details
            conversation_history: Recent messages
            session: Database session for tool execution
            user_profile: User preferences

        Returns:
            SpecialistResponse with research findings
        """
        llm_service = get_llm_service()

        # Format context
        context = self.format_context(article_context, conversation_history)

        # Determine tool strategy
        tool_strategy = await self._plan_tool_usage(query, llm_service)

        # Execute tools if session available
        tool_calls: list[ToolCallRecord] = []
        tool_results: list[str] = []

        if session and tool_strategy.get("use_tools"):
            for tool_call in tool_strategy.get("tool_calls", []):
                result = await self._execute_tool(
                    session, tool_call["tool_name"], tool_call["tool_input"]
                )
                if result:
                    tool_results.append(result)
                    tool_calls.append(tool_call)

        # Build research prompt with tool results
        research_prompt = self._build_research_prompt(
            query, context, tool_results, user_profile
        )

        try:
            # Generate research response
            response = await llm_service.chat_completion(
                system_prompt=self.get_system_prompt(),
                user_prompt=research_prompt,
                temperature=0.4,
                max_tokens=800,
            )

            # Extract citations
            citations = self._extract_citations(response, article_context, tool_results)

            # Estimate tokens
            tokens_used = len(response) // 4 + len(research_prompt) // 4

            # Calculate confidence based on tool results
            confidence = 0.9 if tool_results else 0.6

            logger.info(
                "Researcher response generated",
                query=query[:50],
                tools_used=len(tool_calls),
                tokens_used=tokens_used,
            )

            return SpecialistResponse(
                agent_name=self.name,
                response=response,
                citations=citations,
                tool_calls=tool_calls,
                confidence=confidence,
            )

        except Exception as e:
            logger.error("Researcher failed", error=str(e), query=query[:50])
            return SpecialistResponse(
                agent_name=self.name,
                response=f"研究过程中遇到问题。错误：{str(e)}",
                citations=[],
                tool_calls=tool_calls,
                confidence=0.0,
            )

    async def _plan_tool_usage(
        self,
        query: str,
        llm_service: Any,
    ) -> dict[str, Any]:
        """Plan which tools to use for the query.

        Args:
            query: User's question
            llm_service: LLM service

        Returns:
            Dict with tool usage plan
        """
        # Simple heuristic-based planning for now
        # Can be enhanced with LLM-based planning

        plan = {"use_tools": False, "tool_calls": []}

        # Check for keywords indicating search need
        search_keywords = ["查找", "搜索", "寻找", "相关", "其他文章", "更多信息", "背景"]
        web_keywords = ["最新", "最近", "当前", "新闻", "进展", "动态"]

        query_lower = query.lower()

        # Vector search for related articles
        if any(kw in query_lower for kw in search_keywords):
            # Extract search query from user message
            search_query = self._extract_search_query(query)
            plan["tool_calls"].append({
                "tool_name": "vector_search",
                "tool_input": {"query": search_query, "limit": 3},
            })
            plan["use_tools"] = True

        # Web search for latest info
        if any(kw in query_lower for kw in web_keywords):
            web_query = self._extract_web_search_query(query, search_keywords + web_keywords)
            plan["tool_calls"].append({
                "tool_name": "web_search",
                "tool_input": {"query": web_query},
            })
            plan["use_tools"] = True

        return plan

    def _extract_search_query(self, query: str) -> str:
        """Extract search query from user message.

        Args:
            query: User's message

        Returns:
            Cleaned search query
        """
        # Remove common question words
        words_to_remove = [
            "你能", "请", "查找", "搜索", "寻找",
            "关于的文章", "关于的信息", "告诉我",
        ]
        cleaned = query.lower()
        for word in words_to_remove:
            cleaned = cleaned.replace(word, "")

        # Take remaining meaningful words
        words = cleaned.split()
        # Keep topic-related words
        meaningful = [w for w in words if len(w) > 1 and w not in ["的", "关于", "那个", "这个"]]
        return " ".join(meaningful[:5]) if meaningful else query

    def _extract_web_search_query(self, query: str, keywords: list[str]) -> str:
        """Extract web search query.

        Args:
            query: User's message
            keywords: Keywords to remove

        Returns:
            Search query for web
        """
        cleaned = query.lower()
        for kw in keywords:
            cleaned = cleaned.replace(kw, "")
        words = cleaned.split()
        meaningful = [w for w in words if len(w) > 1]
        return " ".join(meaningful[:4]) if meaningful else query

    async def _execute_tool(
        self,
        session: AsyncSession,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> Optional[str]:
        """Execute a single tool call.

        Args:
            session: Database session
            tool_name: Tool to execute
            tool_input: Input parameters

        Returns:
            Tool output string or None
        """
        try:
            # Check if tool is available
            if tool_name not in get_available_tool_names():
                logger.warning("Tool not available", tool_name=tool_name)
                return None

            # Execute tool
            result = await execute_tool(session, tool_name, tool_input)
            logger.debug("Tool executed", tool_name=tool_name, result_length=len(result))
            return result

        except Exception as e:
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            return f"工具错误：{str(e)}"

    def _build_research_prompt(
        self,
        query: str,
        context: str,
        tool_results: list[str],
        user_profile: Optional[dict[str, Any]],
    ) -> str:
        """Build the research prompt with tool results using centralized template.

        Args:
            query: User's question
            context: Article and conversation context
            tool_results: Results from tool calls
            user_profile: User preferences

        Returns:
            Formatted prompt
        """
        # Build tool results section
        if tool_results:
            results_text = ""
            for i, result in enumerate(tool_results, 1):
                # Truncate long results
                truncated = result[:500] + "..." if len(result) > 500 else result
                results_text += f"\n--- 结果 {i} ---\n{truncated}\n"

            tool_results_section = get_prompt("chat.researcher_tool_results").format(
                tool_results=results_text
            )
        else:
            tool_results_section = get_prompt("chat.researcher_no_tools").template

        prompt = get_prompt("chat.researcher_user")
        return prompt.format(
            context=context,
            query=query,
            tool_results_section=tool_results_section,
        )

    def _extract_citations(
        self,
        response: str,
        article_context: dict[str, Any],
        tool_results: list[str],
    ) -> list[dict[str, Any]]:
        """Extract citations from response and tool results.

        Args:
            response: Generated response
            article_context: Article details
            tool_results: Tool outputs

        Returns:
            List of citation dicts
        """
        import json
        import re

        citations = []

        # Extract from response text (Chinese pattern)
        article_pattern = r"\[文章：\s*([^\]]+)\]"
        web_pattern = r"\[网络：\s*([^\]]+)\]"

        article_refs = re.findall(article_pattern, response)
        for ref in article_refs:
            citations.append({
                "source_type": "article",
                "source_name": ref.strip(),
                "content_snippet": "",
            })

        web_refs = re.findall(web_pattern, response)
        for ref in web_refs:
            citations.append({
                "source_type": "web_search",
                "source_name": ref.strip(),
                "content_snippet": "",
            })

        # Try to extract from tool results JSON
        for result in tool_results:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    for item in parsed:
                        if "title" in item:
                            citations.append({
                                "source_type": "web_search",
                                "source_name": item.get("title", ""),
                                "url": item.get("url", ""),
                                "content_snippet": item.get("snippet", "")[:100],
                            })
            except json.JSONDecodeError:
                pass

        return citations