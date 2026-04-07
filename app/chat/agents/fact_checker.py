"""Fact-checker agent for verifying claims."""

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


class FactCheckerAgent(BaseChatAgent):
    """Agent for verifying claims and checking facts.

    Handles:
    - Verification of claims in the article
    - Cross-checking against external sources
    - Evidence validation
    - "Is this true?" type queries

    Available tools:
    - web_search: Search for verification sources
    - vector_search: Search for related articles with similar claims
    """

    name = "fact_checker"
    description = "Verifies claims and checks facts against sources"
    available_tools = ["web_search", "vector_search"]

    def get_system_prompt(self) -> str:
        """Get fact-checker specific system prompt from centralized registry."""
        return get_prompt("chat.fact_checker_system").template

    async def execute(
        self,
        query: str,
        article_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        session: Optional[AsyncSession] = None,
        user_profile: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> SpecialistResponse:
        """Execute fact-checking with tool calls.

        Args:
            query: User's verification request
            article_context: Article details
            conversation_history: Recent messages
            session: Database session
            user_profile: User preferences

        Returns:
            SpecialistResponse with verification results
        """
        llm_service = get_llm_service()

        # Format context
        context = self.format_context(article_context, conversation_history)

        # Identify claim to verify
        claim = await self._extract_claim(query, context, llm_service)

        # Execute tools for verification
        tool_calls: list[ToolCallRecord] = []
        tool_results: list[str] = []

        if session and claim:
            # Search for verification sources
            verification_result = await self._execute_tool(
                session, "web_search", {"query": f"验证：{claim[:100]}"}
            )
            if verification_result:
                tool_results.append(verification_result)
                tool_calls.append(ToolCallRecord(
                    tool_name="web_search",
                    tool_input={"query": f"验证：{claim[:100]}"},
                    tool_output=verification_result,
                    timestamp=datetime.now().isoformat(),
                    agent=self.name,
                ))

            # Search for related articles
            related_result = await self._execute_tool(
                session, "vector_search", {"query": claim, "limit": 3}
            )
            if related_result:
                tool_results.append(related_result)
                tool_calls.append(ToolCallRecord(
                    tool_name="vector_search",
                    tool_input={"query": claim, "limit": 3},
                    tool_output=related_result,
                    timestamp=datetime.now().isoformat(),
                    agent=self.name,
                ))

        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            query, claim, context, tool_results
        )

        try:
            # Generate verification response
            response = await llm_service.chat_completion(
                system_prompt=self.get_system_prompt(),
                user_prompt=verification_prompt,
                temperature=0.3,  # Lower for factual accuracy
                max_tokens=600,
            )

            # Extract citations and sources
            citations = self._extract_citations(response, tool_results)

            # Estimate confidence based on verification results
            confidence = self._assess_confidence(response, tool_results)

            tokens_used = len(response) // 4 + len(verification_prompt) // 4

            logger.info(
                "Fact-checker response generated",
                claim=claim[:50] if claim else "none",
                tools_used=len(tool_calls),
                confidence=confidence,
            )

            return SpecialistResponse(
                agent_name=self.name,
                response=response,
                citations=citations,
                tool_calls=tool_calls,
                confidence=confidence,
            )

        except Exception as e:
            logger.error("Fact-checker failed", error=str(e), query=query[:50])
            return SpecialistResponse(
                agent_name=self.name,
                response=f"无法完成事实核查。错误：{str(e)}",
                citations=[],
                tool_calls=tool_calls,
                confidence=0.0,
            )

    async def _extract_claim(
        self,
        query: str,
        context: str,
        llm_service: Any,
    ) -> Optional[str]:
        """Extract the claim to verify from the query using centralized prompt.

        Args:
            query: User's message
            context: Article context
            llm_service: LLM service

        Returns:
            Extracted claim string or None
        """
        prompt = get_prompt("chat.claim_extraction")
        extract_prompt = prompt.format(
            context=context[:500],
            query=query,
        )

        try:
            response = await llm_service.chat_completion(
                system_prompt=get_prompt("chat.claim_extraction_system").template,
                user_prompt=extract_prompt,
                temperature=0.2,
                max_tokens=200,
            )
            return response.strip()
        except Exception as e:
            logger.warning("Claim extraction failed", error=str(e))
            return None

    async def _execute_tool(
        self,
        session: AsyncSession,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> Optional[str]:
        """Execute a single tool call."""
        try:
            if tool_name not in get_available_tool_names():
                logger.warning("Tool not available", tool_name=tool_name)
                return None

            result = await execute_tool(session, tool_name, tool_input)
            return result
        except Exception as e:
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            return None

    def _build_verification_prompt(
        self,
        query: str,
        claim: Optional[str],
        context: str,
        tool_results: list[str],
    ) -> str:
        """Build the verification prompt using centralized template."""
        # Build tool results section
        if tool_results:
            results_text = ""
            for i, result in enumerate(tool_results, 1):
                truncated = result[:400] + "..." if len(result) > 400 else result
                results_text += f"\n--- 来源 {i} ---\n{truncated}\n"

            tool_results_section = get_prompt("chat.fact_checker_tool_results").format(
                tool_results=results_text
            )
        else:
            tool_results_section = get_prompt("chat.fact_checker_no_tools").template

        prompt = get_prompt("chat.fact_checker_user")
        return prompt.format(
            context=context,
            query=query,
            claim=claim if claim else "请求一般性验证",
            tool_results_section=tool_results_section,
        )

    def _extract_citations(
        self,
        response: str,
        tool_results: list[str],
    ) -> list[dict[str, Any]]:
        """Extract citations from response and tool results."""
        import json
        import re

        citations = []

        # Extract explicit source mentions (Chinese pattern)
        source_pattern = r"\[来源：\s*([^\]]+)\]"
        source_refs = re.findall(source_pattern, response)
        for ref in source_refs:
            citations.append({
                "source_type": "verification",
                "source_name": ref.strip(),
                "content_snippet": "",
            })

        # Parse tool results for structured sources
        for result in tool_results:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    for item in parsed:
                        if "title" in item or "source" in item:
                            citations.append({
                                "source_type": "verification",
                                "source_name": item.get("title", item.get("source", "")),
                                "url": item.get("url", ""),
                                "content_snippet": item.get("snippet", "")[:50],
                            })
            except json.JSONDecodeError:
                pass

        return citations

    def _assess_confidence(
        self,
        response: str,
        tool_results: list[str],
    ) -> float:
        """Assess confidence level based on verification results.

        Args:
            response: Generated verification response
            tool_results: Tool outputs

        Returns:
            Confidence score (0.0-1.0)
        """
        # Check verification status in response
        if "已验证" in response:
            if "被反驳" in response:
                return 0.3  # Mixed results
            return 0.85  # Confirmed
        elif "部分验证" in response:
            return 0.6
        elif "未验证" in response:
            return 0.4

        # Fallback based on tool results availability
        if tool_results:
            return 0.65  # Some evidence available
        return 0.35  # Limited evidence