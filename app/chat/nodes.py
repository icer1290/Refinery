"""Node implementations for the chat workflow using LangGraph.

Hub-and-Spoke Architecture:
- Supervisor: Central hub that evaluates state and routes to agents
- Researcher: ReAct loop for collecting information
- Explainer: Generates response based on collected info
- FactChecker: Validates response for hallucinations
"""

import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langgraph.runtime import Runtime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.context import ChatContext
from app.chat.state import (
    ChatState,
    CollectedInfo,
    FactCheckResult,
    ResearchThought,
    create_user_message,
    create_assistant_message,
    estimate_tokens,
)
from app.config import get_settings
from app.core import get_logger
from app.models.orm_models import NewsArticle, ChatConversation, ChatMessage
from app.prompts import get_prompt
from app.services.chat_cache import get_chat_cache_service
from app.services.llm_service import get_llm_service
from app.tools import execute_tool, get_available_tool_names

logger = get_logger(__name__)
settings = get_settings()


async def load_context_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Load article context and conversation history.

    Fetches article details and deepsearch report from database,
    and loads cached session from Redis if available.

    Args:
        state: Current chat state
        runtime: LangGraph runtime with context

    Returns:
        State update with article_context, deepsearch_result, messages
    """
    session = runtime.context.session
    article_id = state["article_id"]
    conversation_id = state["conversation_id"]

    logger.info("Loading context", article_id=article_id, conversation_id=conversation_id)

    updates: dict[str, Any] = {}

    try:
        cache = get_chat_cache_service()

        # Load session data (messages, tokens) from Redis
        cached_session = await cache.get_session(conversation_id)
        if cached_session:
            updates["messages"] = cached_session.get("messages", [])
            updates["current_tokens"] = cached_session.get("current_tokens", 0)
            logger.debug("Loaded session from cache", conversation_id=conversation_id)
        else:
            # Fallback: Load session from PostgreSQL if Redis expired/not exist
            db_session_data = await cache.load_memories_from_db(session, conversation_id, memory_type="session")
            if db_session_data:
                updates["messages"] = db_session_data.get("messages", [])
                updates["current_tokens"] = db_session_data.get("current_tokens", 0)
                # Re-cache to Redis for faster subsequent access
                await cache.set_session(conversation_id, db_session_data)
                logger.info("Loaded session from PostgreSQL and re-cached", conversation_id=conversation_id)

        # Load memory data (user_profile, conversation_state) from Redis
        cached_memories = await cache.get_memories(conversation_id)
        if cached_memories:
            updates["user_profile"] = cached_memories.get("user_profile", {})
            updates["conversation_state"] = cached_memories.get("conversation_state", {})
            logger.debug("Loaded memories from cache", conversation_id=conversation_id)
        else:
            # Fallback: Load from PostgreSQL if Redis expired/not exist
            db_memories = await cache.load_memories_from_db(session, conversation_id)
            if db_memories:
                updates["user_profile"] = db_memories.get("user_profile", {})
                updates["conversation_state"] = db_memories.get("conversation_state", {})
                # Re-cache to Redis for faster subsequent access (exclude session)
                memory_data_only = {
                    "user_profile": db_memories.get("user_profile", {}),
                    "conversation_state": db_memories.get("conversation_state", {}),
                }
                await cache.set_memories(conversation_id, memory_data_only)
                logger.info("Loaded memories from PostgreSQL and re-cached", conversation_id=conversation_id)

        # Fetch article from database
        stmt = select(NewsArticle).where(NewsArticle.id == UUID(article_id))
        result = await session.execute(stmt)
        article = result.scalar_one_or_none()

        if article:
            article_context = {
                "id": str(article.id),
                "title": article.chinese_title or article.original_title,
                "original_title": article.original_title,
                "summary": article.chinese_summary or article.original_description or "",
                "content": article.full_content or "",
                "source_name": article.source_name,
                "source_url": article.source_url,
                "published_at": str(article.published_at) if article.published_at else None,
                "total_score": article.total_score,
            }

            # Cache article context
            await cache.set_article_context(article_id, article_context)

            updates["article_context"] = article_context

            # Include deepsearch if available
            if article.deepsearch_report:
                updates["deepsearch_result"] = article.deepsearch_report

        else:
            logger.warning("Article not found", article_id=article_id)
            updates["errors"] = [{"phase": "load_context", "message": "Article not found"}]

        # Load recent messages from database if not cached
        if not cached_session:
            msg_stmt = (
                select(ChatMessage)
                .where(ChatMessage.conversation_id == UUID(conversation_id))
                .order_by(ChatMessage.created_at.desc())
                .limit(10)
            )
            msg_result = await session.execute(msg_stmt)
            messages = msg_result.scalars().all()

            # Convert to state format (reversed order)
            formatted_messages = []
            for msg in reversed(messages):
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "agent_name": msg.agent_name,
                    "tool_calls": msg.tool_calls,
                    "tool_results": msg.tool_results,
                    "citations": msg.citations,
                    "created_at": str(msg.created_at),
                    "tokens_used": msg.tokens_used,
                })

            updates["messages"] = formatted_messages
            updates["current_tokens"] = sum(m.get("tokens_used", 0) for m in formatted_messages)

        return updates

    except Exception as e:
        logger.error("Failed to load context", error=str(e))
        return {"errors": [{"phase": "load_context", "message": str(e)}]}


async def supervisor_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Supervisor node - central hub for routing decisions.

    Evaluates current state and decides which agent should handle next:
    - researcher: Need more information
    - explainer: Have enough info, generate response
    - fact_checker: Response generated, verify it
    - end: Fact check passed, complete

    Args:
        state: Current chat state
        runtime: LangGraph runtime

    Returns:
        State update with routing decision
    """
    user_message = state["user_message"]
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])
    routing_history = state.get("routing_history", [])
    generated_response = state.get("generated_response")
    fact_check_result = state.get("fact_check_result")
    research_iterations = state.get("research_iterations", 0)
    max_research_iterations = state.get("max_research_iterations", 5)

    logger.info(
        "Supervisor evaluating state",
        routing_steps=len(routing_history),
        research_iterations=research_iterations,
        has_response=generated_response is not None,
        fact_check_passed=fact_check_result.get("passed") if fact_check_result else None,
    )

    try:
        llm_service = get_llm_service()

        # Build routing prompt
        routing_prompt = _build_supervisor_routing_prompt(state)

        # Get routing decision from LLM
        response = await llm_service.chat_completion(
            system_prompt=get_prompt("chat.supervisor_route").template,
            user_prompt=routing_prompt,
            temperature=0.3,
            max_tokens=300,
        )

        # Parse decision
        decision = _parse_supervisor_decision(response)

        route = decision.get("route", "researcher")
        info_completeness = decision.get("info_completeness", state.get("info_completeness", 0.0))

        logger.info(
            "Supervisor routing decision",
            route=route,
            info_completeness=info_completeness,
            reasoning=decision.get("reasoning", "")[:100],
        )

        return {
            "current_agent": route,
            "info_completeness": info_completeness,
            "routing_history": ["supervisor"],  # Add to history via list merging
        }

    except Exception as e:
        logger.error("Supervisor routing failed", error=str(e))
        # Default routing based on simple heuristics
        return _fallback_routing(state)


def _build_supervisor_routing_prompt(state: ChatState) -> str:
    """Build the supervisor routing prompt."""
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])
    fact_check_result = state.get("fact_check_result")

    # Format collected info summary
    collected_summary = ""
    if collected_info:
        for i, info in enumerate(collected_info[:5], 1):
            content = info.get("content", "")[:200]
            collected_summary += f"{i}. {info.get('source', 'unknown')}: {content}...\n"
    else:
        collected_summary = "暂无收集的信息"

    # Format fact check reason section
    fact_check_section = ""
    if fact_check_result and not fact_check_result.get("passed"):
        reason = fact_check_result.get("reason", "未知原因")
        fact_check_section = get_prompt("chat.supervisor_fact_check_failed").format(
            fact_check_reason=reason
        )

    prompt_data = {
        "info_completeness": f"{state.get('info_completeness', 0.0):.2f}",
        "collected_info_count": len(collected_info),
        "research_iterations": f"{state.get('research_iterations', 0)}/{state.get('max_research_iterations', 5)}",
        "has_response": "是" if state.get("generated_response") else "否",
        "fact_check_status": _get_fact_check_status(fact_check_result),
        "article_title": article_context.get("title", "未知"),
        "article_summary": article_context.get("summary", "")[:300],
        "user_message": state["user_message"],
        "collected_info_summary": collected_summary,
        "fact_check_reason_section": fact_check_section,
    }

    # Build prompt using template variables
    template = get_prompt("chat.supervisor_route")
    return template.format(**prompt_data)


def _get_fact_check_status(fact_check_result: dict | None) -> str:
    """Get human-readable fact check status."""
    if fact_check_result is None:
        return "未核查"
    if fact_check_result.get("passed"):
        return "通过"
    return "未通过"


def _parse_supervisor_decision(response: str) -> dict[str, Any]:
    """Parse LLM supervisor decision."""
    try:
        # Try to extract JSON
        json_str = _extract_json(response)
        parsed = json.loads(json_str)

        # Validate route
        route = parsed.get("route", "researcher")
        if route not in ["researcher", "explainer", "fact_checker", "end"]:
            route = "researcher"

        return {
            "route": route,
            "reasoning": parsed.get("reasoning", ""),
            "info_completeness": float(parsed.get("info_completeness", 0.5)),
        }
    except (json.JSONDecodeError, ValueError):
        # Fallback: check for keywords
        response_lower = response.lower()
        if "end" in response_lower or "完成" in response_lower:
            return {"route": "end", "reasoning": "Fallback: end detected", "info_completeness": 1.0}
        elif "explainer" in response_lower or "解释" in response_lower:
            return {"route": "explainer", "reasoning": "Fallback: explainer detected", "info_completeness": 0.8}
        elif "fact" in response_lower or "核查" in response_lower:
            return {"route": "fact_checker", "reasoning": "Fallback: fact checker detected", "info_completeness": 0.9}
        else:
            return {"route": "researcher", "reasoning": "Fallback: default to researcher", "info_completeness": 0.3}


def _fallback_routing(state: ChatState) -> dict[str, Any]:
    """Fallback routing based on simple heuristics."""
    generated_response = state.get("generated_response")
    fact_check_result = state.get("fact_check_result")
    research_iterations = state.get("research_iterations", 0)
    max_research_iterations = state.get("max_research_iterations", 5)
    info_completeness = state.get("info_completeness", 0.0)

    # If fact check failed
    if fact_check_result and not fact_check_result.get("passed"):
        # If we already have enough info, route to explainer to retry with fact check issues
        if info_completeness >= 0.7:
            return {
                "current_agent": "explainer",
                "routing_history": ["supervisor"],
            }
        # Otherwise need more research
        return {
            "current_agent": "researcher",
            "routing_history": ["supervisor"],
        }

    # If response generated but not checked
    if generated_response and fact_check_result is None:
        return {
            "current_agent": "fact_checker",
            "routing_history": ["supervisor"],
        }

    # If fact check passed
    if fact_check_result and fact_check_result.get("passed"):
        return {
            "current_agent": "end",
            "routing_history": ["supervisor"],
        }

    # If have enough info but no response
    if info_completeness >= 0.7 and not generated_response:
        return {
            "current_agent": "explainer",
            "routing_history": ["supervisor"],
        }

    # Need more research
    if research_iterations < max_research_iterations:
        return {
            "current_agent": "researcher",
            "routing_history": ["supervisor"],
        }

    # Max iterations reached, try to answer anyway
    return {
        "current_agent": "explainer",
        "routing_history": ["supervisor"],
    }


def _extract_json(text: str) -> str:
    """Extract JSON from text that might contain markdown code blocks."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


async def researcher_think_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Researcher think node - decide next action in ReAct loop.

    This is the reasoning step of ReAct:
    1. Analyze what information is needed
    2. Decide: use tool or conclude research

    Returns to supervisor if concluding, otherwise routes to researcher_tool.

    Args:
        state: Current chat state
        runtime: LangGraph runtime with session

    Returns:
        State update with thought and pending tool (or conclusion)
    """
    user_message = state["user_message"]
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])
    research_iterations = state.get("research_iterations", 0)
    max_research_iterations = state.get("max_research_iterations", 5)

    logger.info(
        "Researcher thinking",
        iteration=research_iterations,
        max_iterations=max_research_iterations,
        collected_count=len(collected_info),
    )

    try:
        llm_service = get_llm_service()

        # Build ReAct prompt
        react_prompt = _build_researcher_react_prompt(state)

        # Get LLM decision
        response = await llm_service.chat_completion(
            system_prompt=get_prompt("chat.researcher_think").template,
            user_prompt=react_prompt,
            temperature=0.6,
            max_tokens=400,
        )

        # Parse decision
        thought = _parse_researcher_thought(response, research_iterations)

        logger.info(
            "Researcher thought",
            action=thought.get("action"),
            thought=thought.get("thought", "")[:100],
        )

        # Check if should conclude or max iterations reached
        if thought.get("action") == "conclude" or research_iterations >= max_research_iterations:
            logger.info("Researcher concluding research")
            return {
                "current_thought": thought,
                "info_completeness": 0.9,  # Research complete
                "research_iterations": research_iterations + 1,
                "routing_history": ["researcher_think"],
                "pending_tool": None,  # Clear pending tool
                "pending_tool_input": None,
            }

        # Set pending tool for tool node
        tool_name = thought.get("action")
        tool_input = thought.get("action_input", {})

        return {
            "current_thought": thought,
            "pending_tool": tool_name,
            "pending_tool_input": tool_input,
            "routing_history": ["researcher_think"],
        }

    except Exception as e:
        logger.error("Researcher think failed", error=str(e))
        return {
            "research_iterations": research_iterations + 1,
            "routing_history": ["researcher_think"],
            "errors": [{"phase": "researcher_think", "message": str(e)}],
            "pending_tool": None,
            "pending_tool_input": None,
        }


async def researcher_tool_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Researcher tool node - execute tool in ReAct loop.

    This is the action step of ReAct:
    1. Execute the pending tool
    2. Record tool result
    3. Always returns to researcher_think to observe and continue

    Args:
        state: Current chat state
        runtime: LangGraph runtime with session

    Returns:
        State update with tool results, then loops back to researcher_think
    """
    session = runtime.context.session
    tool_name = state.get("pending_tool")
    tool_input = state.get("pending_tool_input", {})
    thought = state.get("current_thought", {})
    research_iterations = state.get("research_iterations", 0)

    logger.info(
        "Researcher executing tool",
        tool=tool_name,
        iteration=research_iterations,
    )

    if not tool_name:
        logger.warning("No pending tool to execute")
        return {
            "routing_history": ["researcher_tool"],
            "pending_tool": None,
            "pending_tool_input": None,
        }

    try:
        if tool_name in get_available_tool_names():
            tool_output = await execute_tool(session, tool_name, tool_input)

            # Record tool call
            tool_call = {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Add to collected info
            new_collected_info: CollectedInfo = {
                "source": tool_name,
                "content": tool_output[:1000],  # Truncate long outputs
                "relevance": thought.get("thought", ""),
                "metadata": tool_input,
            }

            logger.info(
                "Tool executed",
                tool=tool_name,
                output_length=len(tool_output),
            )

            return {
                "collected_info": [new_collected_info],
                "tool_history": [tool_call],
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "research_iterations": research_iterations + 1,
                "routing_history": ["researcher_tool"],
                "pending_tool": None,  # Clear pending tool after execution
                "pending_tool_input": None,
            }
        else:
            logger.warning("Unknown tool requested", tool=tool_name)
            return {
                "research_iterations": research_iterations + 1,
                "routing_history": ["researcher_tool"],
                "errors": [{"phase": "researcher_tool", "message": f"Unknown tool: {tool_name}"}],
                "pending_tool": None,
                "pending_tool_input": None,
            }

    except Exception as e:
        logger.error("Tool execution failed", error=str(e), tool=tool_name)
        return {
            "research_iterations": research_iterations + 1,
            "routing_history": ["researcher_tool"],
            "errors": [{"phase": "researcher_tool", "message": str(e)}],
            "pending_tool": None,
            "pending_tool_input": None,
        }


def _build_researcher_react_prompt(state: ChatState) -> str:
    """Build the researcher ReAct prompt."""
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])

    # Format collected info summary
    collected_summary = ""
    if collected_info:
        for i, info in enumerate(collected_info, 1):
            content = info.get("content", "")[:300]
            source = info.get("source", "unknown")
            collected_summary += f"{i}. [{source}] {content}\n\n"
    else:
        collected_summary = "暂无收集的信息"

    prompt = get_prompt("chat.researcher_think")
    return prompt.format(
        article_title=article_context.get("title", "未知"),
        article_name=article_context.get("source_name", "未知"),
        article_summary=article_context.get("summary", "")[:500],
        user_message=state["user_message"],
        collected_info_count=len(collected_info),
        collected_info_summary=collected_summary,
        current_iteration=state.get("research_iterations", 0),
        max_iterations=state.get("max_research_iterations", 5),
    )


def _parse_researcher_thought(response: str, iteration: int) -> ResearchThought:
    """Parse researcher LLM response into ResearchThought."""
    try:
        json_str = _extract_json(response)
        parsed = json.loads(json_str)

        action = parsed.get("action", "conclude")
        action_input = parsed.get("action_input")

        # Validate action - must be one of the valid tools or conclude
        valid_actions = [
            "vector_search",
            "web_search",
            "article_lookup",
            "citation_lookup",
            "conversation_history",
            "related_articles",
            "conclude",
        ]
        if action not in valid_actions:
            action = "conclude"
            action_input = None

        return ResearchThought(
            thought=parsed.get("thought", ""),
            action=action,
            action_input=action_input if action_input else {},
            iteration=iteration,
        )
    except (json.JSONDecodeError, ValueError):
        # Fallback: simple keyword detection
        response_lower = response.lower()
        if "web_search" in response_lower or "网络搜索" in response_lower:
            return ResearchThought(
                thought="Fallback: web search detected",
                action="web_search",
                action_input={"query": "相关信息"},
                iteration=iteration,
            )
        elif "vector_search" in response_lower or "向量搜索" in response_lower or "related_articles" in response_lower:
            return ResearchThought(
                thought="Fallback: vector search detected",
                action="vector_search",
                action_input={"query": "相关文章"},
                iteration=iteration,
            )
        elif "citation_lookup" in response_lower or "实体查询" in response_lower or "知识图谱" in response_lower:
            return ResearchThought(
                thought="Fallback: citation lookup detected",
                action="citation_lookup",
                action_input={"entity_name": "相关信息"},
                iteration=iteration,
            )
        elif "conversation_history" in response_lower or "对话历史" in response_lower:
            return ResearchThought(
                thought="Fallback: conversation history detected",
                action="conversation_history",
                action_input={"conversation_id": ""},
                iteration=iteration,
            )
        else:
            return ResearchThought(
                thought="Fallback: concluding research",
                action="conclude",
                action_input=None,
                iteration=iteration,
            )


async def explainer_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Explainer node - generates response based on collected information.

    Uses article context + collected info from researcher to generate
    a comprehensive response to the user's question.

    Args:
        state: Current chat state
        runtime: LangGraph runtime

    Returns:
        State update with generated_response
    """
    user_message = state["user_message"]
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])

    logger.info(
        "Explainer generating response",
        user_message=user_message[:50],
        collected_count=len(collected_info),
    )

    try:
        llm_service = get_llm_service()

        # Build prompt with collected info
        prompt = _build_explainer_prompt(state)

        # Generate response
        response = await llm_service.chat_completion(
            system_prompt=get_prompt("chat.explainer_with_research").template,
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=800,
        )

        logger.info(
            "Explainer response generated",
            response_length=len(response),
        )

        return {
            "generated_response": response,
            "routing_history": ["explainer"],
            "fact_check_result": None,  # Clear old fact check result for new response
        }

    except Exception as e:
        logger.error("Explainer failed", error=str(e))
        return {
            "generated_response": f"抱歉，生成回复时出现错误：{str(e)}",
            "routing_history": ["explainer"],
            "errors": [{"phase": "explainer", "message": str(e)}],
        }


def _build_explainer_prompt(state: ChatState) -> str:
    """Build the explainer prompt with collected info and fact check issues."""
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])
    fact_check_result = state.get("fact_check_result")

    # Format collected info summary
    collected_summary = ""
    if collected_info:
        for i, info in enumerate(collected_info, 1):
            content = info.get("content", "")[:400]
            source = info.get("source", "unknown")
            relevance = info.get("relevance", "")
            collected_summary += f"### 信息 {i}\n"
            collected_summary += f"来源：{source}\n"
            collected_summary += f"内容：{content}\n"
            if relevance:
                collected_summary += f"相关性：{relevance[:100]}\n"
            collected_summary += "\n"
    else:
        collected_summary = "未收集到额外信息，请基于文章上下文回答。"

    # Build fact check issues section if present
    fact_check_issues_section = ""
    fact_check_avoid_instruction = ""
    if fact_check_result and not fact_check_result.get("passed"):
        issues = fact_check_result.get("issues", [])
        reason = fact_check_result.get("reason", "")
        if issues or reason:
            issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else reason
            fact_check_issues_section = get_prompt("chat.explainer_fact_check_issues").format(
                fact_check_issues=issues_text
            )
            fact_check_avoid_instruction = get_prompt("chat.explainer_avoid_hallucination").template

    prompt = get_prompt("chat.explainer_with_research")
    return prompt.format(
        article_title=article_context.get("title", "未知"),
        article_name=article_context.get("source_name", "未知"),
        article_summary=article_context.get("summary", "")[:500],
        user_message=state["user_message"],
        collected_info_summary=collected_summary,
        fact_check_issues_section=fact_check_issues_section,
        fact_check_avoid_instruction=fact_check_avoid_instruction,
    )


async def fact_checker_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Fact checker node - validates response for hallucinations.

    Checks that the generated response:
    1. Is based on actual collected information
    2. Does not contain fabricated facts
    3. Has accurate citations

    Args:
        state: Current chat state
        runtime: LangGraph runtime with session

    Returns:
        State update with fact_check_result
    """
    generated_response = state.get("generated_response", "")
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])

    logger.info(
        "Fact checker validating response",
        response_length=len(generated_response),
        collected_count=len(collected_info),
    )

    if not generated_response:
        return {
            "fact_check_result": FactCheckResult(
                passed=False,
                reason="没有生成的回复可供核查",
                issues=["空回复"],
            ),
            "routing_history": ["fact_checker"],
        }

    try:
        llm_service = get_llm_service()

        # Build verification prompt
        prompt = _build_fact_checker_prompt(state)

        # Get verification result
        response = await llm_service.chat_completion(
            system_prompt=get_prompt("chat.fact_checker_verify").template,
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=400,
        )

        # Parse result
        result = _parse_fact_check_result(response)

        logger.info(
            "Fact check completed",
            passed=result.get("passed", False),
            issues_count=len(result.get("issues", [])),
        )

        return {
            "fact_check_result": result,
            "routing_history": ["fact_checker"],
        }

    except Exception as e:
        logger.error("Fact checker failed", error=str(e))
        return {
            "fact_check_result": FactCheckResult(
                passed=True,  # Pass on error to avoid blocking
                reason=f"核查过程出错，默认通过：{str(e)}",
                issues=[],
            ),
            "routing_history": ["fact_checker"],
            "errors": [{"phase": "fact_checker", "message": str(e)}],
        }


def _build_fact_checker_prompt(state: ChatState) -> str:
    """Build the fact checker verification prompt."""
    article_context = state.get("article_context", {})
    collected_info = state.get("collected_info", [])
    generated_response = state.get("generated_response", "")

    # Format collected info summary
    collected_summary = ""
    if collected_info:
        for i, info in enumerate(collected_info, 1):
            content = info.get("content", "")[:300]
            source = info.get("source", "unknown")
            collected_summary += f"{i}. [{source}] {content}\n"
    else:
        collected_summary = "未收集到额外信息"

    prompt = get_prompt("chat.fact_checker_verify")
    return prompt.format(
        article_title=article_context.get("title", "未知"),
        article_summary=article_context.get("summary", "")[:300],
        collected_info_summary=collected_summary,
        generated_response=generated_response,
    )


def _parse_fact_check_result(response: str) -> FactCheckResult:
    """Parse fact check LLM response."""
    try:
        json_str = _extract_json(response)
        parsed = json.loads(json_str)

        return FactCheckResult(
            passed=bool(parsed.get("passed", False)),
            reason=parsed.get("reason", ""),
            issues=parsed.get("issues", []),
        )
    except (json.JSONDecodeError, ValueError):
        # Fallback: check for pass/fail keywords
        response_lower = response.lower()
        if "pass" in response_lower or "通过" in response_lower:
            return FactCheckResult(
                passed=True,
                reason="Fallback: pass detected",
                issues=[],
            )
        else:
            return FactCheckResult(
                passed=False,
                reason="Fallback: fail detected",
                issues=["无法解析核查结果"],
            )


async def format_response_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Format final response and save to database.

    Prepares the response for API return and persists
    the conversation to PostgreSQL.

    Args:
        state: Current chat state
        runtime: LangGraph runtime with session

    Returns:
        State update marking completion
    """
    session = runtime.context.session
    conversation_id = state["conversation_id"]
    user_message = state["user_message"]
    generated_response = state.get("generated_response", "")
    collected_info = state.get("collected_info", [])
    tool_history = state.get("tool_history", [])

    logger.info(
        "Formatting response",
        conversation_id=conversation_id,
        response_length=len(generated_response),
    )

    try:
        # Extract citations from collected info
        citations = _extract_citations_from_collected_info(collected_info)

        # Add user message to database
        user_msg = ChatMessage(
            conversation_id=UUID(conversation_id),
            role="user",
            content=user_message,
            tokens_used=estimate_tokens(user_message),
        )
        session.add(user_msg)

        # Add assistant message to database
        assistant_msg = ChatMessage(
            conversation_id=UUID(conversation_id),
            role="assistant",
            content=generated_response,
            agent_name="explainer",
            citations=citations if citations else None,
            tool_calls=[tc for tc in tool_history] if tool_history else None,
            tokens_used=estimate_tokens(generated_response),
        )
        session.add(assistant_msg)

        # Update conversation metadata
        conv_stmt = select(ChatConversation).where(
            ChatConversation.id == UUID(conversation_id)
        )
        conv_result = await session.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        if conversation:
            conversation.message_count += 2  # User + assistant
            conversation.last_message_at = datetime.now(timezone.utc)

        await session.commit()

        # Invalidate Redis cache to force fresh load next time
        cache = get_chat_cache_service()
        await cache.invalidate_session(conversation_id)

        # Create state message objects for the current turn (for LangGraph state)
        user_msg_state = create_user_message(user_message)
        assistant_msg_state = create_assistant_message(
            content=generated_response,
            agent_name="explainer",
            citations=citations if citations else None,
            tool_calls=[tc for tc in tool_history] if tool_history else None,
            tokens_used=estimate_tokens(generated_response),
        )

        return {
            "final_response": generated_response,
            "final_citations": citations,
            "is_complete": True,
            "messages": [user_msg_state, assistant_msg_state],
            "current_tokens": state.get("current_tokens", 0)
                + estimate_tokens(user_message)
                + estimate_tokens(generated_response),
        }

    except Exception as e:
        logger.error("Failed to save response", error=str(e))
        await session.rollback()
        return {
            "final_response": generated_response or f"抱歉，保存回复时出错：{str(e)}",
            "final_citations": [],
            "is_complete": True,
            "errors": [{"phase": "format_response", "message": str(e)}],
        }


def _extract_citations_from_collected_info(collected_info: list[CollectedInfo]) -> list[dict[str, Any]]:
    """Extract citations from collected info."""
    citations = []
    for info in collected_info:
        source = info.get("source", "")
        content = info.get("content", "")

        citation = {
            "source_type": source,
            "source_name": source,
            "content_snippet": content[:100] if content else "",
        }

        # Try to extract URL from web search results
        if source == "web_search":
            import json
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for item in parsed[:3]:
                        if "url" in item:
                            citations.append({
                                "source_type": "web",
                                "source_name": item.get("title", item.get("source", "")),
                                "url": item.get("url", ""),
                                "content_snippet": item.get("snippet", "")[:100],
                            })
            except json.JSONDecodeError:
                pass
        else:
            citations.append(citation)

    return citations[:10]  # Limit to 10 citations


async def memory_manager_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Manage memory: micro-compact and extraction.

    Runs micro-compact to clean old tool results, and
    extracts memories (user_profile, conversation_state) at end of conversation.

    Args:
        state: Current chat state
        runtime: LangGraph runtime

    Returns:
        State update with cleaned messages and extracted memories
    """
    messages = state.get("messages", [])
    tool_call_count = state.get("tool_call_count", 0)
    conversation_id = state["conversation_id"]

    logger.info(
        "Memory manager running",
        conversation_id=conversation_id,
        tool_call_count=tool_call_count,
        messages_count=len(messages),
    )

    updates: dict[str, Any] = {}

    try:
        cache = get_chat_cache_service()

        # === Micro-Compact: Clean tool results >30 min old ===
        from datetime import timedelta

        expiry_threshold = datetime.now(timezone.utc) - timedelta(
            minutes=settings.chat_tool_expiry_minutes
        )

        cleaned_messages = []
        for msg in messages:
            # Keep non-tool messages
            if msg.get("role") != "tool":
                cleaned_messages.append(msg)
                continue

            # Check if tool result is expired
            msg_time_str = msg.get("created_at", "")
            if msg_time_str:
                try:
                    msg_time = datetime.fromisoformat(msg_time_str.replace("Z", "+00:00"))
                    if msg_time > expiry_threshold:
                        cleaned_messages.append(msg)
                    else:
                        logger.debug("Micro-compact removed tool result")
                except ValueError:
                    cleaned_messages.append(msg)  # Keep on parse error

        if len(cleaned_messages) < len(messages):
            updates["messages"] = cleaned_messages  # Replace with cleaned
            logger.info("====== Micro-compact completed ======", messages_removed=len(messages) - len(cleaned_messages))

        # === Memory Extraction: Always extract at end of conversation ===
        # Extract memories from the conversation using LLM
        extracted = await _extract_memories_llm(state)
        if extracted:
            updates["extracted_memories"] = extracted
            # Extract user_profile and conversation_state from results
            for mem in extracted:
                if mem.get("memory_type") == "user_profile":
                    updates["user_profile"] = mem.get("content")
                elif mem.get("memory_type") == "conversation_state":
                    updates["conversation_state"] = mem.get("content")

            # Cache memories to Redis (separate from session cache)
            memory_data = {
                "user_profile": updates.get("user_profile", state.get("user_profile", {})),
                "conversation_state": updates.get("conversation_state", state.get("conversation_state", {})),
            }
            await cache.set_memories(conversation_id, memory_data)
            logger.info("Memories extracted and cached", conversation_id=conversation_id)

        # Get database session for PostgreSQL persistence
        db_session = runtime.context.session

        # Persist memories to PostgreSQL (outlasts Redis TTL)
        if extracted:
            await cache.save_memories_to_db(db_session, conversation_id, memory_data)
            logger.info("Memories persisted to PostgreSQL", conversation_id=conversation_id)

        # Update session cache (messages and tokens only)
        session_data = {
            "messages": cleaned_messages,
            "current_tokens": state.get("current_tokens", 0),
            "message_count": len(cleaned_messages),
        }
        await cache.set_session(conversation_id, session_data)

        # Persist session to PostgreSQL (memory_type="session")
        await cache.save_memories_to_db(db_session, conversation_id, {"session": session_data})
        logger.info("Session persisted to PostgreSQL", conversation_id=conversation_id)

        return updates

    except Exception as e:
        logger.error("Memory manager failed", error=str(e))
        return {"errors": [{"phase": "memory_manager", "message": str(e)}]}


async def _extract_memories_simple(state: ChatState) -> list[dict[str, Any]]:
    """Simple memory extraction without LLM.

    Args:
        state: Current state

    Returns:
        List of extracted memory dicts with structure:
        - user_profile: {"interests": [...]}
        - conversation_state: {"current_topic": str, "information_gathered": [...], "open_questions": [...]}
    """
    messages = state.get("messages", [])
    collected_info = state.get("collected_info", [])

    memories = []

    # === Extract user_profile ===
    existing_profile = state.get("user_profile") or {}
    existing_interests = existing_profile.get("interests", []) if isinstance(existing_profile, dict) else []

    # Detect interests from recent user messages
    recent_user_messages = [m for m in messages[-5:] if m.get("role") == "user"]
    interests = list(existing_interests)  # Copy existing interests

    if recent_user_messages:
        tech_keywords = ["ai", "ml", "machine learning", "neural", "gpt", "llm",
                        "startup", "funding", "acquisition", "cloud", "api",
                        "agent", "rag", "embedding", "vector", "database"]
        for msg in recent_user_messages:
            content = msg.get("content", "").lower()
            for kw in tech_keywords:
                if kw in content and kw not in interests:
                    interests.append(kw)

    user_profile_content = {
        "interests": interests[:10]  # Limit to 10
    }

    memories.append({
        "memory_type": "user_profile",
        "content": user_profile_content,
        "extraction_iteration": state.get("tool_call_count", 0) // settings.chat_extraction_interval,
    })

    # === Extract conversation_state ===
    # Get current topic from last user message
    current_topic = None
    if messages:
        # Find the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_topic = msg.get("content", "")[:200]
                break

    # Extract gathered info from collected_info (research results)
    information_gathered = []
    if collected_info:
        for info in collected_info[:5]:
            source = info.get("source", "unknown")
            snippet = info.get("content", "")[:100]
            information_gathered.append(f"[{source}] {snippet}")

    conversation_state_content = {
        "current_topic": current_topic,
        "information_gathered": information_gathered[:10],
        "open_questions": [],  # Could be enhanced to extract from user questions
    }

    memories.append({
        "memory_type": "conversation_state",
        "content": conversation_state_content,
        "extraction_iteration": state.get("tool_call_count", 0) // settings.chat_extraction_interval,
    })

    return memories


async def _extract_memories_llm(state: ChatState) -> list[dict[str, Any]]:
    """LLM-based memory extraction for user interests and conversation state.

    Uses LLM to intelligently extract:
    - User interests from conversation patterns
    - Expertise level from question complexity
    - Preferred response style
    - Key entities and conversation state

    Args:
        state: Current state

    Returns:
        List of extracted memory dicts with structure:
        - user_profile: {"interests": [...], "expertise_level": str, "preferred_style": str}
        - conversation_state: {"current_topic": str, "key_entities": [...], ...}
    """
    messages = state.get("messages", [])
    collected_info = state.get("collected_info", [])
    existing_profile = state.get("user_profile") or {}

    llm_service = get_llm_service()

    # Format conversation history (last 10 messages)
    history_str = _format_messages_for_extraction(messages[-10:])

    # Format collected info
    info_str = _format_collected_info_for_extraction(collected_info[:5])

    # Format existing profile
    profile_str = json.dumps(existing_profile, ensure_ascii=False) if existing_profile else "暂无"

    # Get prompts
    system_prompt = get_prompt("chat.memory_extraction_system").template
    user_prompt = get_prompt("chat.memory_extraction_user").format(
        existing_profile=profile_str,
        conversation_history=history_str,
        collected_info=info_str,
    )

    try:
        response = await llm_service.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Low temperature for structured extraction
            max_tokens=500,
        )

        # Parse JSON from response
        extracted = _parse_extraction_json(response)

        memories = []

        if extracted.get("user_profile"):
            memories.append({
                "memory_type": "user_profile",
                "content": extracted["user_profile"],
                "extraction_iteration": state.get("tool_call_count", 0) // settings.chat_extraction_interval,
            })

        if extracted.get("conversation_state"):
            memories.append({
                "memory_type": "conversation_state",
                "content": extracted["conversation_state"],
                "extraction_iteration": state.get("tool_call_count", 0) // settings.chat_extraction_interval,
            })

        if memories:
            logger.info("LLM memory extraction succeeded", memories_count=len(memories))
            return memories

        # If no valid memories extracted, fallback to simple
        logger.warning("LLM extraction returned empty, falling back to simple")
        return await _extract_memories_simple(state)

    except Exception as e:
        logger.error("LLM memory extraction failed", error=str(e))
        # Fallback to simple extraction
        return await _extract_memories_simple(state)


def _format_messages_for_extraction(messages: list[dict]) -> str:
    """Format messages for extraction prompt.

    Args:
        messages: List of message dicts

    Returns:
        Formatted string for LLM input
    """
    if not messages:
        return "暂无"

    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:200]  # Truncate long messages
        agent_name = msg.get("agent_name", "")
        role_label = f"{role}" if not agent_name else f"{role}({agent_name})"
        lines.append(f"[{role_label}]: {content}")
    return "\n".join(lines)


def _format_collected_info_for_extraction(info: list[dict]) -> str:
    """Format collected info for extraction prompt.

    Args:
        info: List of collected info dicts

    Returns:
        Formatted string for LLM input
    """
    if not info:
        return "暂无"

    lines = []
    for item in info:
        source = item.get("source", "unknown")
        content = item.get("content", "")[:100]
        relevance = item.get("relevance", "")[:50]
        lines.append(f"- [{source}]: {content} (相关: {relevance})")
    return "\n".join(lines)


def _parse_extraction_json(response: str) -> dict:
    """Parse JSON from LLM extraction response.

    Args:
        response: LLM response text

    Returns:
        Parsed JSON dict or empty dict on failure
    """
    # Find JSON in response
    start = response.find("{")
    end = response.rfind("}") + 1

    if start != -1 and end > start:
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed in extraction", error=str(e), snippet=json_str[:100])

    return {}


async def check_compress_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Check if summary compression is needed.

    Evaluates token count against threshold and flags
    if summary compression should run.

    Args:
        state: Current chat state
        runtime: LangGraph runtime

    Returns:
        State update with should_compress flag
    """
    current_tokens = state.get("current_tokens", 0)
    max_tokens = state.get("max_tokens", settings.chat_max_tokens)
    threshold = state.get("context_threshold", settings.chat_context_threshold)

    should_compress = current_tokens >= max_tokens * threshold

    logger.info(
        "Checking compression threshold",
        current_tokens=current_tokens,
        threshold_tokens=int(max_tokens * threshold),
        should_compress=should_compress,
    )

    return {"should_compress": should_compress}


async def summary_compact_node(
    state: ChatState,
    runtime: Runtime[ChatContext],
) -> dict[str, Any]:
    """Generate summary compact boundary message.

    Creates a summary replacing older messages when
    context threshold is exceeded.

    Args:
        state: Current chat state
        runtime: LangGraph runtime

    Returns:
        State update with compact_boundaries and reduced tokens
    """
    messages = state.get("messages", [])
    current_tokens = state.get("current_tokens", 0)
    max_tokens = state.get("max_tokens", settings.chat_max_tokens)

    logger.info(
        "Running summary compact",
        messages_count=len(messages),
        current_tokens=current_tokens,
    )

    try:
        # Calculate target: 40% of max tokens after compression
        target_tokens = int(max_tokens * 0.4)
        tokens_to_remove = current_tokens - target_tokens

        # Identify messages to summarize (older ones)
        messages_to_summarize = []
        remaining_messages = []
        accumulated_tokens = 0

        for msg in messages:
            msg_tokens = msg.get("tokens_used", estimate_tokens(msg.get("content", "")))
            if accumulated_tokens < tokens_to_remove:
                messages_to_summarize.append(msg)
                accumulated_tokens += msg_tokens
            else:
                remaining_messages.append(msg)

        if not messages_to_summarize:
            return {}  # Nothing to summarize

        # Generate summary (simplified)
        summary = _generate_summary_simple(messages_to_summarize)

        # Create compact boundary message
        compact_boundary = {
            "summary": summary,
            "key_entities": [],
            "key_citations": [],
            "tool_summary": "",
            "period_start": messages_to_summarize[0].get("created_at", ""),
            "period_end": messages_to_summarize[-1].get("created_at", ""),
            "messages_replaced": len(messages_to_summarize),
        }

        # Calculate new token count
        new_tokens = current_tokens - accumulated_tokens + estimate_tokens(summary)

        logger.info(
            "====== Summary summary compact completed ======",
            messages_removed=len(messages_to_summarize),
            new_tokens=new_tokens,
        )

        # Get cache service and db session for persistence
        cache = get_chat_cache_service()
        db_session = runtime.context.session
        conversation_id = state["conversation_id"]

        # Persist compressed session to Redis and PostgreSQL
        session_data = {
            "messages": remaining_messages,
            "current_tokens": new_tokens,
            "message_count": len(remaining_messages),
            "compact_boundaries": [compact_boundary],
        }
        await cache.set_session(conversation_id, session_data)
        await cache.save_memories_to_db(db_session, conversation_id, {"session": session_data})

        logger.info("Session cache updated after compression", conversation_id=conversation_id)

        return {
            "messages": remaining_messages,  # Replace with remaining
            "compact_boundaries": [compact_boundary],
            "current_tokens": new_tokens,
            "should_compress": False,  # Reset flag
        }

    except Exception as e:
        logger.error("Summary compact failed", error=str(e))
        return {"errors": [{"phase": "summary_compact", "message": str(e)}]}


def _generate_summary_simple(messages: list[dict[str, Any]]) -> str:
    """Generate simple summary without LLM.

    Args:
        messages: Messages to summarize

    Returns:
        Summary text
    """
    user_msgs = [m.get("content", "")[:50] for m in messages if m.get("role") == "user"]
    agent_msgs = [m.get("agent_name", "unknown") for m in messages if m.get("role") == "assistant"]

    summary_parts = []
    if user_msgs:
        summary_parts.append(f"用户询问：{', '.join(user_msgs[:3])}")
    if agent_msgs:
        summary_parts.append(f"使用代理：{', '.join(set(agent_msgs))}")

    return " | ".join(summary_parts) if summary_parts else "之前的对话已摘要。"