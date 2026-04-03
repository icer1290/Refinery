"""Node implementations for deep search ReAct workflow."""

import json
import re
from datetime import datetime
from typing import Any

from langgraph.runtime import Runtime
from sqlalchemy import select

from app.core import get_logger
from app.models.orm_models import NewsArticle
from app.deep_search.context import DeepSearchContext
from app.deep_search.prompts import (
    CONCLUSION_PROMPT,
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT,
    format_collected_info,
)
from app.deep_search.state import DeepSearchState
from app.tools import execute_tool, get_available_tool_names
from app.prompts import get_prompt
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)


async def fetch_article_node(
    state: DeepSearchState,
    runtime: Runtime[DeepSearchContext],
) -> dict[str, Any]:
    """Fetch article from database by ID.

    Args:
        state: Current state
        runtime: LangGraph runtime with DeepSearchContext

    Returns:
        Updated state fields
    """
    session = runtime.context.session
    article_id = state["article_id"]

    logger.info("Fetching article", article_id=article_id)

    try:
        from uuid import UUID

        stmt = select(NewsArticle).where(NewsArticle.id == UUID(article_id))
        result = await session.execute(stmt)
        article = result.scalar_one_or_none()

        if not article:
            return {
                "errors": [{"phase": "fetch", "message": "Article not found"}],
                "is_complete": True,
                "should_continue": False,
            }

        article_dict = {
            "id": str(article.id),
            "title": article.chinese_title or article.original_title,
            "original_title": article.original_title,
            "summary": article.chinese_summary or article.original_description,
            "source": article.source_name,
            "url": article.source_url,
            "published_at": str(article.published_at) if article.published_at else "Unknown",
            "content": article.full_content,
        }

        logger.info("Article fetched", title=article_dict["title"][:50])

        return {
            "article": article_dict,
        }

    except Exception as e:
        logger.error("Failed to fetch article", error=str(e))
        return {
            "errors": [{"phase": "fetch", "message": str(e)}],
            "is_complete": True,
            "should_continue": False,
        }


async def reasoning_node(state: DeepSearchState) -> dict[str, Any]:
    """LLM reasoning node to decide next action.

    Args:
        state: Current state

    Returns:
        Updated state fields
    """
    logger.info(
        "Reasoning iteration",
        current=state["current_iteration"],
        max=state["max_iterations"],
    )

    article = state.get("article")
    if not article:
        return {
            "errors": [{"phase": "reasoning", "message": "No article to analyze"}],
            "should_continue": False,
        }

    try:
        # Get current time for context injection
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        logger.info("Injecting current time into prompt", current_time=current_time)

        # Get LLM service
        llm_service = get_llm_service()

        # Build prompt with current time injected
        collected_info_str = format_collected_info(state.get("collected_info", []))
        system_prompt = REACT_SYSTEM_PROMPT.format(current_time=current_time)

        user_prompt = REACT_USER_PROMPT.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            published_at=article.get("published_at", ""),
            summary=article.get("summary", "")[:1000] if article.get("summary") else "",
            collected_info=collected_info_str,
            current_iteration=state["current_iteration"],
            max_iterations=state["max_iterations"],
            tool_count=len(state.get("tool_history", [])),
        )

        # Get LLM response with thinking mode enabled
        response_text = await llm_service.reasoning_analysis(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.6,
            enable_thinking=True,
        )

        logger.info("LLM response received", response_preview=response_text[:300])

        # Parse response
        decision = await _parse_reasoning_decision(
            llm_service,
            system_prompt,
            user_prompt,
            response_text,
        )

        thought = decision.get("thought", "")
        action = decision.get("action", "conclude")
        action_input = decision.get("action_input")

        valid_actions = set(get_available_tool_names()) | {"conclude"}
        if action not in valid_actions:
            logger.warning("LLM returned unknown action", action=action)
            action = "conclude"
            action_input = None

        logger.info("LLM decision", action=action, thought=thought[:100])

        # Check if should conclude
        if action == "conclude" or state["current_iteration"] >= state["max_iterations"]:
            return {
                "current_thought": thought,
                "should_continue": False,
            }

        # Return with pending action for tools node
        return {
            "current_thought": thought,
            "pending_action": action,
            "pending_action_input": action_input,
            "current_iteration": state["current_iteration"] + 1,
        }

    except Exception as e:
        import traceback

        logger.error("Reasoning failed", error=str(e), traceback=traceback.format_exc())
        return {
            "errors": [{"phase": "reasoning", "message": str(e)}],
            "should_continue": False,
        }


async def tools_node(
    state: DeepSearchState,
    runtime: Runtime[DeepSearchContext],
) -> dict[str, Any]:
    """Execute tools based on reasoning decision.

    Args:
        state: Current state
        runtime: LangGraph runtime with DeepSearchContext

    Returns:
        Updated state fields
    """
    session = runtime.context.session
    action = state.get("pending_action")
    action_input = state.get("pending_action_input", {})

    if not action:
        return {}

    logger.info("Executing tool", action=action, input=str(action_input)[:100])

    try:
        # Ensure action_input is a dict
        if not isinstance(action_input, dict):
            action_input = {}

        # Execute tool
        tool_output = await execute_tool(session, action, action_input)

        logger.debug("Tool output", output=tool_output[:200])

        # Record tool call
        tool_call = {
            "tool_name": action,
            "tool_input": action_input,
            "tool_output": tool_output,
            "iteration": state["current_iteration"],
        }

        # Add to collected info
        collected_info = {
            "source": action,
            "content": tool_output,
            "relevance": state.get("current_thought", ""),
            "metadata": action_input,
        }

        return {
            "tool_history": [tool_call],
            "collected_info": [collected_info],
            "pending_action": None,
            "pending_action_input": None,
        }

    except Exception as e:
        logger.error("Tool execution failed", error=str(e))
        return {
            "errors": [{"phase": "tools", "message": str(e)}],
            "pending_action": None,
            "pending_action_input": None,
        }


async def conclude_node(state: DeepSearchState) -> dict[str, Any]:
    """Generate final deep tracking report.

    Args:
        state: Current state

    Returns:
        Updated state fields
    """
    logger.info("Generating final report")

    article = state.get("article")
    if not article:
        return {
            "final_report": "Error: No article to analyze",
            "is_complete": True,
        }

    try:
        # Get current time for context injection
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        logger.info("Injecting current time into report prompt", current_time=current_time)

        # Get LLM service
        llm_service = get_llm_service()

        # Build prompt with current time injected
        collected_info_str = format_collected_info(state.get("collected_info", []))

        prompt = CONCLUSION_PROMPT.format(
            current_time=current_time,
            title=article.get("title", ""),
            source=article.get("source", ""),
            published_at=article.get("published_at", ""),
            summary=article.get("summary", "")[:1500] if article.get("summary") else "",
            collected_info=collected_info_str,
        )

        # Generate report
        report = await llm_service.conclude_deep_search(
            prompt=prompt,
            temperature=0.6,
            enable_thinking=True,
        )

        logger.info("Report generated", length=len(report))

        return {
            "final_report": report,
            "is_complete": True,
        }

    except Exception as e:
        logger.error("Failed to generate report", error=str(e))
        return {
            "final_report": f"生成报告失败: {str(e)}",
            "is_complete": True,
            "errors": [{"phase": "conclude", "message": str(e)}],
        }


def _extract_json(text: str) -> str:
    """Extract JSON from text that might contain markdown code blocks."""
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    # Try to find JSON object directly
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


def _repair_partial_json(text: str) -> str | None:
    """Attempt lightweight JSON repair for truncated LLM responses."""
    extracted = _extract_json(text).strip()
    if not extracted:
        return None

    quote_count = extracted.count('"')
    if quote_count % 2 == 1:
        extracted += '"'

    open_braces = extracted.count("{")
    close_braces = extracted.count("}")
    if open_braces > close_braces:
        extracted += "}" * (open_braces - close_braces)

    open_brackets = extracted.count("[")
    close_brackets = extracted.count("]")
    if open_brackets > close_brackets:
        extracted += "]" * (open_brackets - close_brackets)

    extracted = re.sub(r",\s*([}\]])", r"\1", extracted)
    return extracted


async def _parse_reasoning_decision(
    llm_service,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
) -> dict[str, Any]:
    """Parse the reasoning response with repair and one-shot retry."""
    logger.info("Parsing reasoning response", response_length=len(response_text))

    try:
        json_str = _extract_json(response_text)
        logger.debug("Extracted JSON", json_preview=json_str[:200])
        parsed = json.loads(json_str)
        logger.debug("Parsed JSON", keys=list(parsed.keys()))
        return _validate_reasoning_decision(parsed)
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse LLM response as JSON",
            error=str(e),
            response=response_text[:300],
        )
    except ValueError as exc:
        logger.warning("LLM JSON missing required fields", error=str(exc))

    repaired = _repair_partial_json(response_text)
    if repaired:
        try:
            decision = _validate_reasoning_decision(json.loads(repaired))
            logger.warning("Recovered malformed LLM JSON with local repair")
            return decision
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Repair attempt failed", error=str(e))

    # Retry with LLM
    retry_messages = [
        ("system", system_prompt),
        ("user", user_prompt),
        ("assistant", response_text[:4000]),
        ("user", get_prompt("deep_search.json_repair").template),
    ]

    try:
        retry_text = await llm_service.reasoning_analysis(
            system_prompt=system_prompt,
            user_prompt="\n".join(f"{role}: {content}" for role, content in retry_messages[-2:]),
            temperature=0.6,
            enable_thinking=False,
        )
        return _validate_reasoning_decision(json.loads(_extract_json(retry_text)))
    except Exception as exc:
        logger.warning("Failed to recover LLM JSON response", error=str(exc))
        return {
            "thought": "Failed to parse response after retry, concluding search",
            "action": "conclude",
            "action_input": None,
        }


def _validate_reasoning_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Ensure the reasoning response contains the required fields."""
    if "action" not in decision:
        raise ValueError("Missing action field")
    if "thought" not in decision:
        raise ValueError("Missing thought field")
    if decision["action"] != "conclude" and decision.get("action_input") is None:
        raise ValueError("Missing action_input for non-conclude action")
    return decision