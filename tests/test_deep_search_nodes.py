import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.deep_search import nodes
from app.deep_search.context import DeepSearchContext
from langgraph.runtime import Runtime


class DummyLLMService:
    """Dummy LLM service for testing."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def reasoning_analysis(self, system_prompt, user_prompt, temperature=0.6, enable_thinking=True):
        content = self.responses[self.calls]
        self.calls += 1
        return content

    def _extract_json(self, text):
        # Simple JSON extraction
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


@pytest.mark.asyncio
async def test_parse_reasoning_decision_repairs_truncated_json():
    llm_service = DummyLLMService([])
    response = (
        '{"thought":"need more context","action":"web_search",'
        '"action_input":{"query":"OpenAI military 2025"'
    )

    decision = await nodes._parse_reasoning_decision(
        llm_service, "system prompt", "user prompt", response
    )

    assert decision["action"] == "web_search"
    assert decision["action_input"]["query"] == "OpenAI military 2025"


@pytest.mark.asyncio
async def test_parse_reasoning_decision_retries_when_repair_fails():
    llm_service = DummyLLMService([
        json.dumps({
            "thought": "retry with valid json",
            "action": "conclude",
            "action_input": None,
        })
    ])

    decision = await nodes._parse_reasoning_decision(
        llm_service, "system prompt", "user prompt", '{"thought":"broken'
    )

    assert decision["action"] == "conclude"
    assert llm_service.calls == 1


@pytest.mark.asyncio
async def test_tools_node_appends_history_and_collected_info(monkeypatch):
    async def fake_execute_tool(_session, tool_name, tool_input):
        return f"{tool_name}:{tool_input['query']}"

    monkeypatch.setattr(nodes, "execute_tool", fake_execute_tool)

    state = {
        "pending_action": "web_search",
        "pending_action_input": {"query": "OpenAI"},
        "tool_history": [{"tool_name": "vector_search", "tool_input": {}, "tool_output": "x", "iteration": 0}],
        "collected_info": [{"source": "vector_search", "content": "x", "relevance": "r", "metadata": {}}],
        "current_iteration": 1,
        "current_thought": "search the web",
    }

    # Create a mock runtime with context
    mock_runtime = MagicMock(spec=Runtime)
    mock_runtime.context = MagicMock(spec=DeepSearchContext)
    mock_runtime.context.session = None

    result = await nodes.tools_node(state, mock_runtime)

    assert len(result["tool_history"]) == 1
    assert len(result["collected_info"]) == 1
    assert result["tool_history"][-1]["tool_name"] == "web_search"