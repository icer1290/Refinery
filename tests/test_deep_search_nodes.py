import json

import pytest

from app.deep_search import nodes


class DummyLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def ainvoke(self, _messages):
        content = self.responses[self.calls]
        self.calls += 1
        return type("Response", (), {"content": content})()


@pytest.mark.asyncio
async def test_parse_reasoning_decision_repairs_truncated_json():
    llm = DummyLLM([])
    response = (
        '{"thought":"need more context","action":"web_search",'
        '"action_input":{"query":"OpenAI military 2025"'
    )

    decision = await nodes._parse_reasoning_decision(llm, [], response)

    assert decision["action"] == "web_search"
    assert decision["action_input"]["query"] == "OpenAI military 2025"


@pytest.mark.asyncio
async def test_parse_reasoning_decision_retries_when_repair_fails():
    llm = DummyLLM([
        json.dumps({
            "thought": "retry with valid json",
            "action": "conclude",
            "action_input": None,
        })
    ])

    decision = await nodes._parse_reasoning_decision(llm, [], '{"thought":"broken')

    assert decision["action"] == "conclude"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_tools_node_appends_history_and_collected_info(monkeypatch):
    async def fake_execute_tool(_session, tool_name, tool_input):
        return f"{tool_name}:{tool_input['query']}"

    monkeypatch.setattr(nodes, "execute_tool", fake_execute_tool)

    state = {
        "_pending_action": "web_search",
        "_pending_action_input": {"query": "OpenAI"},
        "tool_history": [{"tool_name": "vector_search", "tool_input": {}, "tool_output": "x", "iteration": 0}],
        "collected_info": [{"source": "vector_search", "content": "x", "relevance": "r", "metadata": {}}],
        "current_iteration": 1,
        "current_thought": "search the web",
    }

    result = await nodes.tools_node(state, session=None)

    assert len(result["tool_history"]) == 2
    assert len(result["collected_info"]) == 2
    assert result["tool_history"][-1]["tool_name"] == "web_search"
