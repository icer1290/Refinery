"""Graph construction and execution for chat workflow using LangGraph StateGraph.

Hub-and-Spoke Architecture with ReAct Loop:
- Supervisor is the central hub that evaluates state and routes to agents
- Researcher uses true ReAct loop: think → tool → think → ... → conclude
- All other agents return to Supervisor after execution

Flow:
    load_context → supervisor → [researcher_think | explainer | fact_checker | END]
                                    ↓
                              researcher_think → researcher_tool → researcher_think → ...
                                    ↓ (when conclude)
                              (returns to supervisor)
"""

from langgraph.graph import END, StateGraph
from langsmith import traceable

from app.chat.context import ChatContext
from app.chat.nodes import (
    check_compress_node,
    explainer_node,
    fact_checker_node,
    format_response_node,
    load_context_node,
    memory_manager_node,
    researcher_think_node,
    researcher_tool_node,
    summary_compact_node,
    supervisor_node,
)
from app.chat.state import ChatState, create_initial_chat_state
from app.config import get_settings
from app.core import get_logger
from app.services.chat_cache import get_chat_cache_service

logger = get_logger(__name__)
settings = get_settings()

# Maximum consecutive fact-check failures to prevent infinite retry loops
# After this many failures, force end to avoid endless explainer -> fact_checker cycles
MAX_FACT_CHECK_FAILURES = 3


def _route_after_supervisor(state: ChatState) -> str:
    """Route after supervisor based on routing decision.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    route = state.get("current_agent", "researcher_think")

    # Safety check: prevent infinite fact-check retry loops
    fact_check_failures = state.get("fact_check_failures", 0)
    if fact_check_failures >= MAX_FACT_CHECK_FAILURES:
        logger.warning(
            "Max fact-check failures reached, forcing end",
            fact_check_failures=fact_check_failures,
        )
        return "format_response"

    if route == "researcher":
        return "researcher_think"
    elif route == "explainer":
        return "explainer"
    elif route == "fact_checker":
        return "fact_checker"
    elif route == "end":
        return "format_response"
    else:
        logger.warning(f"Unknown route: {route}, defaulting to researcher_think")
        return "researcher_think"


def _route_after_researcher_think(state: ChatState) -> str:
    """Route after researcher_think - to tool node or supervisor.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    # If there's a pending tool, execute it (continue ReAct loop)
    if state.get("pending_tool"):
        return "researcher_tool"

    # Otherwise, research concluded - return to supervisor
    return "supervisor"


def _route_after_compress_check(state: ChatState) -> str:
    """Route to summary compression if needed.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    if state.get("should_compress", False):
        return "summary_compact"
    return "format_response"


def create_chat_graph():
    """Create the multi-turn chat workflow graph with ReAct loop.

    Architecture:
    - Supervisor is the central hub
    - Researcher uses true ReAct loop (think → tool → think → ... → conclude)
    - All other agents return to Supervisor after execution

    Flow:
    load_context → supervisor → [researcher_think | explainer | fact_checker]
        ↓
    researcher_think → researcher_tool → researcher_think → ... → supervisor
        ↓
    when route == "end": format_response → END

    Returns:
        Compiled LangGraph workflow with context_schema
    """
    # Create graph with context_schema for dependency injection
    graph = StateGraph(ChatState, context_schema=ChatContext)

    # === Add Nodes ===
    graph.add_node("load_context", load_context_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher_think", researcher_think_node)
    graph.add_node("researcher_tool", researcher_tool_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("format_response", format_response_node)
    graph.add_node("memory_manager", memory_manager_node)
    graph.add_node("check_compress", check_compress_node)
    graph.add_node("summary_compact", summary_compact_node)

    # === Set Entry Point ===
    graph.set_entry_point("load_context")

    # === Linear Flow to Supervisor ===
    graph.add_edge("load_context", "supervisor")

    # === Conditional Routing from Supervisor ===
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "researcher_think": "researcher_think",
            "explainer": "explainer",
            "fact_checker": "fact_checker",
            "format_response": "format_response",
        },
    )

    # === Researcher ReAct Loop ===
    # researcher_think: decides action (tool or conclude)
    graph.add_conditional_edges(
        "researcher_think",
        _route_after_researcher_think,
        {
            "researcher_tool": "researcher_tool",
            "supervisor": "supervisor",
        },
    )
    # researcher_tool: executes tool, then returns to think
    graph.add_edge("researcher_tool", "researcher_think")

    # === Other Agents Return to Supervisor ===
    graph.add_edge("explainer", "supervisor")
    graph.add_edge("fact_checker", "supervisor")

    # === Memory Management Flow (after format_response) ===
    graph.add_edge("format_response", "memory_manager")
    graph.add_edge("memory_manager", "check_compress")

    # === Conditional Routing After Compress Check ===
    graph.add_conditional_edges(
        "check_compress",
        _route_after_compress_check,
        {
            "summary_compact": "summary_compact",
            "format_response": END,  # Already formatted, just end
        },
    )

    # === Summary Compact Loops Back to End ===
    graph.add_edge("summary_compact", END)

    return graph.compile()


@traceable(name="ChatWorkflow", project_name=settings.langsmith_project)
async def run_chat(
    session,
    conversation_id: str,
    article_id: str,
    user_id: int,
    user_message: str,
    max_tokens: int = None,
    context_threshold: float = None,
    max_research_iterations: int = None,
) -> ChatState:
    """Execute the chat workflow with ReAct loop.

    This is the main entry point for processing a chat message.

    Args:
        session: Database session
        conversation_id: Conversation UUID
        article_id: Article UUID
        user_id: User ID
        user_message: User's message text
        max_tokens: Maximum context tokens (optional)
        context_threshold: Compression threshold (optional)
        max_research_iterations: Maximum ReAct iterations for researcher (optional)

    Returns:
        Final chat state with response
    """
    logger.info(
        "Starting chat workflow",
        conversation_id=conversation_id,
        article_id=article_id,
        user_id=user_id,
    )

    # Create initial state
    initial_state = create_initial_chat_state(
        conversation_id=conversation_id,
        article_id=article_id,
        user_id=user_id,
        user_message=user_message,
        max_tokens=max_tokens or settings.chat_max_tokens,
        context_threshold=context_threshold or settings.chat_context_threshold,
        max_research_iterations=max_research_iterations or settings.chat_max_research_iterations,
    )

    # Create the graph
    graph = create_chat_graph()

    try:
        # Execute with context for dependency injection
        context = ChatContext(
            session=session,
            conversation_id=conversation_id,
            article_id=article_id,
            user_id=user_id,
        )
        result = await graph.ainvoke(initial_state, context=context)

        logger.info(
            "Chat workflow completed",
            conversation_id=conversation_id,
            agent=result.get("current_agent"),
            routing_steps=len(result.get("routing_history", [])),
            research_iterations=result.get("research_iterations", 0),
            tokens=result.get("current_tokens", 0),
            errors=len(result.get("errors", [])),
        )

        return result

    except Exception as e:
        logger.error("Chat workflow failed", error=str(e), conversation_id=conversation_id)
        initial_state["errors"] = initial_state.get("errors", []) + [
            {"phase": "orchestration", "message": str(e)}
        ]
        initial_state["final_response"] = f"抱歉，处理过程中出现错误：{str(e)}"
        initial_state["is_complete"] = True
        return initial_state


# Pre-compiled graph for reuse
_chat_graph = None


def get_chat_graph():
    """Get cached chat graph instance.

    Returns:
        Compiled chat graph
    """
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = create_chat_graph()
    return _chat_graph