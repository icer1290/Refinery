"""State definitions for the chat workflow using LangGraph TypedDict."""

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict


class ChatMessageContent(TypedDict):
    """Single message in chat history."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    agent_name: Optional[str]  # Which agent generated this
    tool_calls: Optional[list[dict[str, Any]]]  # Tool call records
    tool_results: Optional[list[dict[str, Any]]]  # Tool result records
    citations: Optional[list[dict[str, Any]]]  # Citation references
    created_at: str  # ISO datetime
    tokens_used: int


class CollectedInfo(TypedDict):
    """Information collected by researcher during ReAct loop."""

    source: str  # Tool name
    content: str  # Tool output
    relevance: str  # Why this info is relevant
    metadata: dict[str, Any]  # Tool input params


class ResearchThought(TypedDict):
    """Single thought/reasoning step in researcher ReAct loop."""

    thought: str  # Reasoning
    action: str  # Tool name or "conclude"
    action_input: dict[str, Any]  # Tool input
    iteration: int  # Which iteration


class FactCheckResult(TypedDict):
    """Result of fact checking."""

    passed: bool
    reason: str  # Explanation if failed
    issues: list[str]  # List of specific issues found


class CompactBoundaryMessage(TypedDict):
    """Summary compact boundary message replacing older messages.

    Generated when context exceeds 70% threshold.
    """

    summary: str  # Conversation summary
    key_entities: list[str]  # Important entities mentioned
    key_citations: list[dict[str, Any]]  # Important citations
    tool_summary: str  # Summary of tool calls made
    period_start: str  # ISO datetime of first message summarized
    period_end: str  # ISO datetime of last message summarized
    messages_replaced: int  # Number of messages replaced by this summary


class ExtractedMemory(TypedDict):
    """Memory extraction result."""

    memory_type: Literal["user_profile", "conversation_state", "key_citations"]
    content: dict[str, Any]  # Memory content
    extraction_iteration: int  # Which extraction cycle (every 5 tool calls)


class ToolCallRecord(TypedDict):
    """Record of a tool call for tracking."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    timestamp: str  # ISO datetime
    agent: str  # Which agent made the call


class SpecialistResponse(TypedDict):
    """Response from a specialist agent."""

    agent_name: str  # researcher, explainer, fact_checker
    response: str  # Generated response text
    citations: list[dict[str, Any]]  # Citations used
    tool_calls: list[ToolCallRecord]  # Tool calls made
    confidence: float  # Confidence score (0.0-1.0)


class RoutingDecision(TypedDict):
    """Supervisor routing decision."""

    agent: Literal["researcher", "explainer", "fact_checker"]
    reasoning: str  # Why this agent was chosen
    query_type: str  # Classification of the query


class ChatState(TypedDict):
    """State for the multi-turn chat workflow with hub-and-spoke architecture.

    Flow:
    1. User message -> Supervisor evaluates completeness
    2. Supervisor routes to:
       - Researcher (if need more info) -> ReAct loop -> back to Supervisor
       - Explainer (if have enough info) -> generates response -> Supervisor
    3. After Explainer, routes to FactChecker
       - If pass: END
       - If fail: add reason to context, back to Supervisor

    Hub-and-Spoke Pattern:
    - Supervisor is the central hub
    - All agents return to Supervisor after execution
    - Supervisor decides next step based on state
    """

    # === Input ===
    conversation_id: str
    article_id: str
    user_id: int
    user_message: str

    # === Article Context ===
    article_context: Optional[dict[str, Any]]  # Title, summary, content, source
    deepsearch_result: Optional[str]  # DeepSearch report if available

    # === Conversation State (with list merging) ===
    messages: Annotated[list[ChatMessageContent], operator.add]
    compact_boundaries: Annotated[list[CompactBoundaryMessage], operator.add]

    # === Research State (ReAct Loop) ===
    info_completeness: float  # 0.0-1.0, how complete is the information
    research_iterations: int  # Current ReAct iteration
    max_research_iterations: int  # Max iterations allowed
    collected_info: Annotated[list[CollectedInfo], operator.add]  # Info from tools
    current_thought: Optional[ResearchThought]  # Current reasoning step
    pending_tool: Optional[str]  # Tool to execute next
    pending_tool_input: Optional[dict[str, Any]]  # Input for pending tool

    # === Agent Routing (Hub-and-Spoke) ===
    current_agent: Optional[str]  # Current agent being executed
    routing_history: Annotated[list[str], operator.add]  # Track agent visits
    generated_response: Optional[str]  # Response from Explainer

    # === Fact Check State ===
    fact_check_result: Optional[FactCheckResult]  # Result of fact check

    # === Tool Tracking ===
    tool_history: Annotated[list[ToolCallRecord], operator.add]
    tool_call_count: int  # Total tool calls for extraction trigger

    # === Memory Management ===
    extracted_memories: Annotated[list[ExtractedMemory], operator.add]
    user_profile: Optional[dict[str, Any]]  # User interests, expertise, preferences
    conversation_state: Optional[dict[str, Any]]  # Current topic, gathered info
    key_citations: Optional[list[dict[str, Any]]]  # Key citations from conversation

    # === Context Management ===
    current_tokens: int  # Estimated token count
    max_tokens: int  # Maximum context tokens
    context_threshold: float  # Compression threshold (0.7)

    # === Output ===
    final_response: str
    final_citations: list[dict[str, Any]]

    # === Control ===
    is_complete: bool
    should_compress: bool  # Flag for summary compression
    errors: Annotated[list[dict[str, Any]], operator.add]


def create_initial_chat_state(
    conversation_id: str,
    article_id: str,
    user_id: int,
    user_message: str,
    max_tokens: int = 1000,
    context_threshold: float = 0.7,
    max_research_iterations: int = 5,
) -> ChatState:
    """Create initial chat state.

    Args:
        conversation_id: Conversation UUID
        article_id: Article UUID
        user_id: User ID
        user_message: User's message text
        max_tokens: Maximum context tokens
        context_threshold: Compression threshold
        max_research_iterations: Maximum ReAct iterations for researcher

    Returns:
        Initial ChatState dict
    """
    return ChatState(
        # Input
        conversation_id=conversation_id,
        article_id=article_id,
        user_id=user_id,
        user_message=user_message,
        # Article context (loaded by load_context_node)
        article_context=None,
        deepsearch_result=None,
        # Conversation state
        messages=[],
        compact_boundaries=[],
        # Research state (ReAct loop)
        info_completeness=0.0,
        research_iterations=0,
        max_research_iterations=max_research_iterations,
        collected_info=[],
        current_thought=None,
        pending_tool=None,
        pending_tool_input=None,
        # Agent routing (hub-and-spoke)
        current_agent=None,
        routing_history=[],
        generated_response=None,
        # Fact check state
        fact_check_result=None,
        # Tool tracking
        tool_history=[],
        tool_call_count=0,
        # Memory management
        extracted_memories=[],
        user_profile=None,
        conversation_state=None,
        key_citations=None,
        # Context management
        current_tokens=0,
        max_tokens=max_tokens,
        context_threshold=context_threshold,
        # Output
        final_response="",
        final_citations=[],
        # Control
        is_complete=False,
        should_compress=False,
        errors=[],
    )


def create_user_message(
    content: str,
) -> ChatMessageContent:
    """Create a user message dict.

    Args:
        content: User message text

    Returns:
        ChatMessageContent dict
    """
    return ChatMessageContent(
        role="user",
        content=content,
        agent_name=None,
        tool_calls=None,
        tool_results=None,
        citations=None,
        created_at=datetime.now().isoformat(),
        tokens_used=len(content) // 4,  # Rough estimate: 1 token ~= 4 chars
    )


def create_assistant_message(
    content: str,
    agent_name: str,
    citations: Optional[list[dict[str, Any]]] = None,
    tool_calls: Optional[list[ToolCallRecord]] = None,
    tokens_used: int = 0,
) -> ChatMessageContent:
    """Create an assistant message dict.

    Args:
        content: Response text
        agent_name: Agent that generated the response
        citations: Citations used
        tool_calls: Tool calls made
        tokens_used: Token count

    Returns:
        ChatMessageContent dict
    """
    return ChatMessageContent(
        role="assistant",
        content=content,
        agent_name=agent_name,
        tool_calls=tool_calls,
        tool_results=None,
        citations=citations,
        created_at=datetime.now().isoformat(),
        tokens_used=tokens_used,
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Rough estimate: 1 token ~= 4 characters for English.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4