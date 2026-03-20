"""Deep search state definition for ReAct workflow."""

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class ToolCall(TypedDict):
    """Represents a tool call in the ReAct loop."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    iteration: int


class CollectedInfo(TypedDict):
    """Represents collected information from tools."""

    source: str  # "vector_search" or "web_search"
    content: str
    relevance: str
    metadata: dict[str, Any]


class DeepSearchState(TypedDict):
    """State for the deep search ReAct workflow.

    Fields:
        article_id: ID of the article to analyze
        article: Article details from database

        collected_info: Information collected from tools
        tool_history: History of tool calls
        current_thought: LLM's current reasoning

        max_iterations: Maximum ReAct iterations
        current_iteration: Current iteration number
        is_complete: Whether search is complete
        should_continue: Whether to continue the loop

        final_report: Generated deep tracking report
        errors: Errors encountered during processing
    """

    # Input
    article_id: str
    article: dict[str, Any] | None

    # ReAct loop state
    collected_info: Annotated[list[CollectedInfo], operator.add]
    tool_history: Annotated[list[ToolCall], operator.add]
    current_thought: str

    # Loop control
    max_iterations: int
    current_iteration: int
    is_complete: bool
    should_continue: bool

    # Output
    final_report: str
    errors: Annotated[list[dict], operator.add]


def create_initial_deep_search_state(
    article_id: str,
    max_iterations: int = 5,
) -> DeepSearchState:
    """Create initial deep search state.

    Args:
        article_id: ID of the article to analyze
        max_iterations: Maximum ReAct iterations

    Returns:
        Initial deep search state
    """
    return DeepSearchState(
        article_id=article_id,
        article=None,
        collected_info=[],
        tool_history=[],
        current_thought="",
        max_iterations=max_iterations,
        current_iteration=0,
        is_complete=False,
        should_continue=True,
        final_report="",
        errors=[],
    )