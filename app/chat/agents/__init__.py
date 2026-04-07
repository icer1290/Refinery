"""Chat specialist agents."""

from app.chat.agents.base import BaseChatAgent
from app.chat.agents.supervisor import SupervisorAgent
from app.chat.agents.explainer import ExplainerAgent
from app.chat.agents.researcher import ResearcherAgent
from app.chat.agents.fact_checker import FactCheckerAgent

__all__ = [
    "BaseChatAgent",
    "SupervisorAgent",
    "ExplainerAgent",
    "ResearcherAgent",
    "FactCheckerAgent",
]


# Agent instances (singleton pattern)
_supervisor_agent: SupervisorAgent | None = None
_explainer_agent: ExplainerAgent | None = None
_researcher_agent: ResearcherAgent | None = None
_fact_checker_agent: FactCheckerAgent | None = None


def get_supervisor_agent() -> SupervisorAgent:
    """Get supervisor agent instance."""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = SupervisorAgent()
    return _supervisor_agent


def get_explainer_agent() -> ExplainerAgent:
    """Get explainer agent instance."""
    global _explainer_agent
    if _explainer_agent is None:
        _explainer_agent = ExplainerAgent()
    return _explainer_agent


def get_researcher_agent() -> ResearcherAgent:
    """Get researcher agent instance."""
    global _researcher_agent
    if _researcher_agent is None:
        _researcher_agent = ResearcherAgent()
    return _researcher_agent


def get_fact_checker_agent() -> FactCheckerAgent:
    """Get fact-checker agent instance."""
    global _fact_checker_agent
    if _fact_checker_agent is None:
        _fact_checker_agent = FactCheckerAgent()
    return _fact_checker_agent


def get_agent_by_name(agent_name: str) -> BaseChatAgent | None:
    """Get agent by name.

    Args:
        agent_name: Agent name (researcher, explainer, fact_checker)

    Returns:
        Agent instance or None
    """
    agents = {
        "supervisor": get_supervisor_agent,
        "explainer": get_explainer_agent,
        "researcher": get_researcher_agent,
        "fact_checker": get_fact_checker_agent,
    }
    getter = agents.get(agent_name)
    return getter() if getter else None