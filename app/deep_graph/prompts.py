"""Prompt templates for GraphRAG workflows.

.. deprecated::
    This module is deprecated. Use `app.prompts` instead.

    Migration guide:
    - ENTITY_EXTRACTION_SYSTEM_PROMPT → get_prompt("deep_graph.entity_extraction_system")
    - ENTITY_EXTRACTION_USER_PROMPT → get_prompt("deep_graph.entity_extraction_user")
    - RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT → get_prompt("deep_graph.relationship_extraction_system")
    - RELATIONSHIP_EXTRACTION_USER_PROMPT → get_prompt("deep_graph.relationship_extraction_user")
    - COMMUNITY_SUMMARY_PROMPT → get_prompt("deep_graph.community_summary")
    - DEEP_GRAPH_REPORT_PROMPT → get_prompt("deep_graph.report")

    Constants and helper functions are available from `app.prompts.formatters`.
"""

import warnings

from app.prompts import get_prompt
from app.prompts.formatters import (
    ENTITY_TYPE_DESCRIPTIONS,
    ENTITY_TYPES,
    format_articles_for_report,
    format_entities_for_prompt,
    format_entity_types,
    format_graph_for_report,
)

# Emit deprecation warning on import
warnings.warn(
    "app.deep_graph.prompts is deprecated. Use app.prompts instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility: expose prompts as module-level constants
ENTITY_EXTRACTION_SYSTEM_PROMPT = get_prompt("deep_graph.entity_extraction_system").template
ENTITY_EXTRACTION_USER_PROMPT = get_prompt("deep_graph.entity_extraction_user").template
RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT = get_prompt("deep_graph.relationship_extraction_system").template
RELATIONSHIP_EXTRACTION_USER_PROMPT = get_prompt("deep_graph.relationship_extraction_user").template
COMMUNITY_SUMMARY_PROMPT = get_prompt("deep_graph.community_summary").template
DEEP_GRAPH_REPORT_PROMPT = get_prompt("deep_graph.report").template


# Backward compatibility: expose helper functions
def get_entity_extraction_prompts() -> tuple[str, str]:
    """Get entity extraction prompts.

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
        entity_types_desc=format_entity_types()
    )
    return system_prompt, ENTITY_EXTRACTION_USER_PROMPT


def get_relationship_extraction_prompts() -> tuple[str, str]:
    """Get relationship extraction prompts.

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    return RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT, RELATIONSHIP_EXTRACTION_USER_PROMPT


__all__ = [
    # Constants
    "ENTITY_TYPES",
    "ENTITY_TYPE_DESCRIPTIONS",
    # Prompts
    "ENTITY_EXTRACTION_SYSTEM_PROMPT",
    "ENTITY_EXTRACTION_USER_PROMPT",
    "RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT",
    "RELATIONSHIP_EXTRACTION_USER_PROMPT",
    "COMMUNITY_SUMMARY_PROMPT",
    "DEEP_GRAPH_REPORT_PROMPT",
    # Helper functions
    "format_entity_types",
    "get_entity_extraction_prompts",
    "get_relationship_extraction_prompts",
    "format_entities_for_prompt",
    "format_graph_for_report",
    "format_articles_for_report",
]