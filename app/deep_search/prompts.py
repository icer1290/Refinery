"""Prompt templates for deep search ReAct workflow.

.. deprecated::
    This module is deprecated. Use `app.prompts` instead.

    Migration guide:
    - REACT_SYSTEM_PROMPT → get_prompt("deep_search.react_system")
    - REACT_USER_PROMPT → get_prompt("deep_search.react_user")
    - CONCLUSION_PROMPT → get_prompt("deep_search.conclusion")

    The helper function `format_collected_info` is available from
    `app.prompts.formatters`.
"""

import warnings

from app.prompts import get_prompt
from app.prompts.formatters import format_collected_info

# Emit deprecation warning on import
warnings.warn(
    "app.deep_search.prompts is deprecated. Use app.prompts instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility: expose prompts as module-level constants
# These are formatted versions of the templates

REACT_SYSTEM_PROMPT = get_prompt("deep_search.react_system").template
REACT_USER_PROMPT = get_prompt("deep_search.react_user").template
CONCLUSION_PROMPT = get_prompt("deep_search.conclusion").template

__all__ = [
    "REACT_SYSTEM_PROMPT",
    "REACT_USER_PROMPT",
    "CONCLUSION_PROMPT",
    "format_collected_info",
]