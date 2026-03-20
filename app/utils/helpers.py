"""Utility functions and helpers."""

from datetime import datetime
from typing import Any


def format_datetime(dt: datetime | None) -> str | None:
    """Format datetime to ISO 8601 string.

    Args:
        dt: Datetime to format

    Returns:
        ISO formatted string or None
    """
    if dt is None:
        return None
    return dt.isoformat()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value.

    Args:
        data: Dictionary to get value from
        *keys: Keys to traverse
        default: Default value if key not found

    Returns:
        Value or default
    """
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result