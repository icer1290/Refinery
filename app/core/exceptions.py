"""Core exceptions for the application."""

from typing import Optional


class NewsAggregatorError(Exception):
    """Base exception for the news aggregator."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class RSSParseError(NewsAggregatorError):
    """Error parsing RSS feed."""

    pass


class WebExtractionError(NewsAggregatorError):
    """Error extracting content from web page."""

    pass


class EmbeddingError(NewsAggregatorError):
    """Error generating embeddings."""

    pass


class LLMError(NewsAggregatorError):
    """Error calling LLM."""

    pass


class DatabaseError(NewsAggregatorError):
    """Database operation error."""

    pass


class WorkflowError(NewsAggregatorError):
    """Workflow execution error."""

    pass


class ExternalServiceError(NewsAggregatorError):
    """Error calling external services."""

    pass