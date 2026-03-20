"""LangGraph workflow state definition."""

import operator
from datetime import datetime
from typing import Annotated, Any

from typing_extensions import TypedDict


class ArticleCandidate(TypedDict, total=False):
    """Represents a candidate article in the workflow.

    Fields:
        source_name: Name of the RSS feed source
        source_url: URL of the original article
        original_title: Original article title
        original_description: Article description/summary
        published_at: Publication timestamp
        content_hash: Hash for deduplication
        embedding: Vector embedding for semantic similarity
        full_content: Extracted full article content
        industry_impact_score: Score for industry impact (0-10)
        milestone_score: Score for milestone significance (0-10)
        attention_score: Score for attention value (0-10)
        total_score: Weighted total score (0-10)
        chinese_title: Translated Chinese title
        chinese_summary: Translated Chinese summary
        entities_preserved: List of entities kept in original language
        reflection_passed: Whether reflection check passed
        reflection_feedback: Feedback from reflection
        reflection_retries: Number of reflection retries
    """

    source_name: str
    source_url: str
    original_title: str
    original_description: str | None
    published_at: datetime | None
    content_hash: str
    embedding: list[float] | None
    full_content: str | None
    industry_impact_score: float | None
    milestone_score: float | None
    attention_score: float | None
    total_score: float | None
    chinese_title: str | None
    chinese_summary: str | None
    entities_preserved: list[str]
    reflection_passed: bool | None
    reflection_feedback: str | None
    reflection_retries: int


class WorkflowError(TypedDict):
    """Represents an error in the workflow."""

    phase: str
    message: str
    details: dict[str, Any] | None


class WorkflowState(TypedDict):
    """State for the news aggregation workflow.

    Fields:
        run_id: Unique identifier for this workflow run
        feed_urls: Specific feeds to fetch (optional)
        score_threshold: Override score threshold (optional)
        force_reprocess: Force reprocessing of existing articles

        raw_articles: Articles fetched from RSS feeds
        deduplicated_articles: Articles after deduplication
        scored_articles: Articles after scoring
        processed_articles: Articles after content processing
        final_articles: Articles ready for storage
        stored_article_ids: IDs of stored articles

        errors: Errors encountered during processing
        current_phase: Current processing phase

        Statistics:
        total_feeds_fetched: Number of feeds fetched
        total_articles_found: Articles found from feeds
        total_articles_after_dedup: Articles after deduplication
        total_articles_after_scoring: Articles after scoring
        total_articles_stored: Articles stored
    """

    # Configuration
    run_id: str
    feed_urls: list[str] | None
    score_threshold: float | None
    force_reprocess: bool

    # Article collections (using Annotated for list merging)
    raw_articles: Annotated[list[ArticleCandidate], operator.add]
    deduplicated_articles: list[ArticleCandidate]
    scored_articles: list[ArticleCandidate]
    processed_articles: list[ArticleCandidate]
    final_articles: list[ArticleCandidate]
    stored_article_ids: list[str]

    # Errors (using Annotated for list merging)
    errors: Annotated[list[WorkflowError], operator.add]

    # Current phase tracking
    current_phase: str

    # Statistics
    total_feeds_fetched: int
    total_articles_found: int
    total_articles_after_dedup: int
    total_articles_after_scoring: int
    total_articles_stored: int


def create_initial_state(
    run_id: str,
    feed_urls: list[str] | None = None,
    score_threshold: float | None = None,
    force_reprocess: bool = False,
) -> WorkflowState:
    """Create initial workflow state.

    Args:
        run_id: Unique identifier for this run
        feed_urls: Specific feeds to fetch (optional)
        score_threshold: Override score threshold (optional)
        force_reprocess: Force reprocessing existing articles

    Returns:
        Initial workflow state
    """
    return WorkflowState(
        run_id=run_id,
        feed_urls=feed_urls,
        score_threshold=score_threshold,
        force_reprocess=force_reprocess,
        raw_articles=[],
        deduplicated_articles=[],
        scored_articles=[],
        processed_articles=[],
        final_articles=[],
        stored_article_ids=[],
        errors=[],
        current_phase="initialized",
        total_feeds_fetched=0,
        total_articles_found=0,
        total_articles_after_dedup=0,
        total_articles_after_scoring=0,
        total_articles_stored=0,
    )