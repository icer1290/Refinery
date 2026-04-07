"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# === Article Schemas ===


class ArticleBase(BaseModel):
    """Base article schema."""

    source_name: str
    source_url: str
    original_title: str
    original_description: Optional[str] = None
    published_at: Optional[datetime] = None


class ArticleCreate(ArticleBase):
    """Schema for creating an article."""

    pass


class ArticleResponse(ArticleBase):
    """Schema for article response."""

    id: UUID
    chinese_title: Optional[str] = None
    chinese_summary: Optional[str] = None
    full_content: Optional[str] = None
    total_score: Optional[float] = None
    industry_impact_score: Optional[float] = None
    milestone_score: Optional[float] = None
    attention_score: Optional[float] = None
    processed_at: datetime
    reflection_retries: int = 0
    reflection_passed: bool = False
    is_published: bool = False
    metadata_: Optional[dict[str, Any]] = None

    # DeepSearch Report
    deepsearch_report: Optional[str] = None
    deepsearch_performed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ArticleListResponse(BaseModel):
    """Schema for article list response."""

    articles: list[ArticleResponse]
    total: int
    page: int
    page_size: int


# === RSS Feed Schemas ===


class RSSFeedBase(BaseModel):
    """Base RSS feed schema."""

    name: str
    url: str
    is_active: bool = True


class RSSFeedCreate(RSSFeedBase):
    """Schema for creating an RSS feed."""

    pass


class RSSFeedResponse(RSSFeedBase):
    """Schema for RSS feed response."""

    id: UUID
    last_fetched_at: Optional[datetime] = None
    fetch_error_count: int = 0
    last_error: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RSSFeedListResponse(BaseModel):
    """Schema for RSS feed list response."""

    feeds: list[RSSFeedResponse]
    total: int


# === Workflow Schemas ===


class WorkflowRunResponse(BaseModel):
    """Schema for workflow run response."""

    id: UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    total_feeds_fetched: int = 0
    total_articles_found: int = 0
    total_articles_after_dedup: int = 0
    total_articles_after_scoring: int = 0
    total_articles_stored: int = 0
    errors: Optional[list[dict[str, Any]]] = None
    metadata_: Optional[dict[str, Any]] = None

    model_config = {"from_attributes": True}


class WorkflowRunListResponse(BaseModel):
    """Schema for workflow run list response."""

    runs: list[WorkflowRunResponse]
    total: int


class WorkflowTriggerRequest(BaseModel):
    """Schema for triggering a workflow run."""

    feed_urls: Optional[list[str]] = Field(
        default=None, description="Specific RSS feeds to fetch, or None for all"
    )
    score_threshold: Optional[float] = Field(
        default=None, description="Override default score threshold"
    )
    force: bool = Field(
        default=False, description="Force reprocessing of existing articles"
    )
    hours_back: Optional[int] = Field(
        default=None,
        ge=1,
        le=168,  # Max 7 days
        description="Only fetch articles from the last N hours (default: 24)",
    )


# === Scoring Schemas ===


class ScoringResult(BaseModel):
    """Schema for article scoring result."""

    industry_impact_score: float = Field(ge=0, le=10)
    milestone_score: float = Field(ge=0, le=10)
    attention_score: float = Field(ge=0, le=10)
    total_score: float = Field(ge=0, le=10)
    reasoning: Optional[str] = None


# === Translation Schemas ===


class TranslationResult(BaseModel):
    """Schema for translation result."""

    chinese_title: str
    chinese_summary: str
    entities_preserved: list[str] = []


# === Reflection Schemas ===


class ReflectionResult(BaseModel):
    """Schema for reflection result."""

    passed: bool
    feedback: Optional[str] = None
    issues: list[str] = []


# === Error Schemas ===


class ErrorResponse(BaseModel):
    """Schema for error response."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# === Deep Search Schemas ===


class DeepSearchRequest(BaseModel):
    """Schema for deep search request."""

    article_id: str = Field(..., description="UUID of the article to analyze")
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of ReAct iterations",
    )


class ToolCallInfo(BaseModel):
    """Schema for tool call information."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    iteration: int


class CollectedInfoResponse(BaseModel):
    """Schema for collected information."""

    source: str
    content: str
    relevance: str
    metadata: dict[str, Any]


class DeepSearchResponse(BaseModel):
    """Schema for deep search response."""

    article_id: str
    article_title: str
    final_report: str
    tools_used: list[ToolCallInfo]
    collected_info: list[CollectedInfoResponse]
    iterations: int
    is_complete: bool
    errors: list[dict[str, Any]] = []


# === DeepGraph Schemas ===


class DeepGraphRequest(BaseModel):
    """Schema for DeepGraph analysis request."""

    user_id: int | None = Field(
        default=None,
        description="User ID from api-server (optional)",
    )
    article_ids: list[str] = Field(
        ...,
        description="List of article UUIDs to analyze",
        min_length=1,
        max_length=20,
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum hops for graph expansion",
    )
    expansion_limit: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Maximum entities to add through expansion",
    )


class GraphNodeResponse(BaseModel):
    """Schema for graph node in visualization."""

    id: str
    label: str
    type: str
    description: Optional[str] = None
    mention_count: int = 1
    article_count: int = 1
    is_expanded: bool = False


class GraphEdgeResponse(BaseModel):
    """Schema for graph edge in visualization."""

    id: str
    source: str
    target: str
    relation_type: str
    description: Optional[str] = None
    weight: float = 1.0
    article_count: int = 1
    is_expanded: bool = False


class CommunityResponse(BaseModel):
    """Schema for community in visualization."""

    id: str
    name: str
    summary: Optional[str] = None
    entity_count: int = 0
    hub_entity: Optional[str] = None
    article_ids: list[str] = []


class VisualizationStats(BaseModel):
    """Schema for visualization statistics."""

    total_entities: int = 0
    seed_entities: int = 0
    expanded_entities: int = 0
    total_relationships: int = 0
    total_communities: int = 0


class VisualizationData(BaseModel):
    """Schema for complete visualization data."""

    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    communities: list[CommunityResponse]
    stats: VisualizationStats


class DeepGraphResponse(BaseModel):
    """Schema for DeepGraph analysis response."""

    article_ids: list[str]
    graph_nodes: list[GraphNodeResponse]
    graph_edges: list[GraphEdgeResponse]
    communities: list[CommunityResponse]
    report: str
    visualization_data: VisualizationData
    errors: list[dict[str, Any]] = []


class EntitySearchRequest(BaseModel):
    """Schema for entity search request."""

    query: str = Field(..., min_length=1, description="Search query for entity name")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")


class EntityResponse(BaseModel):
    """Schema for entity detail response."""

    id: UUID
    name: str
    canonical_name: str
    type: str
    description: Optional[str] = None
    mention_count: int = 1
    aliases: list[str] = []
    article_ids: list[UUID] = []
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class EntityListResponse(BaseModel):
    """Schema for entity list response."""

    entities: list[EntityResponse]
    total: int


class GraphBuilderRunRequest(BaseModel):
    """Schema for triggering graph builder."""

    article_ids: list[str] = Field(
        ...,
        description="List of article UUIDs to build graph from",
        min_length=1,
    )


class GraphBuilderRunResponse(BaseModel):
    """Schema for graph builder run response."""

    id: UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    article_ids: list[UUID] = []
    entities_extracted: int = 0
    relationships_extracted: int = 0
    communities_detected: int = 0
    errors: Optional[list[dict[str, Any]]] = None

    model_config = {"from_attributes": True}


# === DeepGraph Analysis Storage Schemas ===


class DeepGraphAnalysisResponse(BaseModel):
    """Schema for stored DeepGraph analysis response."""

    id: UUID
    user_id: int
    article_ids: list[UUID]
    report: str | None
    visualization_data: dict | None
    max_hops: int
    expansion_limit: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DeepGraphAnalysisListResponse(BaseModel):
    """Schema for list of DeepGraph analyses."""

    analyses: list[DeepGraphAnalysisResponse]
    total: int


# === Chat Schemas ===


class ConversationCreateRequest(BaseModel):
    """Schema for creating a conversation."""

    article_id: str = Field(..., description="Article UUID")
    user_id: int = Field(..., description="User ID from api-server")
    title: Optional[str] = Field(None, max_length=500, description="Optional conversation title")


class ConversationResponse(BaseModel):
    """Schema for conversation response."""

    id: UUID
    article_id: UUID
    user_id: int
    title: Optional[str] = None
    status: str = "active"
    message_count: int = 0
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """Schema for conversation list response."""

    conversations: list[ConversationResponse]
    total: int
    page: int
    page_size: int


class CitationInfo(BaseModel):
    """Schema for citation information."""

    source_type: str  # article, web_search, knowledge_graph
    source_id: Optional[str] = None
    source_name: str
    content_snippet: str
    url: Optional[str] = None
    relevance_score: Optional[float] = None


class ChatMessageResponse(BaseModel):
    """Schema for individual chat message."""

    id: UUID
    role: str  # user, assistant, system, tool
    content: str
    agent_name: Optional[str] = None
    citations: Optional[list[CitationInfo]] = None
    tokens_used: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(BaseModel):
    """Schema for conversation details with article context."""

    id: UUID
    article_id: UUID
    user_id: int
    title: Optional[str] = None
    status: str = "active"
    message_count: int = 0
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
    article_title: Optional[str] = None
    article_summary: Optional[str] = None
    has_deepsearch: bool = False

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response."""

    conversation_id: str
    messages: list[ChatMessageResponse]
    total: int
    has_more: bool


class ChatRequest(BaseModel):
    """Schema for chat message request."""

    conversation_id: str = Field(..., description="Conversation UUID")
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    include_deepsearch: bool = Field(default=True, description="Include deepsearch context")


class ToolCallInfo(BaseModel):
    """Schema for tool call information in chat response."""

    tool_name: str
    tool_input: dict[str, Any]
    timestamp: str


class ChatResponse(BaseModel):
    """Schema for chat response."""

    conversation_id: str
    message_id: str
    response: str
    agent_used: str  # researcher, explainer, fact_checker
    citations: list[CitationInfo] = []
    tool_calls: list[ToolCallInfo] = []
    tokens_used: int = 0
    created_at: datetime


class UserProfileMemory(BaseModel):
    """Schema for extracted user profile memory."""

    interests: list[str] = []
    expertise_level: str = "intermediate"  # beginner, intermediate, advanced
    preferred_format: str = "detailed"  # detailed, brief, technical
    follow_up_patterns: list[str] = []


class ConversationStateMemory(BaseModel):
    """Schema for extracted conversation state memory."""

    current_topic: Optional[str] = None
    information_gathered: list[str] = []
    open_questions: list[str] = []
    last_agent: Optional[str] = None