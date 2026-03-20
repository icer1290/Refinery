"""SQLAlchemy ORM models for database tables."""

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    ForeignKey,
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class NewsArticle(Base):
    """News article model."""

    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    original_title: Mapped[str] = mapped_column(Text, nullable=False)
    original_description: Mapped[str | None] = mapped_column(Text)
    chinese_title: Mapped[str | None] = mapped_column(Text)
    chinese_summary: Mapped[str | None] = mapped_column(Text)
    full_content: Mapped[str | None] = mapped_column(Text)

    # Scores
    total_score: Mapped[float | None] = mapped_column(Float)
    industry_impact_score: Mapped[float | None] = mapped_column(Float)
    milestone_score: Mapped[float | None] = mapped_column(Float)
    attention_score: Mapped[float | None] = mapped_column(Float)

    # Timestamps
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Reflection
    reflection_retries: Mapped[int] = mapped_column(Integer, default=0)
    reflection_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    reflection_feedback: Mapped[str | None] = mapped_column(Text)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)

    # DeepSearch Report
    deepsearch_report: Mapped[str | None] = mapped_column(Text)
    deepsearch_performed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationship to embeddings (one-to-many: summary + chunks)
    embeddings: Mapped[list["ArticleEmbedding"]] = relationship(
        "ArticleEmbedding", back_populates="article", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_news_articles_source_url", source_url),
        Index("ix_news_articles_published_at", published_at),
        Index("ix_news_articles_total_score", total_score),
    )


class ArticleEmbedding(Base):
    """Article embedding model with pgvector.

    Supports both summary embeddings (title + description) and
    chunk embeddings (sections of full_content).
    """

    __tablename__ = "article_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Chunk-specific fields for RAG
    chunk_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_text: Mapped[str | None] = mapped_column(Text)
    chunk_start: Mapped[int | None] = mapped_column(Integer)  # Character offset in original text
    chunk_end: Mapped[int | None] = mapped_column(Integer)  # Character offset in original text
    embedding_type: Mapped[str] = mapped_column(
        String(20), default="summary", nullable=False
    )  # 'summary' | 'chunk'

    # Note: chunk_tsv is a TSVECTOR column managed by database trigger
    # It is NOT included in the ORM model to avoid type conflicts

    # Relationship
    article: Mapped["NewsArticle"] = relationship(
        "NewsArticle", back_populates="embeddings"
    )

    __table_args__ = (
        Index("ix_article_embeddings_article_chunk", article_id, chunk_number, unique=True),
        Index("ix_article_embeddings_content_hash", content_hash),
        # Note: chunk_tsv GIN index is created in migration, not managed by ORM
    )


class RSSFeedSource(Base):
    """RSS feed source configuration."""

    __tablename__ = "rss_feed_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetch_error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_rss_feed_sources_url", url),
        Index("ix_rss_feed_sources_is_active", is_active),
    )


class WorkflowRun(Base):
    """Workflow execution record."""

    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(50), default="running")

    # Statistics
    total_feeds_fetched: Mapped[int] = mapped_column(Integer, default=0)
    total_articles_found: Mapped[int] = mapped_column(Integer, default=0)
    total_articles_after_dedup: Mapped[int] = mapped_column(Integer, default=0)
    total_articles_after_scoring: Mapped[int] = mapped_column(Integer, default=0)
    total_articles_stored: Mapped[int] = mapped_column(Integer, default=0)

    # Error tracking
    errors: Mapped[list | None] = mapped_column(JSON)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)

    __table_args__ = (Index("ix_workflow_runs_started_at", started_at),)


class GraphEntity(Base):
    """Graph entity model for GraphRAG.

    Represents extracted entities (people, organizations, technologies, etc.)
    from news articles.
    """

    __tablename__ = "graph_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(500), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # PERSON, ORGANIZATION, etc.
    description: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    article_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_graph_entities_name", name),
        Index("ix_graph_entities_type", type),
        Index("ix_graph_entities_canonical_name", canonical_name, unique=True),
    )


class GraphRelationship(Base):
    """Graph relationship model for GraphRAG.

    Represents relationships between entities extracted from articles.
    """

    __tablename__ = "graph_relationships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    article_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    evidence_texts: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_graph_relationships_source", source_entity_id),
        Index("ix_graph_relationships_target", target_entity_id),
        Index("ix_graph_relationships_type", relation_type),
        Index(
            "ix_graph_relationships_unique",
            source_entity_id, target_entity_id, relation_type,
            unique=True
        ),
    )


class GraphCommunity(Base):
    """Graph community model for GraphRAG.

    Represents clusters of related entities detected by Leiden algorithm.
    """

    __tablename__ = "graph_communities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    entity_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    hub_entity_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    article_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    level: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_graph_communities_name", name),
        Index("ix_graph_communities_level", level),
    )


class GraphBuilderRun(Base):
    """Graph builder execution record.

    Tracks the history of graph building operations.
    """

    __tablename__ = "graph_builder_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(50), default="running")
    article_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    entities_extracted: Mapped[int] = mapped_column(Integer, default=0)
    relationships_extracted: Mapped[int] = mapped_column(Integer, default=0)
    communities_detected: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list | None] = mapped_column(JSON)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)

    __table_args__ = (
        Index("ix_graph_builder_runs_started_at", started_at),
        Index("ix_graph_builder_runs_status", status),
    )


class DeepGraphAnalysis(Base):
    """DeepGraph analysis record with user tracking.

    Stores analysis results when a user requests a DeepGraph analysis.
    """

    __tablename__ = "deepgraph_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    article_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False
    )
    report: Mapped[str | None] = mapped_column(Text)
    visualization_data: Mapped[dict | None] = mapped_column(JSON)
    max_hops: Mapped[int] = mapped_column(Integer, default=2)
    expansion_limit: Mapped[int] = mapped_column(Integer, default=50)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_deepgraph_analyses_user_id", user_id),
        Index("ix_deepgraph_analyses_created_at", created_at),
    )