"""Initial database schema.

Revision ID: 001
Revises: None
Create Date: 2024-01-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create news_articles table
    op.create_table(
        'news_articles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source_name', sa.String(255), nullable=False),
        sa.Column('source_url', sa.String(2048), unique=True, nullable=False),
        sa.Column('original_title', sa.Text, nullable=False),
        sa.Column('original_description', sa.Text, nullable=True),
        sa.Column('chinese_title', sa.Text, nullable=True),
        sa.Column('chinese_summary', sa.Text, nullable=True),
        sa.Column('full_content', sa.Text, nullable=True),
        sa.Column('total_score', sa.Float, nullable=True),
        sa.Column('industry_impact_score', sa.Float, nullable=True),
        sa.Column('milestone_score', sa.Float, nullable=True),
        sa.Column('attention_score', sa.Float, nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('reflection_retries', sa.Integer, default=0),
        sa.Column('reflection_passed', sa.Boolean, default=False),
        sa.Column('reflection_feedback', sa.Text, nullable=True),
        sa.Column('is_published', sa.Boolean, default=False),
        sa.Column('metadata', postgresql.JSON, nullable=True),
    )

    # Create indexes for news_articles
    op.create_index('ix_news_articles_source_url', 'news_articles', ['source_url'])
    op.create_index('ix_news_articles_published_at', 'news_articles', ['published_at'])
    op.create_index('ix_news_articles_total_score', 'news_articles', ['total_score'])

    # Create article_embeddings table
    op.create_table(
        'article_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('article_id', postgresql.UUID(as_uuid=True), unique=True, nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(1536), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create indexes for article_embeddings
    op.create_index('ix_article_embeddings_article_id', 'article_embeddings', ['article_id'], unique=True)
    op.create_index('ix_article_embeddings_content_hash', 'article_embeddings', ['content_hash'])
    op.create_index('ix_article_embeddings_vector', 'article_embeddings', ['embedding'], postgresql_using='ivfflat')

    # Create rss_feed_sources table
    op.create_table(
        'rss_feed_sources',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('url', sa.String(2048), unique=True, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('last_fetched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('fetch_error_count', sa.Integer, default=0),
        sa.Column('last_error', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create indexes for rss_feed_sources
    op.create_index('ix_rss_feed_sources_url', 'rss_feed_sources', ['url'])
    op.create_index('ix_rss_feed_sources_is_active', 'rss_feed_sources', ['is_active'])

    # Create workflow_runs table
    op.create_table(
        'workflow_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(50), default='running'),
        sa.Column('total_feeds_fetched', sa.Integer, default=0),
        sa.Column('total_articles_found', sa.Integer, default=0),
        sa.Column('total_articles_after_dedup', sa.Integer, default=0),
        sa.Column('total_articles_after_scoring', sa.Integer, default=0),
        sa.Column('total_articles_stored', sa.Integer, default=0),
        sa.Column('errors', postgresql.JSON, nullable=True),
        sa.Column('metadata', postgresql.JSON, nullable=True),
    )

    # Create indexes for workflow_runs
    op.create_index('ix_workflow_runs_started_at', 'workflow_runs', ['started_at'])


def downgrade() -> None:
    op.drop_table('workflow_runs')
    op.drop_table('rss_feed_sources')
    op.drop_table('article_embeddings')
    op.drop_table('news_articles')
    op.execute('DROP EXTENSION IF EXISTS vector')