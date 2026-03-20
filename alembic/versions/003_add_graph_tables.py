"""Add graph tables for GraphRAG system.

Revision ID: 003
Revises: 002
Create Date: 2026-03-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create graph_entities table
    op.create_table(
        'graph_entities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('canonical_name', sa.String(500), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column('article_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default='{}'),
        sa.Column('mention_count', sa.Integer, nullable=False, server_default='1'),
        sa.Column('aliases', postgresql.ARRAY(sa.String), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    # Create indexes for graph_entities
    op.create_index('ix_graph_entities_name', 'graph_entities', ['name'])
    op.create_index('ix_graph_entities_type', 'graph_entities', ['type'])
    op.create_index('ix_graph_entities_canonical_name', 'graph_entities', ['canonical_name'], unique=True)

    # Create vector index for entity embeddings
    op.execute("""
        CREATE INDEX ix_graph_entities_embedding
        ON graph_entities
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    # Create graph_relationships table
    op.create_table(
        'graph_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source_entity_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_entity_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('relation_type', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('weight', sa.Float, nullable=False, server_default='1.0'),
        sa.Column('article_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default='{}'),
        sa.Column('evidence_texts', postgresql.ARRAY(sa.Text), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    # Create indexes for graph_relationships
    op.create_index('ix_graph_relationships_source', 'graph_relationships', ['source_entity_id'])
    op.create_index('ix_graph_relationships_target', 'graph_relationships', ['target_entity_id'])
    op.create_index('ix_graph_relationships_type', 'graph_relationships', ['relation_type'])
    op.create_index(
        'ix_graph_relationships_unique',
        'graph_relationships',
        ['source_entity_id', 'target_entity_id', 'relation_type'],
        unique=True
    )

    # Create graph_communities table
    op.create_table(
        'graph_communities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('summary', sa.Text, nullable=True),
        sa.Column('entity_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default='{}'),
        sa.Column('hub_entity_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('article_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default='{}'),
        sa.Column('level', sa.Integer, nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    # Create indexes for graph_communities
    op.create_index('ix_graph_communities_name', 'graph_communities', ['name'])
    op.create_index('ix_graph_communities_level', 'graph_communities', ['level'])

    # Create graph_builder_runs table
    op.create_table(
        'graph_builder_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='running'),
        sa.Column('article_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default='{}'),
        sa.Column('entities_extracted', sa.Integer, nullable=False, server_default='0'),
        sa.Column('relationships_extracted', sa.Integer, nullable=False, server_default='0'),
        sa.Column('communities_detected', sa.Integer, nullable=False, server_default='0'),
        sa.Column('errors', postgresql.JSON, nullable=True),
        sa.Column('metadata', postgresql.JSON, nullable=True),
    )

    # Create indexes for graph_builder_runs
    op.create_index('ix_graph_builder_runs_started_at', 'graph_builder_runs', ['started_at'])
    op.create_index('ix_graph_builder_runs_status', 'graph_builder_runs', ['status'])


def downgrade() -> None:
    op.drop_index('ix_graph_builder_runs_status', table_name='graph_builder_runs')
    op.drop_index('ix_graph_builder_runs_started_at', table_name='graph_builder_runs')
    op.drop_table('graph_builder_runs')

    op.drop_index('ix_graph_communities_level', table_name='graph_communities')
    op.drop_index('ix_graph_communities_name', table_name='graph_communities')
    op.drop_table('graph_communities')

    op.drop_index('ix_graph_relationships_unique', table_name='graph_relationships')
    op.drop_index('ix_graph_relationships_type', table_name='graph_relationships')
    op.drop_index('ix_graph_relationships_target', table_name='graph_relationships')
    op.drop_index('ix_graph_relationships_source', table_name='graph_relationships')
    op.drop_table('graph_relationships')

    op.execute("DROP INDEX IF EXISTS ix_graph_entities_embedding")
    op.drop_index('ix_graph_entities_canonical_name', table_name='graph_entities')
    op.drop_index('ix_graph_entities_type', table_name='graph_entities')
    op.drop_index('ix_graph_entities_name', table_name='graph_entities')
    op.drop_table('graph_entities')