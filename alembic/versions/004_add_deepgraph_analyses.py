"""Add deepgraph_analyses table for user tracking.

Revision ID: 004
Revises: 003
Create Date: 2026-03-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create deepgraph_analyses table for user tracking
    op.create_table(
        'deepgraph_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.BigInteger, nullable=False),
        sa.Column('article_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('report', sa.Text, nullable=True),
        sa.Column('visualization_data', postgresql.JSON, nullable=True),
        sa.Column('max_hops', sa.Integer, nullable=False, server_default='2'),
        sa.Column('expansion_limit', sa.Integer, nullable=False, server_default='50'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    # Create indexes for user queries
    op.create_index('ix_deepgraph_analyses_user_id', 'deepgraph_analyses', ['user_id'])
    op.create_index('ix_deepgraph_analyses_created_at', 'deepgraph_analyses', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_deepgraph_analyses_created_at', table_name='deepgraph_analyses')
    op.drop_index('ix_deepgraph_analyses_user_id', table_name='deepgraph_analyses')
    op.drop_table('deepgraph_analyses')