"""Add chunk embeddings support for RAG system.

Revision ID: 002
Revises: 001
Create Date: 2024-01-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Remove unique constraint on article_id (allows multiple embeddings per article)
    op.drop_constraint('article_embeddings_article_id_key', 'article_embeddings', type_='unique')
    op.drop_index('ix_article_embeddings_article_id', table_name='article_embeddings')

    # Step 2: Add new columns for chunk support
    op.add_column('article_embeddings', sa.Column('chunk_number', sa.Integer, nullable=True, default=0))
    op.add_column('article_embeddings', sa.Column('chunk_text', sa.Text, nullable=True))
    op.add_column('article_embeddings', sa.Column('chunk_start', sa.Integer, nullable=True))
    op.add_column('article_embeddings', sa.Column('chunk_end', sa.Integer, nullable=True))
    op.add_column('article_embeddings', sa.Column('embedding_type', sa.String(20), nullable=True, default='summary'))

    # Step 3: Update existing records to have embedding_type='summary' and chunk_number=0
    op.execute("""
        UPDATE article_embeddings
        SET embedding_type = 'summary', chunk_number = 0
        WHERE embedding_type IS NULL
    """)

    # Step 4: Make new columns non-nullable after setting defaults
    op.alter_column('article_embeddings', 'chunk_number', nullable=False)
    op.alter_column('article_embeddings', 'embedding_type', nullable=False)

    # Step 5: Create composite unique index for (article_id, chunk_number)
    op.create_index(
        'ix_article_embeddings_article_chunk',
        'article_embeddings',
        ['article_id', 'chunk_number'],
        unique=True
    )

    # Step 6: Add tsvector column for full-text search
    op.add_column('article_embeddings', sa.Column('chunk_tsv', postgresql.TSVECTOR, nullable=True))

    # Step 7: Create GIN index for tsvector
    op.create_index(
        'ix_article_embeddings_chunk_tsv',
        'article_embeddings',
        ['chunk_tsv'],
        postgresql_using='gin'
    )

    # Step 8: Create trigger function to automatically update tsvector
    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunk_tsv()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.chunk_tsv :=
                setweight(to_tsvector('english', COALESCE(NEW.chunk_text, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW.chunk_text, '')), 'B');
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
    """)

    # Step 9: Create trigger on article_embeddings table (must be separate for asyncpg)
    op.execute("DROP TRIGGER IF EXISTS chunk_tsv_update ON article_embeddings")
    op.execute("""
        CREATE TRIGGER chunk_tsv_update
            BEFORE INSERT OR UPDATE ON article_embeddings
            FOR EACH ROW
            WHEN (pg_trigger_depth() = 0)
            EXECUTE FUNCTION update_chunk_tsv()
    """)

    # Step 10: Optimize vector index for better performance
    # Drop old ivfflat index and create new one with better parameters
    op.execute("DROP INDEX IF EXISTS ix_article_embeddings_vector")
    op.execute("""
        CREATE INDEX ix_article_embeddings_vector
        ON article_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS chunk_tsv_update ON article_embeddings")
    op.execute("DROP FUNCTION IF EXISTS update_chunk_tsv()")

    # Drop indexes
    op.drop_index('ix_article_embeddings_chunk_tsv', table_name='article_embeddings')
    op.drop_index('ix_article_embeddings_article_chunk', table_name='article_embeddings')

    # Drop columns
    op.drop_column('article_embeddings', 'chunk_tsv')
    op.drop_column('article_embeddings', 'embedding_type')
    op.drop_column('article_embeddings', 'chunk_end')
    op.drop_column('article_embeddings', 'chunk_start')
    op.drop_column('article_embeddings', 'chunk_text')
    op.drop_column('article_embeddings', 'chunk_number')

    # Restore unique constraint on article_id
    op.create_unique_constraint('article_embeddings_article_id_key', 'article_embeddings', ['article_id'])
    op.create_index('ix_article_embeddings_article_id', 'article_embeddings', ['article_id'], unique=True)