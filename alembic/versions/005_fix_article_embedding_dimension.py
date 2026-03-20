"""Fix article embedding dimension to 1024.

Revision ID: 005
Revises: 004
Create Date: 2026-03-18 12:58:00
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_article_embeddings_embedding", table_name="article_embeddings")
    op.execute(
        """
        ALTER TABLE article_embeddings
        ALTER COLUMN embedding TYPE vector(1024)
        """
    )
    op.create_index(
        "ix_article_embeddings_embedding",
        "article_embeddings",
        ["embedding"],
        postgresql_using="ivfflat",
    )


def downgrade() -> None:
    op.drop_index("ix_article_embeddings_embedding", table_name="article_embeddings")
    op.execute(
        """
        ALTER TABLE article_embeddings
        ALTER COLUMN embedding TYPE vector(1536)
        """
    )
    op.create_index(
        "ix_article_embeddings_embedding",
        "article_embeddings",
        ["embedding"],
        postgresql_using="ivfflat",
    )
