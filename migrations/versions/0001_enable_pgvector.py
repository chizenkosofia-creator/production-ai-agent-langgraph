"""Enable the pgvector extension.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Dropping the extension would cascade-delete every vector column in the
    # database, including data other applications may own. Deliberately a no-op.
    pass
