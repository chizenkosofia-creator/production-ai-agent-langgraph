"""Knowledge-base tables with a fixed-dimension vector column + HNSW index.

``langchain-postgres`` would create these tables itself, but with an
*unbounded* ``vector`` column, and pgvector cannot build an HNSW index on a
column without a declared dimension. We therefore create them here with the
exact schema the library expects (langchain_pg_collection /
langchain_pg_embedding) and ``vector(384)`` (all-MiniLM-L6-v2). The library's
own ``CREATE TABLE IF NOT EXISTS`` then does nothing.

If you change EMBEDDING_MODEL to one with a different dimension, add a new
migration that alters the column (and re-embed your documents).

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS langchain_pg_collection (
            uuid      UUID PRIMARY KEY,
            name      VARCHAR NOT NULL UNIQUE,
            cmetadata JSON
        )
        """)
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
            id            VARCHAR PRIMARY KEY,
            collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
            embedding     vector({EMBEDDING_DIM}),
            document      VARCHAR,
            cmetadata     JSONB
        )
        """)
    # Metadata filtering (matches the index the library declares).
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
        ON langchain_pg_embedding USING gin (cmetadata jsonb_path_ops)
        """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection
        ON langchain_pg_embedding (collection_id)
        """)
    # Approximate nearest-neighbour search by cosine distance (the library's
    # default strategy, i.e. the `<=>` operator).
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kb_embedding_hnsw
        ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS langchain_pg_embedding")
    op.execute("DROP TABLE IF EXISTS langchain_pg_collection")
