"""RAG plumbing: local embeddings + a PGVector store (``langchain-postgres``).

Architecture decisions
----------------------
* ``langchain_community.vectorstores.pgvector`` is deprecated; its maintained
  successor is ``langchain_postgres.PGVector`` (same interface, psycopg3).
* The store gets its **own** async SQLAlchemy engine (separate from the
  psycopg pool used by the LangGraph checkpointer) so the two never contend
  for connections or share transaction state.
* The ``vector`` extension and the tables/indexes are owned by Alembic
  (migrations 0001/0002), hence ``create_extension=False``. The library's
  ``CREATE TABLE IF NOT EXISTS`` is then a no-op.
* Embeddings are computed locally (sentence-transformers) because Groq has no
  embeddings endpoint. Vectors are L2-normalised and compared by cosine
  distance, which is what the HNSW index in migration 0002 is built for.
* The store is a module-level singleton initialised once in the app lifespan;
  tools reach it via ``get_vector_store()`` (tests inject a stub with
  ``set_vector_store``).
"""

import hashlib
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from core.config import Settings

logger = logging.getLogger(__name__)

_store: Optional[PGVector] = None


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the local embedding model (downloads weights on first use)."""
    # Imported lazily: pulls in torch, which tests and tooling do not need.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def init_vector_store(settings: Settings, embeddings: Optional[Embeddings] = None) -> PGVector:
    """Create the process-wide vector store (call once at startup)."""
    global _store
    _store = PGVector(
        embeddings=embeddings or build_embeddings(settings),
        embedding_length=settings.embedding_dim,
        collection_name=settings.rag_collection,
        connection=settings.sqlalchemy_url,
        use_jsonb=True,
        async_mode=True,
        create_extension=False,  # owned by the Alembic migration
    )
    logger.info(
        "Vector store ready (collection=%s, model=%s)",
        settings.rag_collection,
        settings.embedding_model,
    )
    return _store


def set_vector_store(store: object) -> None:
    """Override the singleton (used by tests)."""
    global _store
    _store = store  # type: ignore[assignment]


def get_vector_store() -> PGVector:
    """Return the initialised store or fail loudly."""
    if _store is None:
        raise RuntimeError("Vector store not initialised; call init_vector_store()")
    return _store


async def close_vector_store() -> None:
    """Dispose the store's async engine at shutdown."""
    global _store
    engine = getattr(_store, "_async_engine", None)
    if engine is not None:
        await engine.dispose()
    _store = None


def stable_chunk_id(source: str, index: int, text: str) -> str:
    """Deterministic id so re-ingesting a document upserts instead of duplicating."""
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()
    return digest


async def ingest_documents(docs: list[Document]) -> int:
    """Upsert chunks (ids derived from source + position + content)."""
    ids = [
        stable_chunk_id(str(d.metadata.get("source", "")), i, d.page_content)
        for i, d in enumerate(docs)
    ]
    await get_vector_store().aadd_documents(docs, ids=ids)
    return len(docs)
