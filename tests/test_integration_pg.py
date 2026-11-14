"""Integration tests against a REAL Postgres with pgvector.

Uses ``TEST_DATABASE_URL`` if set (e.g. the compose db), otherwise starts an
embedded Postgres via the optional ``pixeltable-pgserver`` package, otherwise
skips. The LLM and embeddings are still fakes, so no network is needed.
"""

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import agent.graph as graph_module
import agent.rag as rag_module
import agent.tools as tools_module
import api.main as api_main
from agent.graph import graph_lifespan
from agent.locks import thread_lock
from scripts.ingest_kb import load_chunks
from tests.fakes import HashEmbeddings, ScriptedChatModel, tool_call_message
from tests.test_api_stream import parse_sse

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch):
    """No model downloads: hash embeddings for RAG, scripted LLM per test."""
    monkeypatch.setattr(rag_module, "build_embeddings", lambda settings: HashEmbeddings())


def _scalar(url: str, sql: str, params=None):
    with psycopg.connect(url, autocommit=True) as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None


# ----------------------------- migrations ---------------------------------
def test_migrations_create_extension_fixed_dim_column_and_hnsw_index(migrated_url):
    assert (
        _scalar(migrated_url, "SELECT extname FROM pg_extension WHERE extname='vector'") == "vector"
    )
    column_type = _scalar(
        migrated_url,
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid='langchain_pg_embedding'::regclass AND attname='embedding'",
    )
    assert column_type == "vector(384)"
    indexdef = _scalar(
        migrated_url, "SELECT indexdef FROM pg_indexes WHERE indexname='ix_kb_embedding_hnsw'"
    )
    assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef


def test_migrations_are_idempotent(migrated_url):
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")  # second run must not fail
    assert _scalar(migrated_url, "SELECT version_num FROM alembic_version") == "0003"


# ----------------------------- vector store -------------------------------
async def test_ingest_search_ranking_idempotency_and_index_use(pg_settings, migrated_url):
    store = rag_module.init_vector_store(pg_settings)
    try:
        chunks = load_chunks(ROOT / "kb_docs")
        assert chunks, "sample knowledge base must not be empty"
        await rag_module.ingest_documents(chunks)
        await rag_module.ingest_documents(chunks)  # upsert, not duplicate
        total = _scalar(migrated_url, "SELECT count(*) FROM langchain_pg_embedding")
        assert total == len(chunks)

        hits = await store.asimilarity_search_with_score(
            "refund for a starter plan customer within 14 days", k=3
        )
        assert hits[0][0].metadata["source"] == "refund_policy.md"
        assert hits[0][1] <= hits[-1][1], "results must be ordered by ascending distance"

        hits = await store.asimilarity_search_with_score(
            "urgent incident page on-call engineer", k=3
        )
        assert hits[0][0].metadata["source"] == "incident_runbook.md"
    finally:
        await rag_module.close_vector_store()

    # The planner can use the HNSW index for cosine-distance ordering.
    vec = "[" + ",".join(["0.05"] * 384) + "]"
    with psycopg.connect(migrated_url, autocommit=True) as conn:
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT id FROM langchain_pg_embedding "
                f"ORDER BY embedding <=> '{vec}'::vector LIMIT 3"
            ).fetchall()
        )
    assert "ix_kb_embedding_hnsw" in plan


# ----------------------------- checkpointer + lock ------------------------
async def test_thread_survives_restart_and_lock_serialises_turns(pg_settings, monkeypatch):
    llm = ScriptedChatModel(script=[AIMessage("Hello Dana."), AIMessage("You are Dana.")])
    monkeypatch.setattr(graph_module, "build_llm", lambda settings: llm)
    config = {"configurable": {"thread_id": "persist-1"}, "recursion_limit": 30}

    def turn(text):
        from langchain_core.messages import HumanMessage

        return {"messages": [HumanMessage(text)], "iterations": 0, "retrieved_context": []}

    async with graph_lifespan(pg_settings) as runtime:
        await runtime.graph.ainvoke(turn("I am Dana."), config)

        # --- advisory lock: second taker is refused while the first holds it
        async with thread_lock(runtime.lock_pool, "persist-1") as first:
            async with thread_lock(runtime.lock_pool, "persist-1") as second:
                assert first is True and second is False
            async with thread_lock(runtime.lock_pool, "other-thread") as other:
                assert other is True, "locks are per thread"
        async with thread_lock(runtime.lock_pool, "persist-1") as again:
            assert again is True, "lock must be released after use"

    # "Restart": brand-new pools, checkpointer and graph; state must come from Postgres.
    async with graph_lifespan(pg_settings) as runtime:
        snapshot = await runtime.graph.aget_state(config)
        assert [m.content for m in snapshot.values["messages"]] == ["I am Dana.", "Hello Dana."]
        await runtime.graph.ainvoke(turn("Who am I?"), config)
        assert any("I am Dana." in str(m.content) for m in llm.calls[1])


# ----------------------------- end-to-end SSE ------------------------------
def test_sse_end_to_end_over_postgres_with_rag(pg_settings, migrated_url, monkeypatch):
    """Real Postgres checkpointer + pgvector search + advisory lock + SSE."""
    import asyncio

    async def _seed() -> None:
        rag_module.init_vector_store(pg_settings)
        try:
            await rag_module.ingest_documents(load_chunks(ROOT / "kb_docs"))
        finally:
            await rag_module.close_vector_store()

    asyncio.run(_seed())

    llm = ScriptedChatModel(
        script=[
            tool_call_message("search_knowledge_base", {"query": "urgent incident escalation"}),
            AIMessage("Open an urgent ticket and page the on-call engineer."),
        ]
    )
    monkeypatch.setattr(graph_module, "build_llm", lambda settings: llm)
    monkeypatch.setattr(tools_module, "get_settings", lambda: pg_settings)
    monkeypatch.setattr(api_main, "graph_lifespan", lambda _settings: graph_lifespan(pg_settings))

    with TestClient(api_main.app) as client:
        r = client.post(
            "/api/v1/chat/stream",
            json={"thread_id": "e2e-1", "message": "How do I escalate an outage?"},
        )
        events = parse_sse(r.text)
        names = [e for e, _ in events]
        assert names[0] == "tool_start" and "tool_end" in names and names[-1] == "done"
        # threshold is disabled in this fixture (hash embeddings), so lower-ranked
        # chunks may follow; what matters is that the best match comes first.
        assert events[-1][1]["sources"][0] == "incident_runbook.md"
        assert "".join(d["text"] for e, d in events if e == "token").startswith("Open an urgent")

        history = client.get("/api/v1/threads/e2e-1/history").json()
        assert [m["role"] for m in history["messages"]] == ["human", "ai", "tool", "ai"]
