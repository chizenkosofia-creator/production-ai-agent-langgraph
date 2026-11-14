"""Test fixtures.

Environment variables are set BEFORE any application module is imported
because ``api.main`` builds the FastAPI app (and reads settings) at import time.
No test needs network access: the LLM is a scripted fake and embeddings are a
deterministic hash-based stand-in.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("AGENT_MAX_ITERATIONS", "3")
os.environ.setdefault("AGENT_RECURSION_LIMIT", "30")
os.environ.setdefault("SSE_HEARTBEAT_SECONDS", "5")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from agent import rag  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_vector_store():
    """Every test starts and ends without a global vector store."""
    rag.set_vector_store(None)
    yield
    rag.set_vector_store(None)


# --------------------------------------------------------------------------
# Real-Postgres fixtures (integration tests)
# --------------------------------------------------------------------------
import shutil  # noqa: E402
import tempfile  # noqa: E402

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from core.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def pg_url():
    """A Postgres+pgvector URL: ``TEST_DATABASE_URL``, else an embedded server, else skip."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    try:
        import pixeltable_pgserver as pgserver
    except ImportError:
        pytest.skip("set TEST_DATABASE_URL or pip install pixeltable-pgserver")
    workdir = tempfile.mkdtemp(prefix="pgtest-")
    server = pgserver.get_server(workdir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def migrated_url(pg_url):
    """Apply Alembic migrations once for the whole session."""
    os.environ["DATABASE_URL"] = pg_url
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    return pg_url


@pytest.fixture
def pg_settings(migrated_url):
    return Settings(database_url=migrated_url, rag_max_distance=2.0)


@pytest.fixture
async def db(pg_settings):
    """Session factory bound to the migrated test DB, with clean ticket tables.

    ``operators`` keeps its seed rows (created by migration 0003).
    """
    from sqlalchemy import text

    from db.session import close_db, init_db

    factory = init_db(pg_settings)
    async with factory() as session:
        await session.execute(text("TRUNCATE notifications, tickets RESTART IDENTITY CASCADE"))
        await session.execute(text("UPDATE operators SET active = true"))
        await session.commit()
    yield factory
    await close_db()


@pytest.fixture
def eager_worker(monkeypatch, pg_settings):
    """Run Celery tasks inline against the test DB (no Redis needed)."""
    import worker.tasks as worker_tasks
    from services import notifications
    from worker.celery_app import celery_app

    monkeypatch.setattr(worker_tasks, "get_settings", lambda: pg_settings)
    monkeypatch.setitem(celery_app.conf, "task_always_eager", True)
    original_transport = notifications.get_transport()
    yield pg_settings
    notifications.set_transport(original_transport)
