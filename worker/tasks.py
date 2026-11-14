"""Celery tasks: alert delivery, outbox sweeper, KB ingestion, weekly report.

Celery tasks are synchronous, the data layer is async: every task runs its
coroutine with ``asyncio.run`` on a **short-lived engine with ``NullPool``**.
Pooled async connections are bound to the event loop that created them, so a
process-wide pool (as in the API) would break on the second task.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

from celery import Task
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from core.config import Settings, get_settings
from services import notifications, tickets
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

T = TypeVar("T")
SessionFactory = async_sessionmaker[AsyncSession]

# Pending rows older than this are re-published by the sweeper. Rows in a retry
# back-off may be swept a little early; delivery is idempotent so that is harmless.
SWEEP_STALE_AFTER = timedelta(minutes=2)
MAX_BACKOFF_SECONDS = 3600


def run_async(fn: Callable[[SessionFactory], Awaitable[T]]) -> T:
    """Run ``fn(session_factory)`` to completion on a throw-away engine."""

    async def runner() -> T:
        engine = create_async_engine(get_settings().sqlalchemy_url, poolclass=NullPool)
        try:
            return await fn(async_sessionmaker(engine, expire_on_commit=False))
        finally:
            await engine.dispose()

    return asyncio.run(runner())


def backoff_seconds(attempts: int, base: int) -> int:
    """Exponential back-off: base, 2*base, 4*base ... capped at one hour."""
    return min(base * 2 ** max(attempts - 1, 0), MAX_BACKOFF_SECONDS)


# ------------------------------- alert delivery --------------------------------
async def _deliver(
    factory: SessionFactory, notification_id: int, settings: Settings
) -> tuple[str, int]:
    async with factory() as session:
        outcome = await notifications.deliver_once(
            session,
            notification_id,
            transport=notifications.get_transport(),
            max_attempts=settings.notification_max_attempts,
        )
    attempts = 0
    if outcome == "retry":
        async with factory() as session:
            row = await session.get(notifications.Notification, notification_id)
            attempts = row.attempts if row else 0
    return outcome, attempts


# max_retries=None = unlimited *Celery* retries; the real cap is notification_max_attempts,
# enforced in deliver_once. (Passing max_retries=None to self.retry() would instead mean
# "use the task default of 3", which is why it is set here.)
@celery_app.task(
    bind=True, name="notifications.deliver", acks_late=True, max_retries=None, ignore_result=True
)
def deliver_notification(self: Task, notification_id: int) -> str:
    """Deliver one outbox row; re-schedule itself with back-off on failure."""
    settings = get_settings()
    outcome, attempts = run_async(lambda f: _deliver(f, notification_id, settings))
    logger.info("notification %s -> %s", notification_id, outcome)
    if outcome == "retry":
        raise self.retry(
            countdown=backoff_seconds(attempts, settings.notification_retry_base_seconds)
        )
    return outcome


# ------------------------------- outbox sweeper --------------------------------
@celery_app.task(name="notifications.sweep_pending", ignore_result=True)
def sweep_pending() -> int:
    """Re-publish pending alerts nobody delivered (broker was down, worker died)."""
    settings = get_settings()

    async def find(factory: SessionFactory) -> list[int]:
        async with factory() as session:
            return await notifications.stale_pending_ids(
                session,
                older_than=SWEEP_STALE_AFTER,
                max_attempts=settings.notification_max_attempts,
            )

    ids = run_async(find)
    queued = notifications.enqueue_deliveries(ids)
    if ids:
        logger.warning("sweeper re-queued %d/%d pending notifications", queued, len(ids))
    return queued


# ------------------------------- KB ingestion (heavy) ---------------------------
def resolve_kb_dir(requested: str, allowed_root: str) -> Path:
    """Resolve ``requested`` and refuse anything outside ``allowed_root``.

    The task argument is caller-controlled, so without this check it could
    point the ingester at any readable directory (``/etc``, a mounted secret).
    """
    root = Path(allowed_root).resolve()
    target = (
        (root / requested).resolve()
        if not Path(requested).is_absolute()
        else Path(requested).resolve()
    )
    if target != root and root not in target.parents:
        raise ValueError(f"path {requested!r} is outside the allowed directory {allowed_root!r}")
    if not target.is_dir():
        raise ValueError(f"{requested!r} is not a directory")
    return target


@celery_app.task(name="kb.ingest", time_limit=3600, soft_time_limit=3500)
def ingest_kb(directory: str = ".", replace: bool = False) -> int:
    """Chunk, embed and upsert documents under the configured KB directory."""
    from scripts.ingest_kb import main as ingest_main  # lazy: pulls in torch/embeddings

    target = resolve_kb_dir(directory, get_settings().kb_docs_dir)
    count = asyncio.run(ingest_main(target, replace))
    logger.info("ingested %d chunks from %s", count, target)
    return count


# ------------------------------- weekly report (batch) --------------------------
@celery_app.task(name="reports.weekly_ticket_report")
def weekly_ticket_report(week_offset: int = 0) -> dict[str, Any]:
    """Aggregate last week's ticket statistics and log them as one JSON line."""

    async def compute(factory: SessionFactory) -> dict[str, Any]:
        async with factory() as session:
            return await tickets.weekly_stats(session, week_offset=week_offset)

    stats = run_async(compute)
    logger.info("weekly_ticket_report %s", json.dumps(stats))
    return stats
