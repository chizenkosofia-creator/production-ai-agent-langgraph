"""Outbox + Celery tests: delivery, retry/back-off, broker outage, sweeper, path guard.

Celery runs in eager mode (tasks execute inline) against the real test database,
so no Redis is needed; the broker failure case is simulated by making ``delay``
raise.
"""

import asyncio
import json
from datetime import timedelta

import pytest
from sqlalchemy import select, text

from agent.ticket_tools import create_support_ticket
from db.models import Notification
from services import notifications
from worker import tasks

pytestmark = pytest.mark.integration

URGENT = {
    "title": "Production outage",
    "description": "All users are getting 500 errors since 09:00.",
    "priority": "urgent",
}


class RecordingTransport:
    """Fails the first ``fail_times`` sends of each notification, then succeeds."""

    def __init__(self, fail_times: int = 0, always_fail: bool = False) -> None:
        self.fail_times, self.always_fail = fail_times, always_fail
        self.calls: list[tuple[str, str]] = []
        self.sends: dict[str, int] = {}

    async def send(self, channel: str, payload: dict) -> None:
        self.calls.append((channel, payload["ticket_id"]))
        self.sends[channel] = self.sends.get(channel, 0) + 1
        if self.always_fail or self.sends[channel] <= self.fail_times:
            raise ConnectionError("smtp unreachable")


async def _run(task, *args):
    """Run a (synchronous) Celery task off the event loop, as a worker would."""
    return await asyncio.to_thread(lambda: task.apply(args=list(args)).get())


async def _rows(factory) -> list[Notification]:
    async with factory() as session:
        return list(
            (await session.execute(select(Notification).order_by(Notification.id))).scalars()
        )


# ------------------------------ happy path -----------------------------------
async def test_urgent_ticket_writes_outbox_atomically_and_delivers_both_channels(db, eager_worker):
    transport = RecordingTransport()
    notifications.set_transport(transport)

    out = json.loads(await create_support_ticket.ainvoke(URGENT))

    assert out["alerts_queued"] == 2
    rows = await _rows(db)
    assert {r.channel for r in rows} == {"email", "slack"}
    assert all(r.status == "sent" and r.attempts == 1 and r.sent_at for r in rows)
    assert sorted(c for c, _ in transport.calls) == ["email", "slack"]
    assert all(r.payload["ticket_id"] == out["ticket_id"] for r in rows)


async def test_non_urgent_priorities_never_alert(db, eager_worker):
    notifications.set_transport(RecordingTransport())
    for priority in ("low", "medium", "high"):
        await create_support_ticket.ainvoke({**URGENT, "priority": priority})
    assert await _rows(db) == []


# ------------------------------ retry / failure ------------------------------
async def test_transient_failures_are_retried_until_delivered(db, eager_worker):
    transport = RecordingTransport(fail_times=2)  # each channel fails twice, then works
    notifications.set_transport(transport)

    await create_support_ticket.ainvoke(URGENT)

    rows = await _rows(db)
    assert all(r.status == "sent" and r.attempts == 3 and r.last_error is None for r in rows)
    assert len(transport.calls) == 6


async def test_permanent_failure_stops_at_max_attempts_and_keeps_the_error(db, eager_worker):
    notifications.set_transport(RecordingTransport(always_fail=True))

    await create_support_ticket.ainvoke(URGENT)

    rows = await _rows(db)
    assert all(r.status == "failed" for r in rows)
    assert all(r.attempts == eager_worker.notification_max_attempts for r in rows)
    assert all("ConnectionError" in r.last_error for r in rows)


async def test_delivery_is_idempotent(db, eager_worker):
    transport = RecordingTransport()
    notifications.set_transport(transport)
    await create_support_ticket.ainvoke(URGENT)
    first = await _rows(db)

    for row in first:  # a duplicate task (e.g. sweeper + original) must do nothing
        assert await _run(tasks.deliver_notification, row.id) == "skipped"
    assert len(transport.calls) == 2


def test_backoff_is_exponential_and_capped():
    assert [tasks.backoff_seconds(n, 30) for n in (1, 2, 3, 4)] == [30, 60, 120, 240]
    assert tasks.backoff_seconds(30, 30) == tasks.MAX_BACKOFF_SECONDS


# ------------------------------ broker outage + sweeper ----------------------
async def test_broker_outage_does_not_fail_the_ticket_and_sweeper_recovers(
    db, eager_worker, monkeypatch
):
    transport = RecordingTransport()
    notifications.set_transport(transport)

    with monkeypatch.context() as m:

        def boom(*args, **kwargs):
            raise ConnectionError("redis is down")

        m.setattr(tasks.deliver_notification, "delay", boom)
        out = json.loads(await create_support_ticket.ainvoke(URGENT))

    assert out["alerts_queued"] == 0, "nothing could be published"
    assert out["ticket_id"].startswith("TCK-"), "the ticket itself must still be created"
    rows = await _rows(db)
    assert [r.status for r in rows] == ["pending", "pending"], "outbox rows survived the outage"
    assert transport.calls == []

    # fresh rows are left alone by the sweeper ...
    assert await _run(tasks.sweep_pending) == 0
    # ... but once they are stale it re-publishes them and they get delivered
    async with db() as session:
        await session.execute(
            text("UPDATE notifications SET created_at = now() - interval '10 minutes'")
        )
        await session.commit()
    assert await _run(tasks.sweep_pending) == 2
    assert all(r.status == "sent" for r in await _rows(db))


async def test_sweeper_ignores_rows_that_are_not_pending_or_exhausted(db, eager_worker):
    notifications.set_transport(RecordingTransport(always_fail=True))
    await create_support_ticket.ainvoke(URGENT)  # ends up failed
    async with db() as session:
        await session.execute(
            text("UPDATE notifications SET created_at = now() - interval '1 hour'")
        )
        await session.commit()
        stale = await notifications.stale_pending_ids(
            session, older_than=timedelta(minutes=2), max_attempts=5
        )
    assert stale == []


# ------------------------------ batch tasks ----------------------------------
async def test_weekly_report_task_returns_window_stats(db, eager_worker):
    await create_support_ticket.ainvoke(URGENT)
    notifications.set_transport(RecordingTransport())
    stats = await _run(tasks.weekly_ticket_report)
    assert stats["created_in_window"]["open"] == 1
    assert stats["backlog_now"]["urgent"] == 1


def test_kb_ingest_path_guard(tmp_path):
    root = tmp_path / "kb"
    (root / "sub").mkdir(parents=True)
    assert tasks.resolve_kb_dir("sub", str(root)) == (root / "sub").resolve()
    assert tasks.resolve_kb_dir(".", str(root)) == root.resolve()
    for evil in ("..", "../..", "/etc", "sub/../../.."):
        with pytest.raises(ValueError, match="outside"):
            tasks.resolve_kb_dir(evil, str(root))
    with pytest.raises(ValueError, match="not a directory"):
        tasks.resolve_kb_dir("missing", str(root))
