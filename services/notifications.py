"""Urgent-ticket alerting: transactional outbox + simulated delivery transports.

Flow
----
1. ``queue_urgent_alert`` inserts one ``pending`` row per channel **inside the
   ticket's transaction** (nothing is lost if the process dies right after).
2. After commit, ``dispatch_after_commit`` publishes a Celery task per row.
   A broker outage only delays delivery: the periodic sweeper re-queues stale
   ``pending`` rows.
3. The worker calls ``deliver_once``, which locks the row (``SKIP LOCKED``) and
   only acts on ``pending`` rows, so duplicate/overlapping task runs are harmless.

Transports are *simulated* (they log what a real email/Slack call would send).
To go live, implement ``NotificationTransport`` and call ``set_transport``.
"""

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any, Optional, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import Notification, Ticket
from services.tickets import utcnow

logger = logging.getLogger(__name__)

URGENT_CHANNELS = ("email", "slack")


# ------------------------------- transports ------------------------------------
class NotificationTransport(Protocol):
    """Delivers one alert. Raise any exception to signal a (retryable) failure."""

    async def send(self, channel: str, payload: dict[str, Any]) -> None: ...


class SimulatedTransport:
    """Logs the email / Slack-webhook request instead of sending it."""

    async def send(self, channel: str, payload: dict[str, Any]) -> None:
        settings = get_settings()
        if channel == "email":
            logger.info(
                "[SIMULATED EMAIL] to=%s subject=%r",
                settings.notification_email_to,
                f"[URGENT] {payload['ticket_id']}: {payload['title']}",
            )
        elif channel == "slack":
            body = {
                "channel": settings.notification_slack_channel,
                "text": (
                    f":rotating_light: Urgent ticket {payload['ticket_id']}: {payload['title']}"
                ),
            }
            logger.info("[SIMULATED SLACK WEBHOOK] POST body=%s", json.dumps(body))
        else:  # pragma: no cover - guarded by a CHECK constraint
            raise ValueError(f"unknown channel {channel}")
        await asyncio.sleep(0)  # a real transport awaits network I/O here


_transport: NotificationTransport = SimulatedTransport()


def get_transport() -> NotificationTransport:
    return _transport


def set_transport(transport: NotificationTransport) -> None:
    """Swap the transport (real integration, or a fake in tests)."""
    global _transport
    _transport = transport


# ------------------------------- producing -------------------------------------
async def queue_urgent_alert(session: AsyncSession, ticket: Ticket) -> list[int]:
    """Insert outbox rows for an urgent ticket; returns their ids (flushed, not committed)."""
    payload = {
        "ticket_id": ticket.id,
        "title": ticket.title,
        "priority": ticket.priority,
        "customer_id": ticket.customer_id,
        "created_at": ticket.created_at.isoformat(),
    }
    rows = [
        Notification(ticket_id=ticket.id, channel=ch, payload=payload) for ch in URGENT_CHANNELS
    ]
    session.add_all(rows)
    await session.flush()
    return [row.id for row in rows]


def enqueue_deliveries(notification_ids: list[int]) -> int:
    """Publish one Celery task per id. Never raises: the sweeper is the safety net."""
    if not notification_ids:
        return 0
    from worker.tasks import deliver_notification  # lazy: keeps Celery off the import path

    queued = 0
    for notification_id in notification_ids:
        try:
            deliver_notification.delay(notification_id)
            queued += 1
        except Exception:  # noqa: BLE001 - broker down; outbox rows stay pending
            # Stop at the first failure: if the broker is unreachable the rest would
            # each wait out their own timeout and stall the caller (an agent turn).
            logger.warning(
                "Could not enqueue notification %s (and %d more); sweeper will retry",
                notification_id,
                len(notification_ids) - queued - 1,
                exc_info=True,
            )
            break
    return queued


async def dispatch_after_commit(notification_ids: list[int]) -> int:
    """Async wrapper (publishing is blocking I/O, so run it in a worker thread)."""
    return await asyncio.to_thread(enqueue_deliveries, notification_ids)


# ------------------------------- consuming -------------------------------------
async def deliver_once(
    session: AsyncSession,
    notification_id: int,
    *,
    transport: NotificationTransport,
    max_attempts: int,
) -> str:
    """Try to deliver one pending notification.

    Returns ``"sent"``, ``"retry"`` (failed, attempts remain), ``"exhausted"``
    (failed permanently) or ``"skipped"`` (not pending / locked by another worker).
    """
    async with session.begin():
        row = (
            await session.execute(
                select(Notification)
                .where(Notification.id == notification_id, Notification.status == "pending")
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if row is None:
            return "skipped"
        row.attempts += 1
        try:
            await transport.send(row.channel, row.payload)
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
            row.last_error = f"{type(exc).__name__}: {exc}"[:500]
            if row.attempts >= max_attempts:
                row.status = "failed"
                logger.error("Notification %s failed permanently: %s", row.id, row.last_error)
                return "exhausted"
            return "retry"
        row.status = "sent"
        row.sent_at = utcnow()
        row.last_error = None
        return "sent"


async def stale_pending_ids(
    session: AsyncSession, *, older_than: timedelta, max_attempts: int, limit: int = 500
) -> list[int]:
    """Ids of pending notifications nobody has delivered for a while."""
    cutoff: Optional[Any] = utcnow() - older_than
    result = await session.execute(
        select(Notification.id)
        .where(
            Notification.status == "pending",
            Notification.created_at < cutoff,
            Notification.attempts < max_attempts,
        )
        .order_by(Notification.created_at)
        .limit(limit)
    )
    return list(result.scalars())
