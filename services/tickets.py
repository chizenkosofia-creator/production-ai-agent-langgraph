"""Ticket, operator and analytics operations.

Every function takes an ``AsyncSession`` and leaves transaction control to the
caller (``async with session.begin():``), so several operations (e.g. creating a
ticket and queueing its alert) commit atomically.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.errors import DomainError
from db.models import TICKET_STATUSES, Operator, Ticket

WEEK = timedelta(days=7)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_ticket_id() -> str:
    return f"TCK-{uuid.uuid4().hex[:8].upper()}"


def ticket_to_dict(ticket: Ticket, assignee_email: Optional[str] = None) -> dict[str, Any]:
    """JSON-safe view of a ticket (what tools return to the LLM)."""
    return {
        "ticket_id": ticket.id,
        "title": ticket.title,
        "description": ticket.description,
        "priority": ticket.priority,
        "status": ticket.status,
        "customer_id": ticket.customer_id,
        "assignee": assignee_email,
        "created_at": ticket.created_at.isoformat(),
        "updated_at": ticket.updated_at.isoformat(),
        "closed_at": ticket.closed_at.isoformat() if ticket.closed_at else None,
    }


async def create_ticket(
    session: AsyncSession,
    *,
    title: str,
    description: str,
    priority: str,
    customer_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Ticket:
    """Insert a new open ticket."""
    now = now or utcnow()
    ticket = Ticket(
        id=new_ticket_id(),
        title=title,
        description=description,
        priority=priority,
        status="open",
        customer_id=customer_id,
        created_at=now,
        updated_at=now,
    )
    session.add(ticket)
    await session.flush()
    return ticket


async def weekly_stats(
    session: AsyncSession, *, now: Optional[datetime] = None, week_offset: int = 0
) -> dict[str, Any]:
    """Ticket statistics for one 7-day window ``(start, end]``.

    ``week_offset=0`` is the 7 days ending now, ``1`` the 7 days before that, etc.

    * ``created_in_window``: tickets *created* in the window, by their CURRENT status.
    * ``resolved_in_window``: tickets that were *closed* in the window (any age).
    * ``backlog_now``: all tickets not closed right now, by priority (not windowed).
    """
    end = (now or utcnow()) - WEEK * week_offset
    start = end - WEEK

    by_status = dict(
        (
            await session.execute(
                select(Ticket.status, func.count())
                .where(Ticket.created_at > start, Ticket.created_at <= end)
                .group_by(Ticket.status)
            )
        ).all()
    )
    created = {status: int(by_status.get(status, 0)) for status in TICKET_STATUSES}
    created["total"] = sum(created.values())

    resolved = (
        await session.execute(
            select(func.count()).where(Ticket.closed_at > start, Ticket.closed_at <= end)
        )
    ).scalar_one()

    backlog_rows = (
        await session.execute(
            select(Ticket.priority, func.count())
            .where(Ticket.status != "closed")
            .group_by(Ticket.priority)
        )
    ).all()
    backlog = {priority: int(count) for priority, count in backlog_rows}
    backlog["total"] = sum(backlog.values())

    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "created_in_window": created,
        "resolved_in_window": int(resolved),
        "backlog_now": backlog,
    }


async def list_active_operators(session: AsyncSession) -> list[Operator]:
    result = await session.execute(
        select(Operator).where(Operator.active.is_(True)).order_by(Operator.name)
    )
    return list(result.scalars())


async def update_ticket(
    session: AsyncSession,
    ticket_id: str,
    *,
    status: Optional[str] = None,
    assignee_email: Optional[str] = None,
    now: Optional[datetime] = None,
) -> tuple[Ticket, Optional[str]]:
    """Change a ticket's status and/or assignee. Returns ``(ticket, assignee_email)``.

    The ticket row is locked (``FOR UPDATE``) so two concurrent updates serialise
    instead of silently overwriting each other. Closing stamps ``closed_at``;
    re-opening clears it.
    """
    if status is not None and status not in TICKET_STATUSES:
        raise DomainError(f"invalid status '{status}'; use one of {', '.join(TICKET_STATUSES)}")

    ticket = (
        await session.execute(select(Ticket).where(Ticket.id == ticket_id).with_for_update())
    ).scalar_one_or_none()
    if ticket is None:
        raise DomainError(f"ticket {ticket_id} not found")

    now = now or utcnow()
    operator: Optional[Operator] = None
    if assignee_email is not None:
        operator = (
            await session.execute(
                select(Operator).where(func.lower(Operator.email) == assignee_email.lower())
            )
        ).scalar_one_or_none()
        if operator is None:
            raise DomainError(f"no operator with email {assignee_email}; call list_operators")
        if not operator.active:
            raise DomainError(f"operator {operator.email} is inactive and cannot be assigned")
        ticket.assignee_id = operator.id

    if status is not None and status != ticket.status:
        ticket.status = status
        ticket.closed_at = now if status == "closed" else None

    ticket.updated_at = now
    await session.flush()

    if operator is None and ticket.assignee_id is not None:
        operator = await session.get(Operator, ticket.assignee_id)
    return ticket, (operator.email if operator else None)
