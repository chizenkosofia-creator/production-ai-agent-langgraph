"""SQLAlchemy 2.0 models for tickets, support operators and the notification outbox.

Design notes
------------
* Enumerated columns are plain ``VARCHAR`` + ``CHECK`` constraints (not native
  PG enums): adding a status later is a one-line constraint change rather than
  an ``ALTER TYPE`` migration.
* ``notifications`` is a transactional **outbox**: the alert row is written in
  the same transaction as the urgent ticket, so a crash or a Redis outage can
  never lose an alert. Celery delivers rows and a periodic sweeper re-queues any
  that were missed.
* Timestamps are timezone-aware (UTC).
"""

from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

TICKET_STATUSES = ("open", "in_progress", "closed")
TICKET_PRIORITIES = ("low", "medium", "high", "urgent")
NOTIFICATION_CHANNELS = ("email", "slack")
NOTIFICATION_STATUSES = ("pending", "sent", "failed")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    """Declarative base for all application tables."""


class Operator(Base):
    """A support operator tickets can be assigned to."""

    __tablename__ = "operators"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(254), unique=True)  # stored lower-case
    active: Mapped[bool] = mapped_column(default=True, server_default=sa.true())


class Ticket(Base):
    """A customer support ticket."""

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(12), primary_key=True)  # TCK-XXXXXXXX
    title: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(12), default="open", server_default="open")
    customer_id: Mapped[Optional[str]] = mapped_column(String(16))
    assignee_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("operators.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.text("now()")
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(_in_list("status", TICKET_STATUSES), name="ck_tickets_status"),
        CheckConstraint(_in_list("priority", TICKET_PRIORITIES), name="ck_tickets_priority"),
        Index("ix_tickets_status", "status"),
        Index("ix_tickets_created_at", "created_at"),
        Index("ix_tickets_closed_at", "closed_at"),
    )


class Notification(Base):
    """Outbox row: one alert to deliver on one channel."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"))
    channel: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(10), default="pending", server_default="pending")
    attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.text("now()")
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            _in_list("channel", NOTIFICATION_CHANNELS), name="ck_notifications_channel"
        ),
        CheckConstraint(_in_list("status", NOTIFICATION_STATUSES), name="ck_notifications_status"),
        Index("ix_notifications_status_created", "status", "created_at"),
    )
