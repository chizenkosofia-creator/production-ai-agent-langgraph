"""Integration tests for the database-backed ticket tools and analytics.

Run against a real Postgres (see ``tests/conftest.py``); no LLM involved.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import select, text

from agent.ticket_tools import (
    create_support_ticket,
    list_operators,
    ticket_analytics,
    update_ticket,
)
from db.models import Notification, Ticket
from services import tickets

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
ISSUE = {"title": "Dashboards are slow", "description": "Dashboards take 30 seconds to load."}


async def _make(factory, *, created, priority="medium", status="open", closed=None, now=NOW):
    """Insert a ticket with controlled timestamps and return its id."""
    async with factory() as session:
        async with session.begin():
            ticket = await tickets.create_ticket(
                session,
                title="Seeded ticket",
                description="Seeded for analytics tests.",
                priority=priority,
                now=created,
            )
            if status != "open":
                await tickets.update_ticket(
                    session, ticket.id, status=status, now=closed or created
                )
            return ticket.id


# ------------------------------ create ---------------------------------------
async def test_create_ticket_persists_and_normal_priority_sends_no_alert(db):
    out = json.loads(await create_support_ticket.ainvoke({**ISSUE, "priority": "high"}))
    assert out["ticket_id"].startswith("TCK-") and out["status"] == "open"
    assert out["alerts_queued"] == 0
    async with db() as session:
        assert (await session.get(Ticket, out["ticket_id"])).priority == "high"
        assert (await session.execute(select(Notification))).first() is None


async def test_create_ticket_rejects_unknown_customer_and_bad_priority(db):
    bad_customer = await create_support_ticket.ainvoke({**ISSUE, "customer_id": "CUS-9999"})
    assert bad_customer.startswith("ERROR:")
    with pytest.raises(ValidationError):
        await create_support_ticket.ainvoke({**ISSUE, "priority": "catastrophic"})
    async with db() as session:
        assert (await session.execute(select(Ticket))).first() is None


async def test_ticket_text_is_normalised_before_storage(db):
    """Zero-width characters and look-alike unicode are stripped by SafeText."""
    sneaky = {"title": "Pa​ge  is down", "description": "Ｓｅｒｖｅｒ returns errors constantly."}
    out = json.loads(await create_support_ticket.ainvoke(sneaky))
    assert out["title"] == "Page  is down"
    assert out["description"].startswith("Server returns")


# ------------------------------ analytics ------------------------------------
async def test_weekly_stats_windows_status_buckets_and_backlog(db):
    d = lambda days: NOW - timedelta(days=days)  # noqa: E731
    await _make(db, created=d(1), priority="high")  # open, this week
    await _make(db, created=d(3), status="closed", closed=d(2))  # created + closed this week
    await _make(db, created=d(10), status="closed", closed=d(1))  # old, resolved this week
    await _make(db, created=d(8), priority="low")  # open, last week
    await _make(db, created=d(7), priority="urgent")  # exactly on the boundary
    await _make(db, created=d(2), status="in_progress", closed=None)

    async with db() as session:
        this_week = await tickets.weekly_stats(session, now=NOW)
        last_week = await tickets.weekly_stats(session, now=NOW, week_offset=1)

    assert this_week["created_in_window"] == {
        "open": 1,
        "in_progress": 1,
        "closed": 1,
        "total": 3,
    }
    assert this_week["resolved_in_window"] == 2
    # window is (start, end]: the ticket created exactly 7 days ago is NOT in week 0 ...
    assert this_week["backlog_now"] == {
        "high": 1,
        "low": 1,
        "medium": 1,
        "urgent": 1,
        "total": 4,
    }
    # ... but is the newest ticket of week 1
    assert last_week["created_in_window"] == {
        "open": 2,
        "in_progress": 0,
        "closed": 1,
        "total": 3,
    }
    assert last_week["resolved_in_window"] == 0


async def test_analytics_tool_returns_json_for_the_real_clock(db):
    real_now = datetime.now(timezone.utc)
    await _make(db, created=real_now - timedelta(days=1))
    await _make(db, created=real_now - timedelta(days=2), status="closed", closed=real_now)
    stats = json.loads(await ticket_analytics.ainvoke({"week_offset": 0}))
    assert stats["created_in_window"]["total"] == 2
    assert stats["created_in_window"]["closed"] == 1
    assert stats["resolved_in_window"] == 1
    assert json.loads(await ticket_analytics.ainvoke({}))["created_in_window"]["total"] == 2


async def test_analytics_empty_database_is_all_zero(db):
    stats = json.loads(await ticket_analytics.ainvoke({}))
    assert stats["created_in_window"] == {"open": 0, "in_progress": 0, "closed": 0, "total": 0}
    assert stats["backlog_now"] == {"total": 0}


# ------------------------------ update / assign ------------------------------
async def test_status_lifecycle_stamps_and_clears_closed_at(db):
    tid = json.loads(await create_support_ticket.ainvoke(ISSUE))["ticket_id"]

    closed = json.loads(await update_ticket.ainvoke({"ticket_id": tid, "status": "closed"}))
    assert closed["status"] == "closed" and closed["closed_at"] is not None

    reopened = json.loads(await update_ticket.ainvoke({"ticket_id": tid, "status": "in_progress"}))
    assert reopened["status"] == "in_progress" and reopened["closed_at"] is None


async def test_assignment_is_case_insensitive_and_persisted(db):
    tid = json.loads(await create_support_ticket.ainvoke(ISSUE))["ticket_id"]
    out = json.loads(
        await update_ticket.ainvoke(
            {"ticket_id": tid, "assignee_email": "MAYA.COHEN@support.example"}
        )
    )
    assert out["assignee"] == "maya.cohen@support.example"
    async with db() as session:
        assert (await session.get(Ticket, tid)).assignee_id is not None


async def test_update_errors_are_strings_the_agent_can_act_on(db):
    tid = json.loads(await create_support_ticket.ainvoke(ISSUE))["ticket_id"]
    missing = await update_ticket.ainvoke({"ticket_id": "TCK-DEADBEEF", "status": "closed"})
    nobody = await update_ticket.ainvoke(
        {"ticket_id": tid, "assignee_email": "ghost@nowhere.example"}
    )
    assert missing.startswith("ERROR:") and "not found" in missing
    assert nobody.startswith("ERROR:") and "list_operators" in nobody

    async with db() as session:
        await session.execute(text("UPDATE operators SET active = false WHERE name = 'Liam Novak'"))
        await session.commit()
    inactive = await update_ticket.ainvoke(
        {"ticket_id": tid, "assignee_email": "liam.novak@support.example"}
    )
    assert inactive.startswith("ERROR:") and "inactive" in inactive


async def test_update_requires_a_change_and_a_well_formed_id(db):
    with pytest.raises(ValidationError):
        await update_ticket.ainvoke({"ticket_id": "TCK-1A2B3C4D"})
    with pytest.raises(ValidationError):
        await update_ticket.ainvoke({"ticket_id": "1; DROP TABLE tickets", "status": "closed"})


async def test_concurrent_updates_serialise_without_losing_either_change(db):
    tid = json.loads(await create_support_ticket.ainvoke(ISSUE))["ticket_id"]
    await asyncio.gather(
        update_ticket.ainvoke({"ticket_id": tid, "status": "closed"}),
        update_ticket.ainvoke({"ticket_id": tid, "assignee_email": "aisha.rahman@support.example"}),
    )
    async with db() as session:
        row = await session.get(Ticket, tid)
        assert row.status == "closed" and row.assignee_id is not None


async def test_list_operators_returns_active_operators(db):
    operators = json.loads(await list_operators.ainvoke({}))
    assert {o["email"] for o in operators} >= {"maya.cohen@support.example"}
    assert all(set(o) == {"name", "email"} for o in operators)
