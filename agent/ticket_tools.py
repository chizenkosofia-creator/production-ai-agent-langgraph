"""Database-backed ticket tools: create, update/assign, analytics, list operators.

Every tool opens its own short transaction through the shared session factory,
validates input with Pydantic (free-text is normalised via ``SafeText``), and is
wrapped by ``safe_tool`` so failures come back as ``ERROR:`` strings.
"""

import json
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from agent.tool_utils import ToolInputError, safe_tool
from agent.tools import customer_exists
from core.guardrails import SafeText
from db.session import get_sessionmaker
from services import notifications, tickets

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


# ----------------------------- create ticket -------------------------------------
class CreateTicketInput(BaseModel):
    """Arguments for ``create_support_ticket``."""

    title: SafeText = Field(min_length=5, max_length=120)
    description: SafeText = Field(min_length=10, max_length=2000)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    customer_id: Optional[str] = Field(
        default=None,
        pattern=r"^CUS-\d{4}$",
        description="Related CRM customer ID, if known (format CUS-1234).",
    )


@tool("create_support_ticket", args_schema=CreateTicketInput)
@safe_tool
async def create_support_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    customer_id: Optional[str] = None,
) -> str:
    """Create a support ticket and return it.

    Only call this after confirming the issue details with the user. Include the
    customer_id when the account was identified via crm_lookup. Tickets with
    priority 'urgent' automatically alert the on-call team by email and Slack.
    """
    if customer_id and not customer_exists(customer_id):
        raise ToolInputError(f"unknown customer_id {customer_id}; verify with crm_lookup")

    alert_ids: list[int] = []
    async with get_sessionmaker()() as session:
        async with session.begin():  # ticket + alert rows commit atomically
            ticket = await tickets.create_ticket(
                session,
                title=title,
                description=description,
                priority=priority,
                customer_id=customer_id,
            )
            if priority == "urgent":
                alert_ids = await notifications.queue_urgent_alert(session, ticket)
            result = tickets.ticket_to_dict(ticket)

    # Publish only after the commit, so a worker never sees a missing row.
    result["alerts_queued"] = await notifications.dispatch_after_commit(alert_ids)
    return json.dumps(result, indent=2)


# ----------------------------- update / assign -----------------------------------
class UpdateTicketInput(BaseModel):
    """Arguments for ``update_ticket``."""

    ticket_id: str = Field(pattern=r"^TCK-[0-9A-F]{8}$", description="e.g. TCK-1A2B3C4D")
    status: Optional[Literal["open", "in_progress", "closed"]] = None
    assignee_email: Optional[str] = Field(
        default=None,
        pattern=EMAIL_PATTERN,
        max_length=254,
        description="Email of an active support operator (see list_operators).",
    )

    @model_validator(mode="after")
    def _needs_a_change(self) -> "UpdateTicketInput":
        if self.status is None and self.assignee_email is None:
            raise ValueError("provide a new status and/or an assignee_email")
        return self


@tool("update_ticket", args_schema=UpdateTicketInput)
@safe_tool
async def update_ticket(
    ticket_id: str,
    status: Optional[str] = None,
    assignee_email: Optional[str] = None,
) -> str:
    """Change a ticket's status (open / in_progress / closed) and/or assign it to
    a support operator by email. Closing stamps the close time; reopening clears it."""
    async with get_sessionmaker()() as session:
        async with session.begin():
            ticket, assignee = await tickets.update_ticket(
                session, ticket_id, status=status, assignee_email=assignee_email
            )
            return json.dumps(tickets.ticket_to_dict(ticket, assignee), indent=2)


# ----------------------------- operators -----------------------------------------
class ListOperatorsInput(BaseModel):
    """``list_operators`` takes no arguments."""


@tool("list_operators", args_schema=ListOperatorsInput)
@safe_tool
async def list_operators() -> str:
    """List active support operators (name and email) that tickets can be assigned to."""
    async with get_sessionmaker()() as session:
        operators = await tickets.list_active_operators(session)
    if not operators:
        return "No active operators are configured."
    return json.dumps([{"name": o.name, "email": o.email} for o in operators], indent=2)


# ----------------------------- analytics -----------------------------------------
class TicketAnalyticsInput(BaseModel):
    """Arguments for ``ticket_analytics``."""

    week_offset: int = Field(
        default=0,
        ge=0,
        le=52,
        description="0 = the 7 days ending now, 1 = the 7 days before that, and so on.",
    )


@tool("ticket_analytics", args_schema=TicketAnalyticsInput)
@safe_tool
async def ticket_analytics(week_offset: int = 0) -> str:
    """Weekly ticket statistics: tickets created in the window by status
    (open / in_progress / closed), tickets resolved in the window, and the
    current backlog by priority."""
    async with get_sessionmaker()() as session:
        stats = await tickets.weekly_stats(session, week_offset=week_offset)
    return json.dumps(stats, indent=2)


TICKET_TOOLS = [create_support_ticket, update_ticket, list_operators, ticket_analytics]
