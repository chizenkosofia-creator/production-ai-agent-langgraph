"""Enterprise tools exposed to the agent.

Design principles
-----------------
1. **Typed inputs.** Every tool declares a Pydantic v2 ``args_schema``. The LLM
   sees a precise JSON schema and invalid arguments are rejected before our
   code runs.
2. **Tools never crash the graph.** Each tool is wrapped by ``safe_tool``,
   which converts any exception into a short, actionable ``"ERROR: ..."``
   string. The LLM reads that string as a normal tool result and can retry
   with corrected arguments or explain the problem to the user. Unexpected
   exceptions are logged with a stack trace but only a generic message is
   returned so internals never leak into the conversation.
3. **Simulated backends.** The CRM and ticketing backends are in-memory
   stand-ins. Replace the data-access functions with real clients (Salesforce,
   HubSpot, Jira, Zendesk, ...) without touching the tool contracts.
"""

import ast
import functools
import json
import logging
import operator
import statistics
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, EmailStr, Field, ValidationError

logger = logging.getLogger(__name__)


class ToolInputError(ValueError):
    """Raised for caller mistakes the LLM can fix by changing its arguments."""


def safe_tool(func: Callable[..., str]) -> Callable[..., str]:
    """Wrap a tool function so failures become clean strings for the agent."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except (ToolInputError, ValidationError) as exc:
            # Expected, fixable error: tell the model exactly what was wrong.
            logger.warning("Tool %s rejected input: %s", func.__name__, exc)
            return f"ERROR: invalid input - {exc}"
        except Exception:  # noqa: BLE001 - last line of defence by design
            logger.exception("Tool %s failed unexpectedly", func.__name__)
            return (
                "ERROR: the tool failed due to an internal problem. "
                "Do not retry more than once; tell the user the lookup is "
                "temporarily unavailable."
            )

    return wrapper


class CustomerRecord(BaseModel):
    """A CRM customer record."""

    customer_id: str
    name: str
    email: EmailStr
    plan: Literal["starter", "growth", "enterprise"]
    mrr_usd: float = Field(ge=0)
    status: Literal["active", "churn_risk", "churned"]
    open_tickets: int = Field(ge=0)


# Stand-in for a real database table.
_CRM_DB: tuple[CustomerRecord, ...] = (
    CustomerRecord(
        customer_id="CUS-1001",
        name="Acme Logistics",
        email="ops@acme-logistics.example",
        plan="enterprise",
        mrr_usd=12500.0,
        status="active",
        open_tickets=1,
    ),
    CustomerRecord(
        customer_id="CUS-1002",
        name="Globex Retail",
        email="it@globex-retail.example",
        plan="growth",
        mrr_usd=2400.0,
        status="churn_risk",
        open_tickets=4,
    ),
    CustomerRecord(
        customer_id="CUS-1003",
        name="Initech Software",
        email="admin@initech.example",
        plan="starter",
        mrr_usd=299.0,
        status="active",
        open_tickets=0,
    ),
)


class CRMLookupInput(BaseModel):
    """Arguments for ``crm_lookup``."""

    query: str = Field(
        min_length=2,
        max_length=100,
        description="Customer ID (e.g. CUS-1001), company name, or email fragment.",
    )
    limit: int = Field(default=3, ge=1, le=10, description="Maximum records to return.")


@tool("crm_lookup", args_schema=CRMLookupInput)
@safe_tool
def crm_lookup(query: str, limit: int = 3) -> str:
    """Search the CRM for customer accounts by ID, company name, or email.

    Returns a JSON list of matching customer records including plan, monthly
    recurring revenue, health status, and the number of open support tickets.
    """
    needle = query.strip().lower()
    matches = [
        record
        for record in _CRM_DB
        if needle in record.customer_id.lower()
        or needle in record.name.lower()
        or needle in str(record.email).lower()
    ][:limit]

    if not matches:
        return f"No customers found matching '{query}'. Try a different spelling or ID."
    return json.dumps([m.model_dump(mode="json") for m in matches], indent=2)


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node.

    We deliberately avoid ``eval``: only numeric literals and basic arithmetic
    operators are allowed, so there is no code-execution surface.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolInputError("exponent too large (max absolute value is 100)")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolInputError("only numbers and + - * / // % ** ( ) are allowed")


class CalculatorInput(BaseModel):
    """Arguments for ``calculator``."""

    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic expression, e.g. '(12500 + 2400) * 12 * 0.9'.",
    )


@tool("calculator", args_schema=CalculatorInput)
@safe_tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression exactly (no LLM mental math).

    Supports + - * / // % ** and parentheses. Use it for any revenue, discount,
    or growth computation where numerical accuracy matters.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolInputError(f"malformed expression: {exc.msg}") from exc
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolInputError("division by zero") from exc
    return f"{expression.strip()} = {result:,.6g}"


class AnalyzeDataInput(BaseModel):
    """Arguments for ``analyze_data``."""

    values: list[float] = Field(
        min_length=1,
        max_length=10_000,
        description="Numeric series, in chronological order for percent_change.",
    )
    operation: Literal["sum", "mean", "median", "min", "max", "stdev", "percent_change"]


@tool("analyze_data", args_schema=AnalyzeDataInput)
@safe_tool
def analyze_data(values: list[float], operation: str) -> str:
    """Run a statistical operation over a numeric series.

    Operations: sum, mean, median, min, max, stdev (sample), and
    percent_change (first value -> last value, e.g. MRR growth).
    """
    if operation == "stdev" and len(values) < 2:
        raise ToolInputError("stdev needs at least 2 values")
    if operation == "percent_change":
        if len(values) < 2:
            raise ToolInputError("percent_change needs at least 2 values")
        if values[0] == 0:
            raise ToolInputError("percent_change is undefined when the first value is 0")
        result = (values[-1] - values[0]) / abs(values[0]) * 100
        return f"percent_change over {len(values)} points = {result:.2f}%"

    funcs: dict[str, Callable[[list[float]], float]] = {
        "sum": sum,
        "mean": statistics.fmean,
        "median": statistics.median,
        "min": min,
        "max": max,
        "stdev": statistics.stdev,
    }
    return f"{operation} of {len(values)} values = {funcs[operation](values):,.6g}"


class CreateTicketInput(BaseModel):
    """Arguments for ``create_support_ticket``."""

    title: str = Field(min_length=5, max_length=120)
    description: str = Field(min_length=10, max_length=2000)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    customer_id: Optional[str] = Field(
        default=None,
        pattern=r"^CUS-\d{4}$",
        description="Related CRM customer ID, if known (format CUS-1234).",
    )


_TICKETS: dict[str, dict[str, Any]] = {}
_TICKETS_LOCK = threading.Lock()


@tool("create_support_ticket", args_schema=CreateTicketInput)
@safe_tool
def create_support_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    customer_id: Optional[str] = None,
) -> str:
    """Create a support ticket in the ticketing system and return its ID.

    Only call this after confirming the issue details with the user. Include
    the customer_id when the account has been identified via crm_lookup.
    """
    if customer_id and not any(r.customer_id == customer_id for r in _CRM_DB):
        raise ToolInputError(f"unknown customer_id {customer_id}; verify with crm_lookup")

    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "customer_id": customer_id,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _TICKETS_LOCK:
        _TICKETS[ticket_id] = ticket
    logger.info("Created ticket %s (priority=%s)", ticket_id, priority)
    return json.dumps(ticket, indent=2)


ALL_TOOLS: list[BaseTool] = [crm_lookup, calculator, analyze_data, create_support_ticket]
