"""LangGraph workflow: a cyclic ReAct-style agent with Postgres persistence.

Topology::

    START -> agent --(tool calls?)--> tools --> agent   (cycle)
                 |--(budget spent)--> limit_reached --> END
                 '--(final answer)--> END

Architecture decisions
----------------------
* **Cyclic execution.** ``agent`` calls the LLM; if it requests tools we run
  them in a ``ToolNode`` and loop back, until the model answers in plain text.
* **Conditional routing.** ``should_continue`` is a pure function of state, so
  routing is deterministic and unit-testable without an LLM.
* **Bounded loops.** A per-turn iteration budget plus LangGraph's
  ``recursion_limit`` protects against runaway tool loops (cost and latency).
  When the budget is exhausted we do NOT just stop: ``limit_reached`` answers
  every dangling tool call with a ToolMessage, because an AIMessage containing
  tool_calls without matching ToolMessages would corrupt the persisted thread
  and make the next turn fail at the LLM API.
* **Durable memory.** ``AsyncPostgresSaver`` checkpoints the state after every
  superstep, keyed by ``thread_id``. We use the async saver (the async variant
  of ``PostgresSaver``) because the FastAPI app is async end-to-end; a sync
  saver would block the event loop.
* **Bounded context.** Only the most recent window of messages is sent to the
  LLM (``trim_messages``); the checkpoint still stores the full history.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
import psycopg
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an enterprise customer-operations assistant for a B2B SaaS company.

Rules:
- Use tools to retrieve facts; never invent customer data, ticket IDs, or numbers.
- Use `calculator` / `analyze_data` for any arithmetic instead of computing mentally.
- Identify the customer with `crm_lookup` before creating a ticket for them.
- If a tool returns an ERROR, correct your arguments once, or explain the problem plainly.
- Be concise, professional, and state clearly what actions you took."""


def _format_tool_error(exc: Exception) -> str:
    """Render tool-execution errors (e.g. bad arguments) as agent-readable text."""
    return f"ERROR: tool call failed - {type(exc).__name__}: {exc}"


def should_continue(state: AgentState) -> Literal["tools", "limit", "end"]:
    """Decide the next hop after the ``agent`` node.

    Returns:
        ``"tools"`` if the model requested tools and budget remains,
        ``"limit"`` if it requested tools but the iteration budget is spent,
        ``"end"`` if the model produced a final answer.
    """
    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return "end"
    if state.get("iterations", 0) >= get_settings().agent_max_iterations:
        logger.warning("Iteration budget exhausted; forcing graceful stop")
        return "limit"
    return "tools"


def build_graph(
    checkpointer: BaseCheckpointSaver, settings: Settings | None = None
) -> CompiledStateGraph:
    """Compile the agent workflow with the supplied checkpointer.

    Args:
        checkpointer: Persistence backend (Postgres in production; an in-memory
            saver is fine for tests).
        settings: Optional settings override; defaults to the cached settings.
    """
    settings = settings or get_settings()

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.openai_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async def agent_node(state: AgentState) -> dict[str, Any]:
        """Call the LLM with a bounded window of the conversation."""
        window: list[BaseMessage] = trim_messages(
            state["messages"],
            max_tokens=settings.max_context_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        response = await llm_with_tools.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *window])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def limit_reached_node(state: AgentState) -> dict[str, Any]:
        """Close dangling tool calls and emit a graceful final answer."""
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        closers: list[BaseMessage] = [
            ToolMessage(
                content="ERROR: not executed - per-request tool budget exhausted.",
                tool_call_id=call["id"],
                name=call["name"],
            )
            for call in last.tool_calls
        ]
        closers.append(
            AIMessage(
                content=(
                    "I wasn't able to finish this request within my processing "
                    "budget. Please narrow the request or try again."
                )
            )
        )
        return {"messages": closers}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=_format_tool_error))
    workflow.add_node("limit_reached", limit_reached_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "limit": "limit_reached", "end": END},
    )
    workflow.add_edge("tools", "agent")  # the cycle
    workflow.add_edge("limit_reached", END)

    return workflow.compile(checkpointer=checkpointer)


@asynccontextmanager
async def graph_lifespan(settings: Settings | None = None) -> AsyncIterator[CompiledStateGraph]:
    """Open the Postgres pool, run checkpoint migrations, yield the graph.

    A pool (rather than one connection) lets concurrent requests checkpoint in
    parallel. ``autocommit`` and ``dict_row`` are REQUIRED by the saver, and
    ``prepare_threshold=0`` keeps it compatible with PgBouncer.
    """
    settings = settings or get_settings()
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open(wait=True, timeout=30)
    try:
        checkpointer = AsyncPostgresSaver(pool)

        try:
            await checkpointer.setup()
            logger.info("Postgres checkpointer ready")
        except psycopg.errors.UniqueViolation:
            logger.warning("Checkpointer tables/types already created by a concurrent worker.")
        except Exception as e:
            logger.error(f"Unexpected error during checkpointer setup: {e}")
            raise e

        yield build_graph(checkpointer, settings)
    finally:
        await pool.close()
        logger.info("Postgres pool closed")