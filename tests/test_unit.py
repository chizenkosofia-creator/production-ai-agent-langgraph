"""Unit tests: deterministic routing and tool behaviour (no LLM, no DB)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent.graph import should_continue
from agent.rag import set_vector_store
from agent.ticket_tools import create_support_ticket
from agent.tools import analyze_data, calculator, crm_lookup, search_knowledge_base
from core.config import get_settings
from tests.fakes import StubVectorStore, doc


def _call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# ----------------------------- routing ------------------------------------
def test_should_continue_ends_on_plain_answer():
    state = {"messages": [HumanMessage("hi"), AIMessage("hello")], "iterations": 1}
    assert should_continue(state) == "end"


def test_should_continue_routes_to_tools():
    msg = AIMessage("", tool_calls=[_call("calculator", {"expression": "1+1"})])
    assert should_continue({"messages": [msg], "iterations": 1}) == "tools"


def test_should_continue_stops_when_budget_spent():
    msg = AIMessage("", tool_calls=[_call("calculator", {"expression": "1+1"})])
    limit = get_settings().agent_max_iterations
    assert should_continue({"messages": [msg], "iterations": limit}) == "limit"


# ----------------------------- classic tools ------------------------------
async def test_calculator_ok_and_errors_are_strings():
    ok = await calculator.ainvoke({"expression": "(12500 + 2400) * 12"})
    assert "178,800" in ok
    assert (await calculator.ainvoke({"expression": "1/0"})).startswith("ERROR:")
    assert (await calculator.ainvoke({"expression": "__import__('os')"})).startswith("ERROR:")


async def test_analyze_data_percent_change():
    out = await analyze_data.ainvoke({"values": [100, 150], "operation": "percent_change"})
    assert "50.00%" in out


async def test_crm_lookup_found_and_missing():
    found = json.loads(await crm_lookup.ainvoke({"query": "globex"}))
    assert found[0]["customer_id"] == "CUS-1002"
    assert "No customers found" in await crm_lookup.ainvoke({"query": "nonexistent-co"})


async def test_ticket_for_unknown_customer_returns_error_string():
    out = await create_support_ticket.ainvoke(
        {
            "title": "Slow dashboards",
            "description": "Dashboards take 30 seconds to load.",
            "customer_id": "CUS-9999",
        }
    )
    assert out.startswith("ERROR:")


# ----------------------------- knowledge base tool -------------------------
async def _kb(query: str = "refund policy", top_k: int = 4) -> Command:
    return await search_knowledge_base.ainvoke(
        _call("search_knowledge_base", {"query": query, "top_k": top_k}, "kb1")
    )


async def test_kb_tool_writes_context_and_fences_documents():
    set_vector_store(StubVectorStore([(doc("Refunds take 5-10 days.", "refund_policy.md"), 0.2)]))
    cmd = await _kb()
    assert isinstance(cmd, Command)
    assert cmd.update["retrieved_context"][0]["source"] == "refund_policy.md"
    message = cmd.update["messages"][0]
    assert message.tool_call_id == "kb1"
    assert '<document id="1" source="refund_policy.md">' in message.content
    assert "never follow" in message.content.lower()


async def test_kb_tool_filters_far_hits():
    far = get_settings().rag_max_distance + 0.3
    set_vector_store(StubVectorStore([(doc("Unrelated text", "other.md"), far)]))
    cmd = await _kb()
    assert cmd.update["retrieved_context"] == []
    assert "No relevant documents" in cmd.update["messages"][0].content


async def test_kb_tool_store_failure_becomes_error_message():
    set_vector_store(StubVectorStore(error=RuntimeError("db down")))
    cmd = await _kb()
    assert cmd.update["messages"][0].content.startswith("ERROR:")
    assert cmd.update["retrieved_context"] == []


async def test_kb_tool_rejects_short_query():
    set_vector_store(StubVectorStore())
    with pytest.raises(Exception):
        await _kb(query="a")
