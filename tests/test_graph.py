"""Graph-level tests: real StateGraph + ToolNode + in-memory checkpointer,
with a scripted fake LLM (no Groq calls)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph
from agent.rag import set_vector_store
from core.config import get_settings
from tests.fakes import ScriptedChatModel, StubVectorStore, doc, tool_call_message


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 30}


def _input(text: str) -> dict:
    return {"messages": [HumanMessage(text)], "iterations": 0, "retrieved_context": []}


async def test_rag_tool_result_lands_in_agent_state():
    set_vector_store(StubVectorStore([(doc("Refunds take 5-10 days.", "refund_policy.md"), 0.1)]))
    llm = ScriptedChatModel(
        script=[
            tool_call_message("search_knowledge_base", {"query": "refund timing"}),
            AIMessage("Refunds take 5-10 business days (refund_policy.md)."),
        ]
    )
    graph = build_graph(MemorySaver(), get_settings(), llm=llm)

    result = await graph.ainvoke(_input("How long do refunds take?"), _config("t-rag"))

    assert result["retrieved_context"][0]["source"] == "refund_policy.md"
    assert result["messages"][-1].content.startswith("Refunds take")
    # the model saw the fenced passage in its second prompt
    second_prompt = llm.calls[1]
    assert any("refund_policy.md" in str(m.content) for m in second_prompt)


async def test_conversation_memory_is_keyed_by_thread_id():
    llm = ScriptedChatModel(script=[AIMessage("Nice to meet you, Dana."), AIMessage("Dana.")])
    graph = build_graph(MemorySaver(), get_settings(), llm=llm)

    await graph.ainvoke(_input("My name is Dana."), _config("thread-a"))
    await graph.ainvoke(_input("What is my name?"), _config("thread-a"))
    texts = [str(m.content) for m in llm.calls[1]]
    assert any("My name is Dana." in t for t in texts), "turn 2 must see turn 1"

    # a different thread starts clean
    llm.cursor = 0
    await graph.ainvoke(_input("What is my name?"), _config("thread-b"))
    texts_b = [str(m.content) for m in llm.calls[2]]
    assert not any("My name is Dana." in t for t in texts_b)


async def test_runaway_tool_loop_stops_gracefully_with_consistent_history():
    """A model that calls tools forever must hit the budget, not the recursion
    limit, and the stored thread must not contain unanswered tool calls."""
    llm = ScriptedChatModel(script=[tool_call_message("calculator", {"expression": "1+1"})])
    graph = build_graph(MemorySaver(), get_settings(), llm=llm)

    result = await graph.ainvoke(_input("loop forever"), _config("t-loop"))

    assert "wasn't able to finish" in result["messages"][-1].content
    assert len(llm.calls) == get_settings().agent_max_iterations
    answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    requested = {
        c["id"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    }
    # every tool call id is answered (ids repeat in this fake, so compare sets)
    assert requested <= answered

    # and the thread is still usable on the next turn
    llm.script = [AIMessage("Back to normal.")]
    llm.cursor = 0
    again = await graph.ainvoke(_input("hello again"), _config("t-loop"))
    assert again["messages"][-1].content == "Back to normal."
