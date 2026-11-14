"""HTTP-level tests of /chat and /chat/stream against the real graph with a
scripted LLM and an in-memory checkpointer (no database, no network)."""

import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import api.main as api_main
from agent.graph import AgentRuntime, build_graph
from agent.rag import set_vector_store
from core.config import get_settings
from tests.fakes import ScriptedChatModel, StubVectorStore, doc, tool_call_message


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs, dropping ``: ping`` comments."""
    events = []
    for frame in raw.strip().split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def make_client(monkeypatch):
    """Factory: build a TestClient wired to a scripted LLM."""
    clients: list[TestClient] = []

    def _make(
        script: list[AIMessage], lock_free: bool = True
    ) -> tuple[TestClient, ScriptedChatModel]:
        llm = ScriptedChatModel(script=script)
        graph = build_graph(MemorySaver(), get_settings(), llm=llm)

        @asynccontextmanager
        async def fake_lifespan(_settings=None):
            yield AgentRuntime(graph=graph, lock_pool=None)  # type: ignore[arg-type]

        @asynccontextmanager
        async def fake_lock(_pool, _thread_id):
            yield lock_free

        monkeypatch.setattr(api_main, "graph_lifespan", fake_lifespan)
        monkeypatch.setattr(api_main, "thread_lock", fake_lock)
        client = TestClient(api_main.app)
        client.__enter__()
        clients.append(client)
        return client, llm

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def test_stream_emits_tool_and_token_events_in_order(make_client):
    client, _ = make_client(
        [
            tool_call_message("calculator", {"expression": "2+2"}),
            AIMessage("The answer is four, as expected."),
        ]
    )
    r = client.post("/api/v1/chat/stream", json={"thread_id": "s1", "message": "2+2?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"

    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "tool_start" and events[0][1] == {"name": "calculator"}
    assert names[1] == "tool_end"
    tokens = [d["text"] for e, d in events if e == "token"]
    assert len(tokens) > 1, "answer must arrive incrementally, not as one blob"
    assert "".join(tokens) == "The answer is four, as expected."
    assert names[-1] == "done"
    assert events[-1][1]["tool_calls_used"] == 1
    assert events[-1][1]["thread_id"] == "s1"


def test_stream_reports_kb_sources_in_done_event(make_client):
    set_vector_store(StubVectorStore([(doc("Refunds take 5-10 days.", "refund_policy.md"), 0.1)]))
    client, _ = make_client(
        [
            tool_call_message("search_knowledge_base", {"query": "refund timing"}),
            AIMessage("5-10 business days."),
        ]
    )
    r = client.post("/api/v1/chat/stream", json={"thread_id": "s2", "message": "refund time?"})
    done = parse_sse(r.text)[-1]
    assert done[0] == "done" and done[1]["sources"] == ["refund_policy.md"]


def test_stream_follow_up_turn_uses_persisted_thread(make_client):
    client, llm = make_client([AIMessage("Hi Dana."), AIMessage("You are Dana.")])
    client.post("/api/v1/chat/stream", json={"thread_id": "s3", "message": "I am Dana."})
    r = client.post("/api/v1/chat/stream", json={"thread_id": "s3", "message": "Who am I?"})
    assert parse_sse(r.text)[-1][0] == "done"
    assert any("I am Dana." in str(m.content) for m in llm.calls[1])

    history = client.get("/api/v1/threads/s3/history").json()
    assert [m["role"] for m in history["messages"]] == ["human", "ai", "human", "ai"]


def test_busy_thread_is_rejected(make_client):
    client, _ = make_client([AIMessage("never sent")], lock_free=False)
    stream = client.post("/api/v1/chat/stream", json={"thread_id": "busy", "message": "hello"})
    assert parse_sse(stream.text) == [
        ("error", {"detail": "Another request is running on this thread."})
    ]
    plain = client.post("/api/v1/chat", json={"thread_id": "busy", "message": "hello"})
    assert plain.status_code == 409


def test_non_streaming_chat_still_works(make_client):
    client, _ = make_client([AIMessage("Hello there.")])
    r = client.post("/api/v1/chat", json={"thread_id": "p1", "message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Hello there." and body["tool_calls_used"] == 0


def test_validation_errors_use_error_envelope(make_client):
    client, _ = make_client([AIMessage("x")])
    r = client.post("/api/v1/chat/stream", json={"thread_id": "bad id!", "message": "hi"})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_runaway_agent_is_cut_off_in_stream(make_client):
    client, _ = make_client([tool_call_message("calculator", {"expression": "1+1"})])
    r = client.post("/api/v1/chat/stream", json={"thread_id": "loop", "message": "go"})
    events = parse_sse(r.text)
    assert events[-1][0] == "done"
    assert sum(1 for e, _ in events if e == "tool_start") == get_settings().agent_max_iterations
