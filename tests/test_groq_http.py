"""Agent behaviour against a mocked Groq HTTP endpoint (no network, no rate limits).

Unlike the scripted-model tests, these go through the real ``ChatOpenAI`` client:
request construction, authentication, tool-call (de)serialisation, SSE streaming,
token-usage reporting and the retry policy.
"""

import openai
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph, build_llm
from agent.registry import ALL_TOOLS
from core.config import Settings, get_settings
from core.observability import MetricsCallback
from tests.groq_mock import GroqMock, Reply


def _graph(mock: GroqMock, settings: Settings | None = None):
    settings = settings or get_settings()
    llm = build_llm(settings, http_async_client=mock.client())
    return build_graph(MemorySaver(), settings, llm=llm)


def _turn(text: str) -> dict:
    return {"messages": [HumanMessage(text)], "iterations": 0, "retrieved_context": []}


def _config(thread: str, **extra) -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 30, **extra}


async def test_tool_round_trip_hits_the_groq_endpoint_correctly():
    mock = GroqMock(
        [
            Reply(tool_calls=[("calculator", {"expression": "12500 * 12"})], usage=(120, 15)),
            Reply(content="Annual revenue is 150,000 USD.", usage=(180, 12)),
        ]
    )
    result = await _graph(mock).ainvoke(_turn("What is 12500 times 12?"), _config("g1"))

    assert result["messages"][-1].content == "Annual revenue is 150,000 USD."
    assert len(mock.requests) == 2

    first = mock.requests[0]
    settings = get_settings()
    assert str(first.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert first.headers["authorization"] == f"Bearer {settings.llm_api_key.get_secret_value()}"

    body = mock.bodies[0]
    assert body["model"] == settings.llm_model and body["temperature"] == 0
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][-1] == {"role": "user", "content": "What is 12500 times 12?"}
    advertised = {t["function"]["name"] for t in body["tools"]}
    assert advertised == {
        t.name for t in ALL_TOOLS
    }, "every registered tool is offered to the model"

    # the second request carries the assistant tool call AND the executed tool result
    roles = [m["role"] for m in mock.bodies[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    tool_msg = mock.bodies[1]["messages"][-1]
    assert "150,000" in tool_msg["content"] and tool_msg["tool_call_id"] == "call_1"


async def test_streaming_requests_usage_and_yields_tokens():
    mock = GroqMock([Reply(content="Refunds take five to ten days.", usage=(90, 8))])
    graph = _graph(mock)
    metrics = MetricsCallback()

    tokens = []
    async for chunk, meta in graph.astream(
        _turn("refund time?"), _config("g2", callbacks=[metrics]), stream_mode="messages"
    ):
        if meta.get("langgraph_node") == "agent" and chunk.content:
            tokens.append(chunk.content)

    assert len(tokens) > 3 and "".join(tokens) == "Refunds take five to ten days."
    body = mock.bodies[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}, "needed for token accounting"
    assert metrics.summary()["input_tokens"] == 90 and metrics.summary()["output_tokens"] == 8


async def test_streamed_tool_call_is_reassembled_from_deltas():
    mock = GroqMock(
        [
            Reply(tool_calls=[("calculator", {"expression": "2+2"})], usage=(50, 5)),
            Reply(content="Four.", usage=(60, 2)),
        ]
    )
    seen_nodes = []
    async for _, meta in _graph(mock).astream(_turn("2+2?"), _config("g3"), stream_mode="messages"):
        seen_nodes.append(meta.get("langgraph_node"))
    assert "tools" in seen_nodes, "the tool call split across SSE deltas was executed"
    assert any(m["role"] == "tool" and "4" in m["content"] for m in mock.bodies[1]["messages"])


async def test_rate_limit_is_retried_transparently():
    mock = GroqMock([Reply(status=429), Reply(content="Back online.")])
    result = await _graph(mock).ainvoke(_turn("hello"), _config("g4"))
    assert result["messages"][-1].content == "Back online."
    assert len(mock.requests) == 2, "one 429 + one success = a single retry"


async def test_persistent_provider_errors_stop_after_the_configured_retries():
    settings = Settings(llm_max_retries=2)
    mock = GroqMock([Reply(status=500)])
    with pytest.raises(openai.InternalServerError):
        await _graph(mock, settings).ainvoke(_turn("hello"), _config("g5"))
    assert len(mock.requests) == 1 + settings.llm_max_retries


async def test_auth_failure_is_not_retried():
    mock = GroqMock([Reply(status=401)])
    with pytest.raises(openai.AuthenticationError):
        await _graph(mock).ainvoke(_turn("hello"), _config("g6"))
    assert len(mock.requests) == 1
