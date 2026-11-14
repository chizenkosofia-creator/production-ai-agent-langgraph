"""FastAPI application exposing the LangGraph agent.

Architecture decisions
----------------------
* The compiled graph, its Postgres pools and the vector store are created once
  in the app ``lifespan`` and stored on ``app.state.runtime``; endpoints get
  them through FastAPI dependency injection, so tests can swap the runtime.
* Everything is ``async``: ``graph.ainvoke`` / ``graph.astream`` never block
  the event loop.
* Errors are mapped to proper HTTP codes and a uniform ``ErrorResponse``;
  internals and stack traces are logged, never returned to clients.
* Turns on the same ``thread_id`` are serialised with a Postgres advisory lock
  (409 on ``/chat``, an ``error`` event on ``/chat/stream`` if the thread is
  busy), so concurrent submits cannot corrupt a conversation checkpoint.

SSE contract for ``POST /api/v1/chat/stream`` (``text/event-stream``)::

    event: token       data: {"text": "..."}              # incremental answer
    event: tool_start  data: {"name": "crm_lookup"}
    event: tool_end    data: {"name": "crm_lookup"}
    event: done        data: {"thread_id": "...", "tool_calls_used": 1,
                              "sources": ["kb/refunds.md"]}
    event: error       data: {"detail": "..."}            # terminal
    : ping                                               # keep-alive comment

Browsers' ``EventSource`` only supports GET, so use ``fetch`` with a streamed
body; a ready-to-paste example is in README.md.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

from agent.graph import AgentRuntime, graph_lifespan
from agent.locks import thread_lock
from api.schemas import (
    THREAD_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HistoryMessage,
    HistoryResponse,
)
from core.config import Settings, get_settings
from core.observability import MetricsCallback, configure_observability, log_turn_metrics

logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_runtime(request: Request) -> AgentRuntime:
    """Provide the runtime (graph + lock pool) created at startup."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Agent not ready")
    return runtime


RuntimeDep = Annotated[AgentRuntime, Depends(get_runtime)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _content_text(content: Any) -> str:
    """Flatten message content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def _message_text(message: BaseMessage) -> str:
    return _content_text(message.content)


def _tool_calls_this_turn(messages: list[BaseMessage]) -> int:
    """Count ToolMessages after the most recent human message."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            count += 1
    return count


def _graph_input(payload: ChatRequest) -> dict[str, Any]:
    """Per-turn input. ``messages`` is appended by the reducer; the counters
    are reset so each user turn gets a fresh iteration budget and context."""
    return {
        "messages": [HumanMessage(content=payload.message)],
        "iterations": 0,
        "retrieved_context": [],
    }


def _graph_config(
    payload: ChatRequest,
    settings: Settings,
    request_id: str = "-",
    metrics: Optional[MetricsCallback] = None,
) -> dict[str, Any]:
    """LangGraph run config.

    ``recursion_limit`` is the hard stop on supersteps (it raises
    ``GraphRecursionError``); the softer per-turn iteration budget in the graph
    normally ends a runaway tool loop first. ``tags``/``metadata``/``run_name``
    make every LangSmith trace filterable by thread, request and model.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": payload.thread_id},
        "recursion_limit": settings.agent_recursion_limit,
        "run_name": "support_agent_turn",
        "tags": ["support-agent", settings.app_env],
        "metadata": {
            "thread_id": payload.thread_id,
            "request_id": request_id,
            "llm_model": settings.llm_model,
        },
    }
    if metrics is not None:
        config["callbacks"] = [metrics]
    return config


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# SSE machinery
# ---------------------------------------------------------------------------
async def _graph_events(
    runtime: AgentRuntime, payload: ChatRequest, settings: Settings, request_id: str = "-"
) -> AsyncIterator[str]:
    """Run one turn and translate LangGraph stream parts into SSE frames.

    Two stream modes are combined:
    * ``messages`` -> token-level chunks from the LLM (only the ``agent`` node;
      tool output is not echoed as answer text);
    * ``updates``  -> complete per-node deltas, used for reliable
      ``tool_start`` (the AI message's finished tool_calls), ``tool_end`` and
      the retrieved sources.
    """
    tools_used = 0
    sources: list[str] = []
    metrics = MetricsCallback()
    async for mode, data in runtime.graph.astream(
        _graph_input(payload),
        _graph_config(payload, settings, request_id, metrics),
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, meta = data
            if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "agent":
                text = _content_text(chunk.content)
                if text:
                    yield _sse("token", {"text": text})
        elif mode == "updates":
            for node, delta in data.items():
                if not isinstance(delta, dict):
                    continue
                for message in delta.get("messages", []) or []:
                    if node == "agent" and isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            yield _sse("tool_start", {"name": call["name"]})
                    elif node == "tools" and isinstance(message, ToolMessage):
                        tools_used += 1
                        yield _sse("tool_end", {"name": message.name})
                for item in delta.get("retrieved_context", []) or []:
                    if item["source"] not in sources:
                        sources.append(item["source"])
    summary = metrics.summary()
    log_turn_metrics(payload.thread_id, request_id, summary)
    yield _sse(
        "done",
        {
            "thread_id": payload.thread_id,
            "tool_calls_used": tools_used,
            "sources": sources,
            "usage": summary,
        },
    )


async def _with_heartbeat(
    events: AsyncIterator[str], interval: float, timeout: float
) -> AsyncIterator[str]:
    """Interleave ``: ping`` keep-alives and enforce a total deadline.

    The next event is awaited in a task and polled with ``asyncio.wait`` (never
    ``wait_for``) so a quiet period does not cancel the running graph step.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    pending: asyncio.Future[str] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(events))
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            done, _ = await asyncio.wait({pending}, timeout=min(interval, remaining))
            if not done:
                yield ": ping\n\n"
                continue
            task, pending = pending, None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    finally:
        if pending is not None:
            pending.cancel()
        await events.aclose()


async def _stream_turn(
    runtime: AgentRuntime, payload: ChatRequest, settings: Settings, request: Request
) -> AsyncIterator[str]:
    request_id = getattr(request.state, "request_id", "-")
    """Full SSE generator for one chat turn, with locking and error frames.

    Once the first byte is sent an HTTP status can no longer be changed, so
    every failure after this point is reported as a terminal ``error`` event.
    """
    try:
        async with thread_lock(runtime.lock_pool, payload.thread_id) as acquired:
            if not acquired:
                yield _sse("error", {"detail": "Another request is running on this thread."})
                return
            events = _with_heartbeat(
                _graph_events(runtime, payload, settings, request_id),
                settings.sse_heartbeat_seconds,
                settings.request_timeout_seconds,
            )
            async for frame in events:
                if await request.is_disconnected():
                    logger.info("Client disconnected (thread=%s)", payload.thread_id)
                    await events.aclose()  # stops the graph; saved checkpoints persist
                    return
                yield frame
    except GraphRecursionError:
        logger.error("Recursion limit hit (thread=%s)", payload.thread_id)
        yield _sse("error", {"detail": "The request required too many reasoning steps."})
    except TimeoutError:
        logger.error("Agent timed out (thread=%s)", payload.thread_id)
        yield _sse("error", {"detail": "Agent timed out."})
    except Exception:  # noqa: BLE001 - never leak internals to the client
        logger.exception("Stream failed (thread=%s)", payload.thread_id)
        yield _sse("error", {"detail": "Unexpected server error."})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/v1", tags=["agent"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat(
    payload: ChatRequest, request: Request, runtime: RuntimeDep, settings: SettingsDep
) -> ChatResponse:
    """Send a message to the agent and return the complete reply.

    State for ``thread_id`` is loaded from / saved to Postgres automatically,
    so clients only send the *new* user message each time.
    """
    async with thread_lock(runtime.lock_pool, payload.thread_id) as acquired:
        if not acquired:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Another request is running on this thread."
            )
        request_id = getattr(request.state, "request_id", "-")
        metrics = MetricsCallback()
        try:
            result = await asyncio.wait_for(
                runtime.graph.ainvoke(
                    _graph_input(payload),
                    _graph_config(payload, settings, request_id, metrics),
                ),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            logger.error("Agent timed out (thread=%s)", payload.thread_id)
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Agent timed out") from exc
        except GraphRecursionError as exc:
            logger.error("Recursion limit hit (thread=%s)", payload.thread_id)
            raise HTTPException(
                422,
                "The request required too many reasoning steps.",
            ) from exc

    messages: list[BaseMessage] = result["messages"]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    summary = metrics.summary()
    log_turn_metrics(payload.thread_id, request_id, summary)
    return ChatResponse(
        thread_id=payload.thread_id,
        reply=_message_text(final) if final else "",
        tool_calls_used=_tool_calls_this_turn(messages),
        total_messages=len(messages),
        usage=summary,
    )


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_stream(
    payload: ChatRequest, request: Request, runtime: RuntimeDep, settings: SettingsDep
) -> StreamingResponse:
    """Stream the agent's answer token by token as Server-Sent Events."""
    return StreamingResponse(
        _stream_turn(runtime, payload, settings, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
            "Connection": "keep-alive",
        },
    )


@router.get("/threads/{thread_id}/history", response_model=HistoryResponse)
async def get_history(
    runtime: RuntimeDep,
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)],
) -> HistoryResponse:
    """Return the persisted conversation for a thread (404 if unknown)."""
    snapshot = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages: list[BaseMessage] = (snapshot.values or {}).get("messages", [])
    if not messages:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")

    return HistoryResponse(
        thread_id=thread_id,
        messages=[
            HistoryMessage(
                role=m.type if m.type in ("human", "ai", "tool", "system") else "system",
                content=_message_text(m),
                tool_name=m.name if isinstance(m, ToolMessage) else None,
            )
            for m in messages
        ],
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    configure_observability(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with graph_lifespan(settings) as runtime:
            app.state.runtime = runtime
            logger.info("%s started (env=%s)", settings.app_name, settings.app_env)
            yield
        app.state.runtime = None

    app = FastAPI(title=settings.app_name, version="1.1.0", lifespan=lifespan)

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "x-request-id"],
            expose_headers=["x-request-id"],
        )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request ID and log latency for every request."""
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %s (%.0f ms) rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error="http_error", detail=str(exc.detail)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail=json.dumps(
                    [
                        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                        for e in exc.errors()
                    ]
                ),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # Full trace goes to logs; the client only sees a generic message.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error", detail="Unexpected server error"
            ).model_dump(),
        )

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict[str, str]:
        """Liveness/readiness probe used by Docker."""
        if getattr(request.app.state, "runtime", None) is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Agent not ready")
        return {"status": "ok"}

    app.include_router(router)
    return app


app = create_app()
