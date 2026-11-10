"""FastAPI application exposing the LangGraph agent.

Architecture decisions
----------------------
* The compiled graph (and its Postgres pool) is created once in the app
  ``lifespan`` and stored on ``app.state``; endpoints receive it through
  FastAPI dependency injection, so tests can swap in a stub graph via
  ``app.dependency_overrides``.
* Everything is ``async``: ``graph.ainvoke`` never blocks the event loop.
* Errors are mapped to proper HTTP codes and a uniform ``ErrorResponse``;
  internals and stack traces are logged, never returned to clients.
* Concurrency note: two simultaneous requests on the *same* ``thread_id``
  race on the checkpoint. Serialise per-thread calls at the client or gateway
  (or add a Postgres advisory lock) if your UI can double-submit.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from agent.graph import graph_lifespan
from api.schemas import (
    THREAD_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HistoryMessage,
    HistoryResponse,
)
from core.config import Settings, get_settings

logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_graph(request: Request) -> CompiledStateGraph:
    """Provide the compiled graph created at startup."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Agent not ready")
    return graph


GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _message_text(message: BaseMessage) -> str:
    """Flatten message content (str or list of content blocks) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = [
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    ]
    return "".join(parts)


def _tool_calls_this_turn(messages: list[BaseMessage]) -> int:
    """Count ToolMessages after the most recent human message."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            count += 1
    return count


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/v1", tags=["agent"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def chat(payload: ChatRequest, graph: GraphDep, settings: SettingsDep) -> ChatResponse:
    """Send a message to the agent and return its reply.

    State for ``thread_id`` is loaded from / saved to Postgres automatically,
    so clients only send the *new* user message each time.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": payload.thread_id},
        "recursion_limit": settings.agent_recursion_limit,
    }
    # `iterations` is reset every turn; `messages` is appended by the reducer.
    graph_input = {"messages": [HumanMessage(content=payload.message)], "iterations": 0}

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(graph_input, config), timeout=settings.request_timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        logger.error("Agent timed out (thread=%s)", payload.thread_id)
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Agent timed out") from exc
    except GraphRecursionError as exc:
        logger.error("Recursion limit hit (thread=%s)", payload.thread_id)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The request required too many reasoning steps.",
        ) from exc

    messages: list[BaseMessage] = result["messages"]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    reply = _message_text(final) if final else ""

    return ChatResponse(
        thread_id=payload.thread_id,
        reply=reply,
        tool_calls_used=_tool_calls_this_turn(messages),
        total_messages=len(messages),
    )


@router.get("/threads/{thread_id}/history", response_model=HistoryResponse)
async def get_history(
    graph: GraphDep,
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)],
) -> HistoryResponse:
    """Return the persisted conversation for a thread (404 if unknown)."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with graph_lifespan(settings) as graph:
            app.state.graph = graph
            logger.info("%s started (env=%s)", settings.app_name, settings.app_env)
            yield
        app.state.graph = None

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request ID and log latency for every request."""
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(error="validation_error", detail=str(exc.errors())).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # Full trace goes to logs; the client only sees a generic message.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="internal_error", detail="Unexpected server error").model_dump(),
        )

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict[str, str]:
        """Liveness/readiness probe used by Docker."""
        ready = getattr(request.app.state, "graph", None) is not None
        if not ready:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Agent not ready")
        return {"status": "ok"}

    app.include_router(router)
    return app


app = create_app()
