"""Observability: LangSmith bindings and a per-turn latency / token-usage collector.

* ``configure_observability`` maps our validated settings onto the environment
  variables LangChain / LangSmith read, so every graph node, LLM call and tool
  call is traced automatically when ``LANGSMITH_TRACING=true``. Tracing is
  explicitly switched OFF otherwise, so a stray variable in the host
  environment can never leak prompts to a third party.
* ``MetricsCallback`` is a LangChain callback handler attached to every turn. It
  works with or without LangSmith and produces a compact summary (LLM calls,
  token usage, per-node and per-tool latency) that is logged as JSON and
  returned to API clients.
"""

import json
import logging
import os
import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from core.config import Settings

logger = logging.getLogger("observability")


def configure_observability(settings: Settings) -> bool:
    """Bind LangSmith environment variables. Returns True when tracing is on."""
    enabled = settings.langsmith_tracing and settings.langsmith_api_key is not None
    flag = "true" if enabled else "false"
    # Current names, plus the legacy LANGCHAIN_* aliases older SDKs still read.
    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag
    if enabled:
        key = settings.langsmith_api_key.get_secret_value()  # type: ignore[union-attr]
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    return enabled


class MetricsCallback(AsyncCallbackHandler):
    """Collects latency and token usage for ONE graph turn (create one per request)."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_ms = 0.0
        self.node_ms: dict[str, float] = {}
        self.tool_ms: dict[str, float] = {}
        self._started = time.perf_counter()
        self._open: dict[UUID, tuple[str, str, float]] = {}

    # -- bookkeeping --------------------------------------------------------------
    def _begin(self, run_id: UUID, kind: str, name: str) -> None:
        self._open[run_id] = (kind, name, time.perf_counter())

    def _end(self, run_id: UUID) -> None:
        entry = self._open.pop(run_id, None)
        if entry is None:
            return
        kind, name, started = entry
        elapsed = (time.perf_counter() - started) * 1000
        if kind == "node":
            self.node_ms[name] = self.node_ms.get(name, 0.0) + elapsed
        elif kind == "tool":
            self.tool_ms[name] = self.tool_ms.get(name, 0.0) + elapsed
            self.tool_calls += 1
        else:
            self.llm_ms += elapsed
            self.llm_calls += 1

    # -- graph nodes ---------------------------------------------------------------
    async def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: UUID,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        # Inner runnables inherit the node metadata; only the node itself has its name.
        if node and kwargs.get("name") == node:
            self._begin(run_id, "node", node)

    async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    # -- tools ---------------------------------------------------------------------
    async def on_tool_start(
        self, serialized: Any, input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._begin(run_id, "tool", name)

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    # -- LLM -----------------------------------------------------------------------
    async def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._begin(run_id, "llm", "llm")

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)
        found = False
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += int(usage.get("input_tokens", 0))
                    self.output_tokens += int(usage.get("output_tokens", 0))
                    found = True
        if not found:  # providers that only fill llm_output
            raw = (response.llm_output or {}).get("token_usage") or {}
            self.input_tokens += int(raw.get("prompt_tokens", 0))
            self.output_tokens += int(raw.get("completion_tokens", 0))

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    # -- output ----------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """JSON-safe snapshot of the turn so far."""
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "tool_calls": self.tool_calls,
            "latency_ms": round((time.perf_counter() - self._started) * 1000, 1),
            "llm_latency_ms": round(self.llm_ms, 1),
            "node_latency_ms": {k: round(v, 1) for k, v in self.node_ms.items()},
            "tool_latency_ms": {k: round(v, 1) for k, v in self.tool_ms.items()},
        }


def log_turn_metrics(thread_id: str, request_id: str, summary: dict[str, Any]) -> None:
    """Emit one structured log line per turn (ship it to your log pipeline)."""
    logger.info(
        "turn_metrics %s",
        json.dumps({"thread_id": thread_id, "request_id": request_id, **summary}),
    )
