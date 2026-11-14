"""Offline test doubles: scripted chat model, hash embeddings, stub vector store."""

import hashlib
import json
import math
import re
from collections.abc import Iterator
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AI messages in order; records every prompt it sees.

    Replaces the Groq API in tests. It supports ``bind_tools`` (a no-op) and
    token streaming, so the real graph + SSE code paths are exercised. Once the
    script is exhausted the last message is repeated, which lets a test model a
    runaway agent that never stops calling tools.
    """

    script: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        message = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        # Fresh id per call: real model responses are unique, and add_messages
        # de-duplicates by id (a repeated script entry would overwrite itself).
        return message.model_copy(update={"id": None})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        words = re.findall(r"\S+\s*", str(message.content))
        for word in words:
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        tool_chunks = [
            {
                "name": call["name"],
                "args": json.dumps(call["args"]),
                "id": call["id"],
                "index": i,
            }
            for i, call in enumerate(message.tool_calls)
        ]
        if tool_chunks or not words or message.usage_metadata:
            # final chunk carries tool calls and (like real providers) token usage
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=tool_chunks,
                    usage_metadata=message.usage_metadata,
                )
            )


def tool_call_message(
    name: str, args: dict[str, Any], call_id: str = "call_1", tokens: tuple[int, int] = (0, 0)
) -> AIMessage:
    """An AI message that requests one tool call (optionally reporting token usage)."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata=with_usage(*tokens),
    )


def with_usage(input_tokens: int, output_tokens: int) -> Optional[dict[str, int]]:
    """``usage_metadata`` for a scripted response (None when both are zero)."""
    if not (input_tokens or output_tokens):
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (no model download, no network).

    Texts sharing words get a high cosine similarity, which is enough to test
    ranking and the pgvector plumbing; it is NOT a semantic model.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class StubVectorStore:
    """Minimal stand-in for ``PGVector`` used by unit tests."""

    def __init__(
        self,
        hits: Optional[list[tuple[Document, float]]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.hits = hits or []
        self.error = error
        self.queries: list[tuple[str, int]] = []

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        self.queries.append((query, k))
        if self.error:
            raise self.error
        return self.hits[:k]


def doc(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})
