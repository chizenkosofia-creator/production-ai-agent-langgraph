"""Pydantic v2 request/response models for the HTTP API."""

import hashlib
import logging
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.guardrails import normalize_text, scan_prompt

logger = logging.getLogger("guardrails")

# Restrict thread IDs to a safe charset: they become DB keys and log fields.
THREAD_ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,128}$"

REJECTION_MESSAGE = "message rejected by input policy"


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat`` and ``POST /api/v1/chat/stream``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str = Field(
        pattern=THREAD_ID_PATTERN,
        description="Conversation identifier. Reuse it to continue a conversation.",
        examples=["customer-42-session-1"],
    )
    message: str = Field(min_length=1, max_length=8000, description="The user's message.")

    @field_validator("message", mode="before")
    @classmethod
    def _normalise(cls, value: object) -> object:
        """Fold look-alike unicode, drop invisible/control characters."""
        return normalize_text(value)

    @field_validator("message")
    @classmethod
    def _screen_for_injection(cls, value: str) -> str:
        """Reject known prompt-injection patterns (see core/guardrails.py)."""
        rules = scan_prompt(value)
        if rules:
            # Log the rule names and a digest, never the text itself.
            digest = hashlib.sha256(value.encode()).hexdigest()[:12]
            logger.warning("Prompt rejected rules=%s sha256=%s", rules, digest)
            raise ValueError(REJECTION_MESSAGE)
        return value


class UsageSummary(BaseModel):
    """Per-turn token usage and latency profile (see core/observability.py)."""

    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: int
    latency_ms: float
    llm_latency_ms: float
    node_latency_ms: dict[str, float]
    tool_latency_ms: dict[str, float]


class ChatResponse(BaseModel):
    """Response of ``POST /api/v1/chat``."""

    thread_id: str
    reply: str
    tool_calls_used: int = Field(ge=0, description="Tool executions during this turn.")
    total_messages: int = Field(ge=0, description="Messages stored in the thread.")
    usage: Optional[UsageSummary] = None


class HistoryMessage(BaseModel):
    """One persisted message in a thread."""

    role: Literal["human", "ai", "tool", "system"]
    content: str
    tool_name: Optional[str] = None


class HistoryResponse(BaseModel):
    """Response of ``GET /api/v1/threads/{thread_id}/history``."""

    thread_id: str
    messages: list[HistoryMessage]


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error: str
    detail: Optional[str] = None
