"""Pydantic v2 request/response models for the HTTP API."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

THREAD_ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,128}$"


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str = Field(
        pattern=THREAD_ID_PATTERN,
        description="Conversation identifier. Reuse it to continue a conversation.",
        examples=["customer-42-session-1"],
    )
    message: str = Field(min_length=1, max_length=8000, description="The user's message.")


class ChatResponse(BaseModel):
    """Response of ``POST /api/v1/chat``."""

    thread_id: str
    reply: str
    tool_calls_used: int = Field(ge=0, description="Tool executions during this turn.")
    total_messages: int = Field(ge=0, description="Messages stored in the thread.")


class HistoryMessage(BaseModel):
    """One persisted message in a thread."""

    role: Literal["human", "ai", "tool", "system"]
    content: str
    tool_name: Optional[str] = None


class HistoryResponse(BaseModel):
    """Response of GET /api/v1/threads/{thread_id}/history."""

    thread_id: str
    messages: list[HistoryMessage]


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error: str
    detail: Optional[str] = None