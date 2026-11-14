"""A fake Groq (OpenAI-compatible) HTTP endpoint for offline tests.

Plugs into the *real* ``ChatOpenAI`` client through an ``httpx.MockTransport``,
so tests exercise the genuine request building, auth header, retries and SSE
parsing without any network access or rate-limit consumption.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class Reply:
    """One scripted completion (or error) returned by the fake endpoint."""

    content: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    usage: tuple[int, int] = (0, 0)  # (prompt_tokens, completion_tokens)
    status: int = 200  # non-200 returns an error body instead


class GroqMock:
    """Records every request and answers from a script (last reply repeats)."""

    def __init__(self, replies: list[Reply], model: str = "llama-3.1-8b-instant") -> None:
        self.replies = replies
        self.model = model
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    # -- plumbing -------------------------------------------------------------
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(request)
        self.bodies.append(body)
        reply = self.replies[min(len(self.bodies) - 1, len(self.replies) - 1)]

        if reply.status != 200:
            return httpx.Response(
                reply.status,
                json={"error": {"message": f"simulated {reply.status}", "type": "mock_error"}},
                # the OpenAI SDK honours this header, which keeps retry tests fast
                headers={"retry-after-ms": "1"},
            )
        if body.get("stream"):
            return httpx.Response(
                200,
                content=self._sse(reply).encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=self._completion(reply))

    # -- payloads -------------------------------------------------------------
    @staticmethod
    def _tool_call_json(index: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": index,
            "id": f"call_{index + 1}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    def _completion(self, reply: Reply) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.content or None}
        if reply.tool_calls:
            message["tool_calls"] = [
                {k: v for k, v in self._tool_call_json(i, n, a).items() if k != "index"}
                for i, (n, a) in enumerate(reply.tool_calls)
            ]
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if reply.tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": reply.usage[0],
                "completion_tokens": reply.usage[1],
                "total_tokens": sum(reply.usage),
            },
        }

    def _chunk(self, delta: dict[str, Any], finish: Optional[str] = None) -> str:
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def _sse(self, reply: Reply) -> str:
        out = [self._chunk({"role": "assistant", "content": ""})]
        words = reply.content.split(" ") if reply.content else []
        for i, word in enumerate(words):
            out.append(self._chunk({"content": word + (" " if i < len(words) - 1 else "")}))
        for i, (name, args) in enumerate(reply.tool_calls):
            call = self._tool_call_json(i, name, args)
            arguments = call["function"].pop("arguments")
            call["function"]["arguments"] = ""
            out.append(self._chunk({"tool_calls": [call]}))
            out.append(
                self._chunk({"tool_calls": [{"index": i, "function": {"arguments": arguments}}]})
            )
        out.append(self._chunk({}, "tool_calls" if reply.tool_calls else "stop"))
        usage = {
            "prompt_tokens": reply.usage[0],
            "completion_tokens": reply.usage[1],
            "total_tokens": sum(reply.usage),
        }
        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": self.model,
            "choices": [],
            "usage": usage,
        }
        out.append(f"data: {json.dumps(final)}\n\n")
        out.append("data: [DONE]\n\n")
        return "".join(out)
