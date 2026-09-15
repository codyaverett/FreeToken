"""Tool-call parsing for Metal upstreams that cannot do it themselves.

mlx_lm picks a tool-call parser by pattern-matching the model's chat template
and has no branch for every family it serves: a model whose template it does not
recognize reports ``has_tool_calling = False``, templates the request's ``tools``
into the prompt anyway, and hands the resulting call back as ordinary assistant
text. The client then sees prose where it expected ``tool_calls`` and the agent
loop stalls (MiniCPM5 and its bare ``<function name="...">`` grammar is the case
this was written for).

The proxy already stands between the client and mlx_lm, so it parses those calls
out of the text itself with FreeToken's own detectors and re-emits them in the
OpenAI wire shape -- streaming and buffered, which also covers the Anthropic
route since that one converts from an OpenAI chat response.

The OpenAI serializers here are deliberately local rather than imported from
``openai_api``: that module reaches the CUDA generation stack and pulls in torch,
which a Metal host does not have.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from .function_call_parser import FunctionCallParser, ToolCallItem

# Families the Metal upstream cannot parse, matched against the model path.
_PROXY_TOOL_PARSERS: tuple[tuple[str, str], ...] = (("minicpm5", "minicpm5"),)


def proxy_tool_parser(model_path: str | None) -> str | None:
    """The parser name this model's calls need from the proxy, or None when the
    upstream already returns structured ``tool_calls``."""
    marker = (model_path or "").lower()
    return next((parser for token, parser in _PROXY_TOOL_PARSERS if token in marker), None)


def rewriter_for(body: bytes, model_path: str | None) -> "ToolCallRewriter | None":
    """A rewriter for this request, or None when it needs no parsing: no proxy
    parser for the model, or a request that offered no tools."""
    parser_name = proxy_tool_parser(model_path)
    if parser_name is None:
        return None
    try:
        payload = json.loads(body or b"{}")
    except (ValueError, UnicodeDecodeError):
        return None
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not tools:
        return None
    return ToolCallRewriter(tools, parser_name)


def _tool_call_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"call_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


class ToolCallRewriter:
    """Per-request parse state. One rewriter serves one generation: it holds the
    detector's streaming buffer, so it must not be shared across requests."""

    def __init__(self, tools: list[Any], parser_name: str) -> None:
        self.parser = FunctionCallParser(tools, parser_name)
        self._open: dict[str, str] | None = None
        self._emitted = 0

    # -- buffered ---------------------------------------------------------

    def rewrite_payload(self, payload: Any) -> bool:
        """Move tool calls out of ``message.content`` and into ``message.tool_calls``.
        Returns whether anything changed; a payload with no call is left alone."""
        if not isinstance(payload, dict):
            return False
        changed = False
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("tool_calls"):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not self.parser.has_tool_call(content):
                continue
            result = self.parser.parse_non_stream(content)
            if not result.calls:
                continue
            message["content"] = result.normal_text or None
            message["tool_calls"] = [
                {
                    "id": _tool_call_id(call.name, index),
                    "index": index,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.parameters},
                }
                for index, call in enumerate(result.calls)
            ]
            choice["finish_reason"] = "tool_calls"
            changed = True
        return changed

    # -- streaming --------------------------------------------------------

    def rewrite_line(self, line: bytes) -> bytes:
        """Rewrite one SSE line. Returns the replacement bytes: the line itself when
        it carries no assistant content, one or more re-framed chunks when calls or
        text come out of the parser, and empty bytes while a call is still arriving
        (the stray event separator that leaves behind is inert in SSE)."""
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line
        data = stripped[5:].strip()
        if data == b"[DONE]":
            return line
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return line
        if not isinstance(payload, dict):
            return line
        choices = payload.get("choices") or []
        # Usage-only final chunk (stream_options.include_usage) carries no choices.
        if not choices or not isinstance(choices[0], dict):
            return line
        choice = choices[0]
        delta = choice.get("delta")
        content = delta.get("content") if isinstance(delta, dict) else None
        finish = choice.get("finish_reason")
        if not content and finish is None:
            return line

        frames: list[bytes] = []
        if content:
            text, items = self.parser.parse_stream_chunk(content)
            for item in items:
                self._consume(item, payload, frames)
            if text:
                frames.append(self._chunk(payload, {"content": text}))

        if finish is not None:
            if finish == "length":
                for item in self.parser.recover_truncated_call():
                    self._consume(item, payload, frames)
            residual = self.parser.finish_stream()
            if residual:
                frames.append(self._chunk(payload, {"content": residual}))
            frames.extend(self._close_open(payload))
            if self._emitted:
                choice["finish_reason"] = "tool_calls"
            if isinstance(delta, dict):
                delta.pop("content", None)
            frames.append(_sse(payload))

        return b"".join(frames)

    def _consume(self, item: ToolCallItem, payload: dict, frames: list[bytes]) -> None:
        """Fold one detector item into the open call. The detectors stream a call as
        a name-bearing item followed by argument fragments; the proxy holds those
        until the call closes and sends it as one complete delta instead."""
        if item.name:
            frames.extend(self._close_open(payload))
            self._open = {"name": item.name, "arguments": item.parameters or ""}
        elif self._open is not None:
            self._open["arguments"] += item.parameters or ""

    def _close_open(self, payload: dict) -> list[bytes]:
        if self._open is None:
            return []
        call, self._open = self._open, None
        index = self._emitted
        self._emitted += 1
        return [
            self._chunk(
                payload,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": _tool_call_id(call["name"], index),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"] or "{}",
                            },
                        }
                    ]
                },
            )
        ]

    def _chunk(self, template: dict, delta: dict[str, Any]) -> bytes:
        """An SSE chunk carrying ``delta``, borrowing the upstream chunk's identity
        so the client sees one coherent stream."""
        return _sse(
            {
                "id": template.get("id"),
                "object": template.get("object", "chat.completion.chunk"),
                "created": template.get("created"),
                "model": template.get("model"),
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
