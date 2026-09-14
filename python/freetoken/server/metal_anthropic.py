"""Anthropic Messages API (``/v1/messages``) on the Metal proxy.

The Metal upstreams (mlx_lm.server, llama-server) only speak OpenAI chat
completions, so this module translates an Anthropic request into an upstream
chat request and the chat answer (buffered or SSE) back into Anthropic wire
events. It duplicates the prompt-side conversion of ``anthropic_api.py`` on
purpose: that module drives the CUDA generation primitive and imports torch,
which a Metal venv does not have.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .anthropic_models import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)

# OpenAI finish_reason -> Anthropic stop_reason.
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


# ------------------------------------------------------------------ request --
def _content_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and item.get("text"):
                parts.append(item["text"])
        elif getattr(item, "type", None) == "text" and getattr(item, "text", None):
            parts.append(item.text)
    return "".join(parts)


def anthropic_to_chat_request(
    req: AnthropicMessagesRequest | AnthropicCountTokensRequest, upstream_model: str
) -> dict[str, Any]:
    """Build the upstream ``/v1/chat/completions`` body for an Anthropic request.

    ``model`` is always the upstream's own model id: mlx_lm.server loads whatever
    model a request names, so a client's ``claude-*`` id must never pass through.
    Image blocks are dropped (the Metal upstreams are text-only) and every
    message collapses to plain-string content."""
    system_texts: list[str] = []
    if req.system:
        if isinstance(req.system, str):
            system_texts.append(req.system)
        else:
            system_texts.append(
                "".join(b.text for b in req.system if b.type == "text" and b.text)
            )

    other: list[dict[str, Any]] = []
    for msg in req.messages:
        if msg.role == "system":
            system_texts.append(_content_text(msg.content))
            continue
        if isinstance(msg.content, str):
            other.append({"role": msg.role, "content": msg.content})
            continue

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in msg.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "thinking" and block.thinking:
                thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                )
            elif block.type == "tool_result":
                text = _content_text(block.content)
                if msg.role == "user":
                    other.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id or block.id or "",
                            "content": text,
                        }
                    )
                else:
                    text_parts.append(f"Tool result: {text}")
            # image / redacted_thinking / unknown blocks are skipped.

        openai_msg: dict[str, Any] = {"role": msg.role}
        if thinking_parts:
            openai_msg["reasoning_content"] = "\n\n".join(thinking_parts)
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls
        if text_parts:
            openai_msg["content"] = "".join(text_parts)
        elif not tool_calls and not thinking_parts:
            continue
        other.append(openai_msg)

    messages: list[dict[str, Any]] = []
    system_text = "\n\n".join(t for t in system_texts if t)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(other)

    body: dict[str, Any] = {"model": upstream_model, "messages": messages}

    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in (req.tools or [])
    ]
    # "none" hides the tools; the upstreams have no forced-tool mode, so
    # "any"/"tool" degrade to "auto" rather than being emulated.
    if tools and not (req.tool_choice and req.tool_choice.type == "none"):
        body["tools"] = tools

    if req.thinking and req.thinking.get("type") in {"enabled", "disabled"}:
        body["chat_template_kwargs"] = {
            "enable_thinking": req.thinking["type"] == "enabled"
        }

    if isinstance(req, AnthropicMessagesRequest):
        body["max_tokens"] = req.max_tokens
        for key in ("temperature", "top_p", "top_k"):
            value = getattr(req, key)
            if value is not None:
                body[key] = value
        if req.stop_sequences:
            body["stop"] = list(req.stop_sequences)
        if req.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
    return body


# ----------------------------------------------------------------- response --
def _tool_use_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"toolu_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _parse_json_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments:
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _usage(payload: dict[str, Any] | None) -> AnthropicUsage:
    usage = payload or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return AnthropicUsage(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cache_read_input_tokens=cached or None,
    )


def _stop_reason(finish_reason: str | None, made_tool_call: bool) -> str | None:
    if made_tool_call:
        return "tool_use"
    if finish_reason is None:
        return None
    return STOP_REASON_MAP.get(finish_reason, "end_turn")


def _message_field(message: dict[str, Any], key: str) -> Any:
    # mlx_lm names the thinking channel "reasoning"; the proxy's chat route
    # renames it to FreeToken's "reasoning_content", so accept both here.
    if key == "reasoning":
        return message.get("reasoning") or message.get("reasoning_content")
    return message.get(key)


def chat_response_to_anthropic(payload: dict[str, Any], model: str) -> AnthropicMessagesResponse:
    """Map a buffered upstream chat completion to an Anthropic message."""
    choices = payload.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}

    content: list[AnthropicContentBlock] = []
    reasoning = _message_field(message, "reasoning")
    if reasoning:
        content.append(AnthropicContentBlock(type="thinking", thinking=reasoning, signature=""))
    content.append(AnthropicContentBlock(type="text", text=message.get("content") or ""))
    tool_calls = message.get("tool_calls") or []
    for index, call in enumerate(tool_calls):
        function = call.get("function") or {}
        content.append(
            AnthropicContentBlock(
                type="tool_use",
                id=call.get("id") or _tool_use_id(function.get("name"), index),
                name=function.get("name"),
                input=_parse_json_args(function.get("arguments")),
            )
        )
    return AnthropicMessagesResponse(
        id=f"msg_{payload.get('id') or uuid.uuid4().hex}",
        content=content,
        model=model,
        stop_reason=_stop_reason(choice.get("finish_reason"), bool(tool_calls)),
        usage=_usage(payload.get("usage")),
    )


def _event(event: AnthropicStreamEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def error_event(kind: str, message: str) -> str:
    return _event(
        AnthropicStreamEvent(type="error", error=AnthropicError(type=kind, message=message))
    )


def _sse_data(line: bytes) -> dict[str, Any] | None:
    """Decode one upstream SSE line into its JSON payload, or None for
    non-data lines and the ``[DONE]`` sentinel."""
    if not line.startswith(b"data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def chat_stream_to_anthropic(
    lines: AsyncIterator[bytes], model: str
) -> AsyncIterator[str]:
    """Translate an upstream chat-completion SSE stream into Anthropic events.

    mlx_lm streams each tool call complete (one chunk carries the full
    ``arguments`` string), so a tool_use block opens, takes one
    ``input_json_delta`` and closes in the same step. Usage rides on the
    trailing chunk that ``stream_options.include_usage`` requests; when the
    upstream sends none, the terminal ``message_delta`` reports zeros."""
    block_index = 0
    block_open: str | None = None  # "text" | "thinking"
    made_tool_call = False
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    message_id = f"msg_{uuid.uuid4().hex}"
    finished = False

    def _open(kind: str) -> str:
        nonlocal block_open
        block_open = kind
        block = (
            AnthropicContentBlock(type="text", text="")
            if kind == "text"
            else AnthropicContentBlock(type="thinking", thinking="")
        )
        return _event(
            AnthropicStreamEvent(type="content_block_start", index=block_index, content_block=block)
        )

    def _close() -> list[str]:
        nonlocal block_open, block_index
        frames: list[str] = []
        if block_open == "thinking":
            # Real thinking blocks end with a signature; emit an empty one for
            # shape compliance (nothing verifies replayed signatures here).
            frames.append(
                _event(
                    AnthropicStreamEvent(
                        type="content_block_delta",
                        index=block_index,
                        delta=AnthropicDelta(type="signature_delta", signature=""),
                    )
                )
            )
        frames.append(_event(AnthropicStreamEvent(type="content_block_stop", index=block_index)))
        block_open = None
        block_index += 1
        return frames

    def _delta(kind: str, text: str) -> list[str]:
        frames: list[str] = []
        if block_open != kind:
            if block_open:
                frames.extend(_close())
            frames.append(_open(kind))
        delta = (
            AnthropicDelta(type="text_delta", text=text)
            if kind == "text"
            else AnthropicDelta(type="thinking_delta", thinking=text)
        )
        frames.append(
            _event(AnthropicStreamEvent(type="content_block_delta", index=block_index, delta=delta))
        )
        return frames

    def _tool(call: dict[str, Any]) -> list[str]:
        nonlocal block_open, block_index, made_tool_call
        frames: list[str] = []
        if block_open:
            frames.extend(_close())
        function = call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments") or ""
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        frames.append(
            _event(
                AnthropicStreamEvent(
                    type="content_block_start",
                    index=block_index,
                    content_block=AnthropicContentBlock(
                        type="tool_use",
                        id=call.get("id") or _tool_use_id(name, block_index),
                        name=name,
                        input={},
                    ),
                )
            )
        )
        frames.append(
            _event(
                AnthropicStreamEvent(
                    type="content_block_delta",
                    index=block_index,
                    delta=AnthropicDelta(type="input_json_delta", partial_json=arguments),
                )
            )
        )
        frames.append(_event(AnthropicStreamEvent(type="content_block_stop", index=block_index)))
        block_index += 1
        made_tool_call = True
        return frames

    def _finish() -> list[str]:
        nonlocal finished
        finished = True
        frames = _close() if block_open else []
        frames.append(
            _event(
                AnthropicStreamEvent(
                    type="message_delta",
                    delta=AnthropicDelta(stop_reason=_stop_reason(finish_reason, made_tool_call)),
                    usage=_usage(usage),
                )
            )
        )
        frames.append(_event(AnthropicStreamEvent(type="message_stop")))
        return frames

    yield _event(
        AnthropicStreamEvent(
            type="message_start",
            message=AnthropicMessagesResponse(
                id=message_id,
                content=[],
                model=model,
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            ),
        )
    )
    try:
        async for line in lines:
            line = line.rstrip(b"\r\n")
            if line.startswith(b":"):
                # Upstream keepalive comment -> protocol-native ping.
                yield _event(AnthropicStreamEvent(type="ping"))
                continue
            payload = _sse_data(line)
            if payload is None:
                continue
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            reasoning = _message_field(delta, "reasoning")
            if reasoning:
                for frame in _delta("thinking", reasoning):
                    yield frame
            text = delta.get("content")
            if text:
                for frame in _delta("text", text):
                    yield frame
            for call in delta.get("tool_calls") or []:
                for frame in _tool(call):
                    yield frame
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        for frame in _finish():
            yield frame
    except Exception as exc:  # noqa: BLE001 - surface as an Anthropic error event
        if not finished:
            if block_open:
                for frame in _close():
                    yield frame
            yield error_event("internal_error", str(exc))


# ------------------------------------------------------------- count_tokens --
def count_tokens(req: AnthropicCountTokensRequest, tokenizer: Any | None) -> int:
    """Count the rendered prompt with the upstream model's tokenizer when one is
    available, else estimate at four characters per token."""
    body = anthropic_to_chat_request(req, upstream_model="")
    if tokenizer is not None:
        try:
            ids = tokenizer.apply_chat_template(
                body["messages"],
                tools=body.get("tools"),
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 - a template that rejects the prompt falls back to the estimate
            ids = None
        if ids is not None:
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            return len(ids)
    text = json.dumps(body["messages"]) + json.dumps(body.get("tools") or [])
    return max(1, len(text) // 4)
