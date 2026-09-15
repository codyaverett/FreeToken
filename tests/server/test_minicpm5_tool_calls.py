"""MiniCPM5 tool calls: the detector, and the Metal proxy that applies it.

MiniCPM5 emits ``<function name="f"><param name="p">v</param></function>`` and
mlx_lm does not recognize that grammar, so the calls reach the proxy as assistant
text. These cover the detector (buffered and streamed, including the CDATA the
chat template wraps awkward values in) and the proxy rewriting both response
shapes into OpenAI ``tool_calls``.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_minicpm5_tool_calls.py -v
"""

from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from freetoken.server import metal
from freetoken.server.function_call_parser import FunctionCallParser
from freetoken.server.metal_tool_calls import ToolCallRewriter, proxy_tool_parser, rewriter_for

MODEL = "openbmb/MiniCPM5-2B-MLX"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    },
    {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}},
]

CALL = '<function name="get_weather"><param name="city">Paris</param><param name="days">3</param></function>'


def _parser() -> FunctionCallParser:
    return FunctionCallParser(TOOLS, "minicpm5")


def _stream(text: str, size: int = 5) -> tuple[str, list[tuple[str | None, str]]]:
    """Feed the text through the detector in fixed-size slices, as the wire does."""
    parser = _parser()
    out: list[str] = []
    calls: list[tuple[str | None, str]] = []
    for start in range(0, len(text), size):
        normal, items = parser.parse_stream_chunk(text[start : start + size])
        out.append(normal)
        calls.extend((item.name, item.parameters) for item in items)
    out.append(parser.finish_stream())
    return "".join(out), calls


def _arguments(calls: list[tuple[str | None, str]]) -> str:
    """Concatenate one call's streamed fragments the way a client does."""
    return "".join(params for _, params in calls)


# ------------------------------------------------------------------ detector --


def test_buffered_parse_extracts_call_and_types_arguments():
    result = _parser().parse_non_stream("Let me check.\n" + CALL)
    assert result.normal_text == "Let me check."
    assert [c.name for c in result.calls] == ["get_weather"]
    # days is declared integer in the schema, so it must not come back as "3".
    assert json.loads(result.calls[0].parameters) == {"city": "Paris", "days": 3}


def test_streamed_parse_matches_buffered_parse():
    text, calls = _stream(CALL)
    assert text == ""
    assert [name for name, _ in calls if name] == ["get_weather"]
    assert json.loads(_arguments(calls)) == {"city": "Paris", "days": 3}


def test_two_calls_in_one_turn_stream_separately():
    text, calls = _stream(CALL + '\n<function name="ping"></function>')
    assert text == ""
    assert [name for name, _ in calls if name] == ["get_weather", "ping"]


def test_cdata_wrapper_is_stripped_from_values():
    call = '<function name="get_weather"><param name="city"><![CDATA[a<b\nc]]></param></function>'
    assert json.loads(_parser().parse_non_stream(call).calls[0].parameters) == {"city": "a<b\nc"}
    _, calls = _stream(call)
    assert json.loads(_arguments(calls)) == {"city": "a<b\nc"}


def test_text_after_a_call_is_still_delivered():
    text, calls = _stream('<function name="ping"></function>\nAnything else?')
    assert text.strip() == "Anything else?"
    assert [name for name, _ in calls if name] == ["ping"]


def test_plain_answer_passes_through_untouched():
    answer = "No tool needed, the answer is 4."
    assert _parser().parse_non_stream(answer).calls == []
    text, calls = _stream(answer)
    assert (text, calls) == (answer, [])


# ------------------------------------------------------------------ rewriter --


def test_rewriter_gate_only_fires_for_minicpm5_with_tools():
    assert proxy_tool_parser(MODEL) == "minicpm5"
    assert proxy_tool_parser("mlx-community/Qwen3-8B-4bit") is None
    body = json.dumps({"messages": [], "tools": TOOLS}).encode()
    assert rewriter_for(body, MODEL) is not None
    assert rewriter_for(body, "mlx-community/Qwen3-8B-4bit") is None
    # A request that offers no tools cannot have meant the text as a call.
    assert rewriter_for(json.dumps({"messages": []}).encode(), MODEL) is None


def test_rewriter_moves_buffered_call_into_tool_calls():
    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "One moment.\n" + CALL},
            }
        ]
    }
    assert ToolCallRewriter(TOOLS, "minicpm5").rewrite_payload(payload) is True
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "One moment."
    (call,) = choice["message"]["tool_calls"]
    assert call["type"] == "function"
    assert call["id"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}


def test_rewriter_leaves_an_ordinary_answer_alone():
    payload = {
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"content": "Just 4."}}
        ]
    }
    assert ToolCallRewriter(TOOLS, "minicpm5").rewrite_payload(payload) is False
    assert payload["choices"][0]["finish_reason"] == "stop"


def _sse_chunk(delta: dict, finish: str | None = None) -> bytes:
    return (
        b"data: "
        + json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ).encode()
        + b"\n"
    )


def _collect(rewriter: ToolCallRewriter, lines: list[bytes]) -> list[dict]:
    events: list[dict] = []
    for line in lines:
        for frame in rewriter.rewrite_line(line).split(b"\n"):
            frame = frame.strip()
            if frame.startswith(b"data:") and frame[5:].strip() != b"[DONE]":
                events.append(json.loads(frame[5:]))
    return events


def test_rewriter_streams_a_call_as_tool_call_deltas():
    rewriter = ToolCallRewriter(TOOLS, "minicpm5")
    pieces = ["<fun", 'ction name="get_wea', 'ther"><param name="ci', "ty\">Par", "is</param></fun", "ction>"]
    lines = [_sse_chunk({"role": "assistant"})]
    lines += [_sse_chunk({"content": piece}) for piece in pieces]
    lines.append(_sse_chunk({}, finish="stop"))

    events = _collect(rewriter, lines)
    calls = [c for e in events for c in (e["choices"][0]["delta"].get("tool_calls") or [])]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    # No raw markup leaks into the content channel, and the finish reason flips.
    assert not any(e["choices"][0]["delta"].get("content") for e in events)
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_rewriter_closes_a_call_truncated_by_the_token_limit():
    rewriter = ToolCallRewriter(TOOLS, "minicpm5")
    cut = '<function name="get_weather"><param name="city">Par'
    events = _collect(rewriter, [_sse_chunk({"content": cut}), _sse_chunk({}, finish="length")])
    calls = [c for e in events for c in (e["choices"][0]["delta"].get("tool_calls") or [])]
    # The call is reported with empty arguments rather than left half-open: the
    # client sees a call it can reject, not a stream that never resolves.
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert calls[0]["function"]["arguments"] == "{}"


def test_rewriter_passes_through_text_done_and_usage_chunks():
    rewriter = ToolCallRewriter(TOOLS, "minicpm5")
    usage = b'data: {"choices": [], "usage": {"total_tokens": 7}}\n'
    done = b"data: [DONE]\n"
    keepalive = b": ping\n"
    assert rewriter.rewrite_line(usage) == usage
    assert rewriter.rewrite_line(done) == done
    assert rewriter.rewrite_line(keepalive) == keepalive
    events = _collect(rewriter, [_sse_chunk({"content": "plain text"}), _sse_chunk({}, "stop")])
    assert events[0]["choices"][0]["delta"]["content"] == "plain text"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------- proxy route --


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_upstream(port: int, pieces: list[str]):
    """An mlx_lm-shaped upstream that streams the XML call as assistant text."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        if not body.get("stream"):
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "".join(pieces)},
                    }
                ],
            }

        def events():
            for piece in pieces:
                yield _sse_chunk({"content": piece}) + b"\n"
            yield _sse_chunk({}, finish="stop") + b"\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "upstream test server failed to start"

    def stop():
        server.should_exit = True
        thread.join(timeout=5)

    return stop


def _proxy_client(port: int) -> TestClient:
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url=f"http://127.0.0.1:{port}",
        backend="mlx",
        model_path=MODEL,
    )
    handle.load_state = "ready"
    proxy = FastAPI()
    metal.register_metal_proxy_routes(proxy, lambda: handle)
    return TestClient(proxy, raise_server_exceptions=False)


def test_proxy_converts_upstream_xml_into_tool_calls():
    port = _free_port()
    stop = _start_upstream(port, ['<function name="ping">', "</function>"])
    try:
        client = _proxy_client(port)
        r = client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "tools": TOOLS},
        )
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "ping"

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": TOOLS,
                "stream": True,
            },
        )
        assert r.status_code == 200
        events = [
            json.loads(line[5:])
            for line in r.text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        calls = [c for e in events for c in (e["choices"][0]["delta"].get("tool_calls") or [])]
        assert [c["function"]["name"] for c in calls] == ["ping"]
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        stop()
