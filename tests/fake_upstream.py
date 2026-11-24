#!/usr/bin/env python3
"""Fake model provider used to test the anonymizing proxy.

It speaks just enough of the OpenAI Responses / Chat Completions APIs and the
Anthropic Messages API to satisfy Codex CLI and Claude Code, and it appends every
request it receives to a JSONL log. That log is "what the provider would see".

How it answers (based on the last item of the conversation):
  * a tool result                  -> text "tool said: <result>"
  * user text with "CALL_TOOL: X"  -> one shell tool call running `echo X`
  * anything else                  -> text "echo: <last user text>"
Text and tool arguments are streamed in 3-character pieces, so any value the proxy
has to restore gets split across SSE events.

    python tests/fake_upstream.py --port 18080 --log capture.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Python 3.14+
    from compression import zstd
except ImportError:  # pragma: no cover
    zstd = None

PIECE = 3
SECRET_HEADERS = {"authorization", "x-api-key", "api-key", "proxy-authorization"}
TOOL_PREFERENCE = ("shell_command", "exec_command", "shell", "Bash")
_log_lock = threading.Lock()
LOG_PATH = "capture.jsonl"


def pieces(s: str) -> list[str]:
    return [s[i:i + PIECE] for i in range(0, len(s), PIECE)] or [""]


def redact(name: str, value: str) -> str:
    if name.lower() not in SECRET_HEADERS:
        return value
    scheme, _, secret = value.partition(" ")
    return f"{scheme} <redacted len={len(secret)}>" if secret else f"<redacted len={len(value)}>"


def decode(raw: bytes, encoding: str) -> bytes:
    enc = (encoding or "identity").strip().lower()
    if enc == "identity":
        return raw
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        return zlib.decompress(raw)
    if enc == "zstd" and zstd is not None:
        return zstd.decompress(raw)
    raise ValueError(f"unsupported Content-Encoding {encoding!r}")


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    out = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            for key in ("text", "output", "content"):
                if key in part:
                    out.append(text_of(part[key]))
                    break
    return "\n".join(x for x in out if x)


def pick_tool(names) -> str | None:
    names = set(n for n in names if n)
    return next((t for t in TOOL_PREFERENCE if t in names), None)


def decide(last_kind: str, last_text: str, tool: str | None) -> tuple[str, str]:
    if last_kind == "tool_result":
        return "text", "tool said: " + last_text.strip()[:400]
    if tool and "CALL_TOOL:" in last_text:
        payload = last_text.split("CALL_TOOL:", 1)[1].strip().splitlines()[0].strip()
        return "tool", payload
    return "text", "echo: " + last_text.strip()[-400:]


def tool_arguments(tool: str, payload: str) -> dict:
    cmd = f"echo {payload}"
    if tool == "shell":
        return {"command": ["cmd", "/c", cmd] if os.name == "nt" else ["bash", "-lc", cmd]}
    if tool == "exec_command":
        return {"cmd": cmd}
    if tool == "Bash":
        return {"command": cmd, "description": "Echo a test payload"}
    return {"command": cmd}


# --- OpenAI Responses -------------------------------------------------------------

def responses_turn(body: dict):
    items = body.get("input")
    kind, text = "user", items if isinstance(items, str) else ""
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        typ = it.get("type") or ""
        if typ.endswith("_output"):  # function_call_output, custom_tool_call_output, ...
            kind, text = "tool_result", text_of(it.get("output"))
        elif it.get("role") == "user":
            kind, text = "user", text_of(it.get("content"))
    tools = [t.get("name") for t in body.get("tools") or [] if isinstance(t, dict) and t.get("type") == "function"]
    tool = pick_tool(tools)
    action, value = decide(kind, text, tool)
    return action, value, tool


def responses_events(model: str, action: str, value: str, tool: str | None):
    now = int(time.time())
    base = {"id": f"resp_fake_{now}", "object": "response", "created_at": now, "model": model}
    usage = {"input_tokens": 12, "input_tokens_details": {"cached_tokens": 0},
             "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 16}
    events: list = []
    seq = 0

    def emit(kind: str, **fields):
        nonlocal seq
        events.append((kind, {"type": kind, "sequence_number": seq, **fields}))
        seq += 1

    emit("response.created", response={**base, "status": "in_progress", "output": []})
    if action == "tool":
        args = json.dumps(tool_arguments(tool, value), ensure_ascii=False)
        item = {"id": "fc_fake_1", "type": "function_call", "status": "completed",
                "call_id": f"call_fake_{now}", "name": tool, "arguments": args}
        emit("response.output_item.added", output_index=0, item={**item, "status": "in_progress", "arguments": ""})
        for p in pieces(args):
            emit("response.function_call_arguments.delta", item_id=item["id"], output_index=0, delta=p)
        emit("response.function_call_arguments.done", item_id=item["id"], output_index=0, arguments=args)
    else:
        part = {"type": "output_text", "text": value, "annotations": []}
        item = {"id": "msg_fake_1", "type": "message", "status": "completed", "role": "assistant", "content": [part]}
        emit("response.output_item.added", output_index=0, item={**item, "status": "in_progress", "content": []})
        emit("response.content_part.added", item_id=item["id"], output_index=0, content_index=0,
             part={**part, "text": ""})
        for p in pieces(value):
            emit("response.output_text.delta", item_id=item["id"], output_index=0, content_index=0, delta=p)
        emit("response.output_text.done", item_id=item["id"], output_index=0, content_index=0, text=value)
        emit("response.content_part.done", item_id=item["id"], output_index=0, content_index=0, part=part)
    emit("response.output_item.done", output_index=0, item=item)
    final = {**base, "status": "completed", "output": [item], "usage": usage}
    emit("response.completed", response=final)
    return events, final


# --- OpenAI Chat Completions ------------------------------------------------------

def chat_turn(body: dict):
    kind, text = "user", ""
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            kind, text = "tool_result", text_of(m.get("content"))
        elif m.get("role") == "user":
            kind, text = "user", text_of(m.get("content"))
    tools = [(t.get("function") or {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    tool = pick_tool(tools)
    action, value = decide(kind, text, tool)
    return action, value, tool


def chat_events(model: str, action: str, value: str, tool: str | None):
    now = int(time.time())
    cid = f"chatcmpl-fake-{now}"

    def chunk(delta: dict, finish=None):
        return (None, {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    events = [chunk({"role": "assistant", "content": ""})]
    if action == "tool":
        args = json.dumps(tool_arguments(tool, value), ensure_ascii=False)
        call_id = f"call_fake_{now}"
        events.append(chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                             "function": {"name": tool, "arguments": ""}}]}))
        for p in pieces(args):
            events.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": p}}]}))
        events.append(chunk({}, "tool_calls"))
        message = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool, "arguments": args}}]}
        finish = "tool_calls"
    else:
        for p in pieces(value):
            events.append(chunk({"content": p}))
        events.append(chunk({}, "stop"))
        message = {"role": "assistant", "content": value}
        finish = "stop"
    events.append((None, "[DONE]"))
    final = {"id": cid, "object": "chat.completion", "created": now, "model": model,
             "choices": [{"index": 0, "message": message, "finish_reason": finish}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}}
    return events, final


# --- Anthropic Messages -----------------------------------------------------------

def messages_turn(body: dict):
    kind, text = "user", ""
    for m in body.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            tr_texts = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_result":
                    tr_texts.append(text_of(c.get("content")))
                    kind, text = "tool_result", "\n".join(x for x in tr_texts if x)
                elif c.get("type") == "text" and c.get("text"):
                    tr_texts = []
                    kind, text = "user", text_of(c.get("text"))
        else:
            kind, text = "user", text_of(content)
    tools = [t.get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    with_thinking = bool(body.get("thinking")) and "WITH_THINKING" in text
    if "apply_patch" in tools and "CALL_PATCH:" in text:
        payload = text.split("CALL_PATCH:", 1)[1].strip().splitlines()[0].strip()
        return "patch", payload, "apply_patch", with_thinking
    tool = pick_tool(tools)
    action, value = decide(kind, text, tool)
    return action, value, tool, with_thinking


def messages_events(model: str, action: str, value: str, tool: str | None, with_thinking: bool = False):
    now = int(time.time())
    msg_id = f"msg_fake_{now}"
    usage = {"input_tokens": 12, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    events = [("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": usage}})]
    blocks = []
    idx = 0
    if with_thinking:
        thought = f"plan for {value}"
        sig = "sig_fake_opus_55=="
        tblock = {"type": "thinking", "thinking": thought, "signature": sig}
        blocks.append(tblock)
        events.append(("content_block_start", {"type": "content_block_start", "index": idx,
                                               "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
        for p in pieces(thought):
            events.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                   "delta": {"type": "thinking_delta", "thinking": p}}))
        events.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                               "delta": {"type": "signature_delta", "signature": sig}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
        idx += 1
    if action in ("tool", "patch"):
        args = {"input": f"*** Begin Patch\n*** Update File: {value}\n*** End Patch"} if action == "patch" \
            else tool_arguments(tool, value)
        block = {"type": "tool_use", "id": f"toolu_fake_{now}", "name": tool, "input": args}
        blocks.append(block)
        events.append(("content_block_start", {"type": "content_block_start", "index": idx,
                                               "content_block": {**block, "input": {}}}))
        for p in pieces(json.dumps(args, ensure_ascii=False)):
            events.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                   "delta": {"type": "input_json_delta", "partial_json": p}}))
        stop = "tool_use"
    else:
        block = {"type": "text", "text": value}
        blocks.append(block)
        events.append(("content_block_start", {"type": "content_block_start", "index": idx,
                                               "content_block": {"type": "text", "text": ""}}))
        for p in pieces(value):
            events.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                   "delta": {"type": "text_delta", "text": p}}))
        stop = "end_turn"
    events.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                     "usage": {"output_tokens": 4}}))
    events.append(("message_stop", {"type": "message_stop"}))
    final = {"id": msg_id, "type": "message", "role": "assistant", "model": model, "content": blocks,
             "stop_reason": stop, "stop_sequence": None, "usage": {**usage, "output_tokens": 4}}
    return events, final


# --- HTTP -------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-upstream/1"

    def log_message(self, fmt, *args):  # keep stdout quiet
        pass

    def _read_body(self) -> bytes:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            data = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return bytes(data)
                data += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _record(self, raw: bytes):
        entry = {"ts": round(time.time(), 3), "method": self.command, "path": self.path,
                 "headers": [[k, redact(k, v)] for k, v in self.headers.items()]}
        body = None
        if raw:
            try:
                body = json.loads(decode(raw, self.headers.get("Content-Encoding", "")))
                entry["json"] = body
            except Exception as e:  # keep going, just note it
                entry["body_error"] = repr(e)
                entry["body_len"] = len(raw)
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return body

    def _send_json(self, code: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, events) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for name, data in events:
            payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            frame = (f"event: {name}\n" if name else "") + f"data: {payload}\n\n"
            self.wfile.write(frame.encode())
            self.wfile.flush()
            time.sleep(0.003)

    def do_GET(self):
        self._record(b"")
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/models"):
            return self._send_json(200, {"object": "list", "data": [
                {"id": "fake-model", "object": "model", "created": 0, "owned_by": "fake"}]})
        self._send_json(404, {"error": {"message": f"unknown path {path}", "type": "invalid_request_error"}})

    def do_POST(self):
        body = self._record(self._read_body())
        if not isinstance(body, dict):
            return self._send_json(400, {"error": {"message": "expected a JSON body", "type": "invalid_request_error"}})
        path = self.path.split("?", 1)[0].rstrip("/")
        model = str(body.get("model") or "fake-model")
        if path.endswith("/responses"):
            events, final = responses_events(model, *responses_turn(body))
        elif path.endswith("/chat/completions"):
            events, final = chat_events(model, *chat_turn(body))
        elif path.endswith("/messages/count_tokens"):
            return self._send_json(200, {"input_tokens": 42})
        elif path.endswith("/messages"):
            events, final = messages_events(model, *messages_turn(body))
        else:
            return self._send_json(404, {"error": {"message": f"unknown path {path}", "type": "invalid_request_error"}})
        if body.get("stream"):
            self._send_sse(events)
        else:
            self._send_json(200, final)


def main() -> None:
    global LOG_PATH
    ap = argparse.ArgumentParser(description="Fake OpenAI/Anthropic upstream that logs what it receives.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--log", default="capture.jsonl")
    args = ap.parse_args()
    LOG_PATH = args.log
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake upstream on http://{args.host}:{args.port}  log -> {LOG_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
