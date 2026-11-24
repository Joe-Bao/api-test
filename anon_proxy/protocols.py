"""Per-protocol request scrubbing and response restoration.

Request side: drop or pseudonymize identifiers the clients attach (Codex: client_metadata
with installation_id, session/turn ids, prompt_cache_key, client-generated item ids;
Claude Code: metadata.user_id with device_id, the billing header block in `system`),
then mask every free-text string.

Response side: put the originals back, including inside streamed deltas.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass

from .masking import Masker, RestoreDict, StreamRestorer, Vault, restore_tree

CLIENT_UUID_ID = re.compile(
    r"^(?P<p>[A-Za-z]+_)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# string values under these keys are never masked (ids, enums, schemas, opaque blobs)
MASK_SKIP_KEYS = frozenset({
    "encrypted_content", "signature", "data", "id", "call_id", "tool_call_id", "tool_use_id", "model",
    "type", "role", "status", "name", "cache_control", "reasoning", "include", "tool_choice",
    "service_tier", "truncation", "parameters", "input_schema", "schema", "format", "prompt_cache_key",
    "safety_identifier", "previous_response_id", "conversation", "media_type", "detail", "mcp_servers",
    "container",
})
SIGNED_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
BILLING_PREFIX = "x-anthropic-billing-header"


THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"


@dataclass
class Privacy:
    end_user_id: str = "none"          # "none" | "pseudonym"
    force_store_false: bool = True
    strip_billing_header: bool = True
    # Opus 5.5 / Fable 5.x reject replayed thinking blocks whose earlier history changed. Masking is
    # deterministic, but a gateway restart or a config change can still alter old turns; "drop_block"
    # makes the API drop those blocks instead of failing the request. "off" leaves the request alone.
    thinking_mismatch: str = "drop_block"   # "drop_block" | "error" | "off"


class RequestContext:
    def __init__(self, user: str, vault: Vault, masker: Masker, privacy: Privacy, secret: bytes):
        self.user, self.vault, self.masker, self.privacy = user, vault, masker, privacy
        self._secret = secret
        self.used: set[str] = set()
        self.stats: Counter = Counter()
        self.notes: Counter = Counter()
        self.extra_betas: list[str] = []

    # -- pseudonyms for identifiers (not reversible, not restored)
    def pseudo(self, purpose: str, value: str, n: int = 32) -> str:
        msg = f"{purpose}\x00{self.user}\x00{value}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()[:n]

    def pseudo_uuid(self, value: str) -> str:
        digest = hmac.new(self._secret, f"id\x00{self.user}\x00{value}".encode(), hashlib.sha256).digest()
        return str(uuid.UUID(bytes=digest[:16], version=4))

    # -- masking
    def text(self, s: str) -> str:
        return self.masker.mask(s, self.vault, self.used, self.stats)

    def json_text(self, s: str) -> str:
        try:
            parsed = json.loads(s)
        except ValueError:
            return self.text(s)
        return json.dumps(mask_tree(parsed, self), ensure_ascii=False)

    def learn_tree(self, obj) -> None:
        for s in iter_strings(obj):
            self.masker.learn(s, self.vault)

    def restore_dicts(self) -> tuple[RestoreDict, RestoreDict]:
        m = self.vault.mapping(self.used)
        return RestoreDict(m), RestoreDict(m, json_escape=True)


def iter_strings(obj):
    if isinstance(obj, dict):
        if obj.get("type") in SIGNED_BLOCK_TYPES:
            return
        for k, v in obj.items():
            if k in MASK_SKIP_KEYS:
                continue
            if isinstance(v, str):
                if not v.startswith("data:"):
                    yield v
            elif isinstance(v, (dict, list)):
                yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str):
                yield v
            elif isinstance(v, (dict, list)):
                yield from iter_strings(v)


def mask_tree(obj, ctx: RequestContext):
    if isinstance(obj, dict):
        if obj.get("type") in SIGNED_BLOCK_TYPES:   # signed by the provider: must come back byte-identical
            return obj
        for k, v in obj.items():
            if k in MASK_SKIP_KEYS:
                continue
            if isinstance(v, str):
                if v.startswith("data:"):
                    continue
                obj[k] = ctx.json_text(v) if k == "arguments" else ctx.text(v)
            elif isinstance(v, (dict, list)):
                mask_tree(v, ctx)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = ctx.text(v)
            elif isinstance(v, (dict, list)):
                mask_tree(v, ctx)
    return obj


# ------------------------------------------------------------------ request scrubbing

def scrub_openai(endpoint: str, body: dict, ctx: RequestContext) -> dict:
    """endpoint is the path after /v1, e.g. "/responses" or "/chat/completions"."""
    for key in ("client_metadata", "metadata", "user"):
        body.pop(key, None)
    if ctx.privacy.end_user_id == "pseudonym":
        body["safety_identifier"] = ctx.pseudo("end_user", "")
    else:
        body.pop("safety_identifier", None)
    if isinstance(body.get("prompt_cache_key"), str):
        body["prompt_cache_key"] = ctx.pseudo("prompt_cache_key", body["prompt_cache_key"])
    if ctx.privacy.force_store_false and (endpoint == "/responses" or "store" in body):
        body["store"] = False

    for item in body.get("input") if isinstance(body.get("input"), list) else []:
        if not isinstance(item, dict):
            continue
        m = CLIENT_UUID_ID.match(item.get("id") or "") if isinstance(item.get("id"), str) else None
        if m:   # ids the client minted itself (UUIDv7 carries a timestamp); server ids stay as they are
            item["id"] = (m.group("p") or "") + ctx.pseudo_uuid(item["id"])
        if "role" in item:
            item.pop("name", None)
    for msg in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if isinstance(msg, dict) and msg.get("role") in ("user", "system", "developer"):
            msg.pop("name", None)

    ctx.learn_tree(body)
    return mask_tree(body, ctx)


def scrub_anthropic(endpoint: str, body: dict, ctx: RequestContext) -> dict:
    if ctx.privacy.end_user_id == "pseudonym":
        body["metadata"] = {"user_id": ctx.pseudo("end_user", "")}
    else:
        body.pop("metadata", None)

    system = body.get("system")
    if ctx.privacy.strip_billing_header:
        if isinstance(system, list):
            body["system"] = [b for b in system if not (
                isinstance(b, dict) and isinstance(b.get("text"), str)
                and b["text"].lstrip().startswith(BILLING_PREFIX))]
        elif isinstance(system, str):
            body["system"] = "\n".join(
                line for line in system.splitlines() if not line.lstrip().startswith(BILLING_PREFIX))

    thinking = body.get("thinking")
    if ctx.privacy.thinking_mismatch != "off" and endpoint == "/messages" and isinstance(thinking, dict) \
            and thinking.get("type") == "adaptive" and "block_binding" not in thinking:
        # block_binding is accepted only with adaptive thinking and only with the beta header
        thinking["block_binding"] = {"prefix_mismatch_behavior": ctx.privacy.thinking_mismatch}
        ctx.extra_betas.append(THINKING_BINDING_BETA)

    ctx.learn_tree(body)
    return mask_tree(body, ctx)


def note_thinking_drops(message: dict, notes: Counter) -> None:
    """Count thinking blocks the API dropped because the replayed history no longer matched."""
    for t in message.get("input_transformations") or []:
        if isinstance(t, dict) and t.get("type") == "thinking_dropped":
            notes["thinking_dropped"] += 1
            notes[f"thinking_dropped:{t.get('reason')}"] += 1


# ------------------------------------------------------------------ response restoration

class _Base:
    def __init__(self, plain: RestoreDict, js: RestoreDict):
        self.plain, self.js = plain, js

    def finish(self) -> list:
        return []


class ResponsesStream(_Base):
    """OpenAI Responses API SSE (what Codex speaks)."""

    DELTAS = {  # event type -> delta lives inside a JSON string?
        "response.output_text.delta": False,
        "response.refusal.delta": False,
        "response.reasoning_summary_text.delta": False,
        "response.reasoning_text.delta": False,
        "response.custom_tool_call_input.delta": False,
        "response.code_interpreter_call_code.delta": False,
        "response.function_call_arguments.delta": True,
        "response.mcp_call_arguments.delta": True,
    }

    def __init__(self, plain, js):
        super().__init__(plain, js)
        self.streams: dict[tuple, list] = {}   # key -> [StreamRestorer, event name, last delta event]

    def event(self, name, data):
        if not isinstance(data, dict):
            return [(name, data)]
        typ = data.get("type")
        if typ in self.DELTAS and isinstance(data.get("delta"), str):
            key = (typ, data.get("item_id"), data.get("output_index"), data.get("content_index"),
                   data.get("summary_index"))
            st = self.streams.get(key)
            if st is None:
                st = self.streams[key] = [(self.js if self.DELTAS[typ] else self.plain).stream(), name, data]
            st[1], st[2] = name, data
            data["delta"] = st[0].feed(data["delta"])
            return [(name, data)] if data["delta"] else []
        out = self._flush()
        restore_tree(data, self.plain, self.js)
        out.append((name, data))
        return out

    def _flush(self):
        out = []
        for sr, name, template in self.streams.values():
            rest = sr.flush()
            if rest:
                out.append((name, {**template, "delta": rest}))
        return out

    def finish(self):
        return self._flush()


class ChatStream(_Base):
    """OpenAI Chat Completions SSE."""

    TEXT_FIELDS = ("content", "reasoning_content", "reasoning", "refusal")

    def __init__(self, plain, js):
        super().__init__(plain, js)
        self.streams: dict[tuple, StreamRestorer] = {}

    def _sr(self, key, json_ctx=False) -> StreamRestorer:
        sr = self.streams.get(key)
        if sr is None:
            sr = self.streams[key] = (self.js if json_ctx else self.plain).stream()
        return sr

    def event(self, name, data):
        if not isinstance(data, dict):
            if data == "[DONE]":
                return self.finish() + [(name, data)]
            return [(name, data)]
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            idx = choice.get("index", 0)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                for f in self.TEXT_FIELDS:
                    if isinstance(delta.get(f), str):
                        delta[f] = self._sr((idx, f)).feed(delta[f])
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        fn["arguments"] = self._sr((idx, "tool", tc.get("index", 0)), True).feed(fn["arguments"])
            if choice.get("finish_reason"):
                self._flush_choice(idx, choice)
            if isinstance(choice.get("message"), dict):
                restore_tree(choice["message"], self.plain, self.js)
        return [(name, data)]

    def _flush_choice(self, idx, choice):
        delta = choice.setdefault("delta", {})
        for key, sr in self.streams.items():
            if key[0] != idx:
                continue
            rest = sr.flush()
            if not rest:
                continue
            if key[1] == "tool":
                calls = delta.setdefault("tool_calls", [])
                target = next((c for c in calls if c.get("index", 0) == key[2]), None)
                if target is None:
                    calls.append({"index": key[2], "function": {"arguments": rest}})
                else:
                    fn = target.setdefault("function", {})
                    fn["arguments"] = (fn.get("arguments") or "") + rest
            else:
                delta[key[1]] = (delta.get(key[1]) or "") + rest

    def finish(self):
        out = []
        for key, sr in self.streams.items():
            rest = sr.flush()
            if not rest:
                continue
            delta = {"tool_calls": [{"index": key[2], "function": {"arguments": rest}}]} if key[1] == "tool" \
                else {key[1]: rest}
            out.append((None, {"object": "chat.completion.chunk",
                               "choices": [{"index": key[0], "delta": delta, "finish_reason": None}]}))
        return out


class AnthropicStream(_Base):
    """Anthropic Messages SSE (what Claude Code speaks). Thinking text is left alone because
    thinking blocks are signed and must be sent back unchanged."""

    def __init__(self, plain, js, notes: Counter | None = None):
        super().__init__(plain, js)
        self.streams: dict[int, list] = {}   # block index -> [StreamRestorer, delta type, event name, template]
        self.notes = notes if notes is not None else Counter()

    def event(self, name, data):
        if not isinstance(data, dict):
            return [(name, data)]
        typ = data.get("type")
        if typ == "message_start" and isinstance(data.get("message"), dict):
            note_thinking_drops(data["message"], self.notes)
        if typ == "content_block_delta":
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            field = {"text_delta": "text", "input_json_delta": "partial_json"}.get(dtype)
            if field and isinstance(delta.get(field), str):
                idx = data.get("index")
                st = self.streams.get(idx)
                if st is None:
                    st = self.streams[idx] = [(self.js if field == "partial_json" else self.plain).stream(),
                                              dtype, name, data]
                st[2], st[3] = name, data
                delta[field] = st[0].feed(delta[field])
            return [(name, data)]
        if typ == "content_block_stop":
            out = self._flush(data.get("index"))
            out.append((name, data))
            return out
        out = self._flush() if typ in ("message_delta", "message_stop", "error") else []
        restore_tree(data, self.plain, self.js)
        out.append((name, data))
        return out

    def _flush(self, only=None):
        out = []
        for idx, (sr, dtype, name, template) in self.streams.items():
            if only is not None and idx != only:
                continue
            rest = sr.flush()
            if rest:
                field = "text" if dtype == "text_delta" else "partial_json"
                out.append((name, {**template, "delta": {"type": dtype, field: rest}}))
        return out

    def finish(self):
        return self._flush()


def stream_restorer(provider: str, endpoint: str, plain: RestoreDict, js: RestoreDict,
                    notes: Counter | None = None) -> _Base:
    if provider == "anthropic":
        return AnthropicStream(plain, js, notes)
    if endpoint.startswith("/responses"):
        return ResponsesStream(plain, js)
    return ChatStream(plain, js)
