"""Virtual Provider bridge: wash any downstream protocol into one canonical Anthropic
`/v1/messages` request, and synthesize the downstream protocol's response on the way back.

Supported downstream dialects -> unified Anthropic `/v1/messages`:
  * OpenAI Responses API (`/v1/responses`, `/v1/responses/compact`) — used by Codex CLI
  * OpenAI Chat Completions API (`/v1/chat/completions`) — used by open-source coding clients
  * Anthropic Messages API (`/v1/messages`) — canonicalized to the exact same envelope

Key multi-turn invariants handled here:
  1. Signed Claude `thinking` / `redacted_thinking` blocks round-trip statelesly inside
     Codex's `reasoning.encrypted_content` (`ap1:<hmac>:<b64>`), keeping history byte-identical.
  2. Codex `type: "custom"` freeform tools (such as `apply_patch`) are wrapped as single-string
     JSON `input_schema` tools for Claude and unwrapped back into `custom_tool_call` items.
  3. Anthropic message role alternation, `thinking`-first ordering, and `tool_result`-first
     pairing (with synthetic error results for aborted calls) are strictly enforced.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from .masking import RestoreDict, restore_tree
from .protocols import (BILLING_PREFIX, SIGNED_BLOCK_TYPES, THINKING_BINDING_BETA, AnthropicStream,
                        RequestContext, _strip_git_in_tree, mask_tree, note_thinking_drops)

SAFE_TOOL_ID = re.compile(r"[^a-zA-Z0-9_-]")
COMPACT_PROMPT = (
    "Summarize the conversation so far into a concise, self-contained technical handoff. "
    "Preserve user goals, file paths, architectural decisions, commands executed, test outcomes, "
    "and exact remaining tasks so work can continue seamlessly from this summary alone."
)


@dataclass
class ProviderConfig:
    target: str = ""                           # "" (passthrough per endpoint) | "anthropic" (unify all to Claude)
    default_model: str = "claude-sonnet-4-20250514"
    force_model: bool = False                  # True: even "claude-*" names are rewritten to default_model
    model_map: dict[str, str] = field(default_factory=dict)
    default_max_tokens: int = 16384
    thinking_mode: str = "auto"                # "auto" | "adaptive" | "off"
    strip_client_betas: bool = True            # strip client-identifying beta flags in unified mode
    sanitize_system: bool = True               # neutralize Codex/OpenAI self-branding & bundled .system skills


@dataclass
class BridgeMeta:
    client_protocol: str                       # "responses" | "responses_compact" | "chat_completions" | "anthropic"
    client_model: str
    upstream_model: str
    stream: bool
    custom_tools: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ signed opaque blobs

def pack_signed_block(block: dict, secret: bytes) -> str:
    raw = json.dumps(block, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(secret, b"ap1-think\x00" + raw, hashlib.sha256).hexdigest()[:32]
    return f"ap1:{sig}:{base64.urlsafe_b64encode(raw).decode('ascii')}"


def unpack_signed_block(token: str, secret: bytes) -> dict | None:
    if not isinstance(token, str) or not token.startswith("ap1:"):
        return None
    parts = token.split(":", 2)
    if len(parts) != 3:
        return None
    try:
        raw = base64.urlsafe_b64decode(parts[2].encode("ascii"))
    except Exception:
        return None
    expected = hmac.new(secret, b"ap1-think\x00" + raw, hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(parts[1], expected):
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and obj.get("type") in SIGNED_BLOCK_TYPES else None


def pack_compaction(summary: str, secret: bytes) -> str:
    raw = summary.encode("utf-8")
    sig = hmac.new(secret, b"ap1-compact\x00" + raw, hashlib.sha256).hexdigest()[:32]
    return f"ap1c:{sig}:{base64.urlsafe_b64encode(raw).decode('ascii')}"


def unpack_compaction(token: str, secret: bytes) -> str | None:
    if not isinstance(token, str) or not token.startswith("ap1c:"):
        return None
    parts = token.split(":", 2)
    if len(parts) != 3:
        return None
    try:
        raw = base64.urlsafe_b64decode(parts[2].encode("ascii"))
    except Exception:
        return None
    expected = hmac.new(secret, b"ap1-compact\x00" + raw, hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(parts[1], expected):
        return None
    return raw.decode("utf-8", "replace")


# ------------------------------------------------------------------ request normalization helpers

def _clean_tool_id(raw_id: str | None, ctx: RequestContext, fallback_idx: int) -> str:
    val = (raw_id or "").strip()
    if not val:
        return "toolu_" + ctx.pseudo("tool_id", str(fallback_idx), 24)
    cleaned = SAFE_TOOL_ID.sub("_", val)[:64]
    return cleaned or ("toolu_" + ctx.pseudo("tool_id", val, 24))


def _resolve_model(raw_model: str | None, prov: ProviderConfig) -> str:
    m = (raw_model or "").strip()
    if prov.force_model or not m:
        return prov.default_model
    if m in prov.model_map:
        return prov.model_map[m]
    if m.startswith("claude-"):
        return m
    return prov.default_model


def _clean_schema(schema) -> dict:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = dict(schema)
    if out.get("type") != "object":
        out["type"] = "object"
    if not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    return out


def _image_block(url_or_obj) -> dict | None:
    url = url_or_obj.get("url") if isinstance(url_or_obj, dict) else url_or_obj
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:") and ";base64," in url:
        head, _, data = url.partition(";base64,")
        media = head[5:] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _text_blocks_from_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        content = [content]
    blocks: list[dict] = []
    for p in content if isinstance(content, list) else []:
        if isinstance(p, str):
            if p:
                blocks.append({"type": "text", "text": p})
            continue
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if pt in ("text", "input_text", "output_text"):
            txt = p.get("text")
            if isinstance(txt, str) and txt:
                blocks.append({"type": "text", "text": txt})
        elif pt == "refusal" and isinstance(p.get("refusal"), str) and p["refusal"]:
            blocks.append({"type": "text", "text": p["refusal"]})
        elif pt in ("input_image", "image_url"):
            img = _image_block(p.get("image_url") or p.get("url"))
            if img:
                blocks.append(img)
        elif pt == "image" and isinstance(p.get("source"), dict):
            blocks.append(p)
    return blocks


def _tool_output_content(output):
    if isinstance(output, str):
        return output
    blocks = _text_blocks_from_parts(output)
    if blocks:
        return blocks
    return json.dumps(output, ensure_ascii=False) if output is not None else ""


def _parse_json_args(raw_args) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except ValueError:
            return {"raw": raw_args}
    return {}


SKILLS_BLOCK_RX = re.compile(r"<skills_instructions>[\s\S]*?</skills_instructions>")
SKILL_ROOT_LINE_RX = re.compile(r"^\s*-\s*`(?P<id>r\d+)`\s*=\s*`(?P<path>[^`]+)`\s*$")
CODEX_SELF_ID_SENTENCES = (
    (re.compile(r"You are a coding agent running in the Codex CLI,\s*a terminal-based coding assistant\.\s*"
                r"Codex CLI is an open source project led by OpenAI\.\s*"),
     "You are a coding agent running in a terminal-based coding assistant. "),
    (re.compile(r"\n*Within this context,\s*Codex refers to the open-source agentic coding interface"
                r"[^\n]*\n*"),
     "\n\n"),
    (re.compile(r"\bCodex CLI\b"), "the CLI"),
    (re.compile(r"\$CODEX_HOME\b"), "$AGENT_HOME"),
)


def _sanitize_skills_block(block_text: str) -> str:
    """Drop bundled Codex `.system` skills (openai-docs, imagegen, skill-installer, etc.) while keeping
    any custom user skills."""
    sys_roots: set[str] = set()
    kept_roots: list[str] = []
    kept_skills: list[str] = []
    section = ""
    for line in block_text.splitlines():
        s = line.strip()
        if s.startswith("### Skill roots"):
            section = "roots"
            continue
        if s.startswith("### Available skills"):
            section = "skills"
            continue
        if section == "roots" and s.startswith("- "):
            m = SKILL_ROOT_LINE_RX.match(line)
            if m and ("/.system" in m.group("path") or "\\.system" in m.group("path")):
                sys_roots.add(m.group("id"))
            else:
                kept_roots.append(line)
        elif section == "skills" and s.startswith("- "):
            if any(f"(file: {rid}/" in line or f"(file: `{rid}/" in line for rid in sys_roots):
                continue
            if re.match(r"^-\s*(?:openai-docs|imagegen|skill-creator|skill-installer|create-statusline)\s*:", s):
                continue
            kept_skills.append(line)
    if not kept_skills:
        return ""
    out = ["<skills_instructions>", "## Skills",
           "A skill is a set of local instructions stored in a `SKILL.md` file."]
    if kept_roots:
        out.append("### Skill roots")
        out.extend(kept_roots)
    out.append("### Available skills")
    out.extend(kept_skills)
    out.append("</skills_instructions>")
    return "\n".join(out)


def _neutralize_system_text(txt: str) -> str:
    if "<skills_instructions>" in txt:
        txt = SKILLS_BLOCK_RX.sub(lambda m: _sanitize_skills_block(m.group(0)), txt)
    for rx, repl in CODEX_SELF_ID_SENTENCES:
        txt = rx.sub(repl, txt)
    return txt


def _clean_system_blocks(blocks: list[dict], strip_billing: bool, sanitize_system: bool = False) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        txt = b.get("text")
        if not isinstance(txt, str):
            continue
        if strip_billing:
            if txt.lstrip().startswith(BILLING_PREFIX):
                continue
            lines = [ln for ln in txt.splitlines() if not ln.lstrip().startswith(BILLING_PREFIX)]
            txt = "\n".join(lines)
        if sanitize_system:
            txt = _neutralize_system_text(txt)
        if txt.strip():
            out.append({"type": "text", "text": txt})
    return out


def _enforce_anthropic_invariants(messages: list[dict]) -> list[dict]:
    """Coalesce same-role turns, put thinking first in assistant messages, put tool_result
    first in user messages, and pair every tool_use with a tool_result."""
    coalesced: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}] if content else []
        elif isinstance(content, list):
            blocks = [b for b in content if isinstance(b, dict) and not (
                b.get("type") == "text" and not b.get("text"))]
        else:
            blocks = []
        if not blocks:
            continue
        if coalesced and coalesced[-1]["role"] == role:
            coalesced[-1]["content"].extend(blocks)
        else:
            coalesced.append({"role": role, "content": list(blocks)})

    if not coalesced:
        return [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    if coalesced[0]["role"] != "user":
        coalesced.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue."}]})

    # Reorder blocks within each turn and ensure tool_use -> tool_result pairing.
    fixed: list[dict] = []
    i = 0
    while i < len(coalesced):
        cur = coalesced[i]
        if cur["role"] == "assistant":
            signed = [b for b in cur["content"] if b.get("type") in SIGNED_BLOCK_TYPES]
            others = [b for b in cur["content"] if b.get("type") not in SIGNED_BLOCK_TYPES]
            cur["content"] = signed + others
            fixed.append(cur)
            tool_ids = [b["id"] for b in others if b.get("type") == "tool_use" and b.get("id")]
            if tool_ids:
                if i + 1 < len(coalesced) and coalesced[i + 1]["role"] == "user":
                    nxt = coalesced[i + 1]
                    present = {b.get("tool_use_id") for b in nxt["content"] if b.get("type") == "tool_result"}
                    missing = [{"type": "tool_result", "tool_use_id": tid, "content": "cancelled", "is_error": True}
                               for tid in tool_ids if tid not in present]
                    results = [b for b in nxt["content"] if b.get("type") == "tool_result"] + missing
                    rest = [b for b in nxt["content"] if b.get("type") != "tool_result"]
                    nxt["content"] = results + rest
                else:
                    synth = [{"type": "tool_result", "tool_use_id": tid, "content": "cancelled", "is_error": True}
                             for tid in tool_ids]
                    fixed.append({"role": "user", "content": synth})
        else:
            results = [b for b in cur["content"] if b.get("type") == "tool_result"]
            rest = [b for b in cur["content"] if b.get("type") != "tool_result"]
            cur["content"] = results + rest
            fixed.append(cur)
        i += 1
    return fixed


# ------------------------------------------------------------------ inbound translation -> Anthropic

def _from_responses(endpoint: str, body: dict, ctx: RequestContext, prov: ProviderConfig) -> tuple[dict, BridgeMeta]:
    is_compact = endpoint.endswith("/compact")
    client_model = str(body.get("model") or prov.default_model)
    up_model = _resolve_model(body.get("model"), prov)
    meta = BridgeMeta(
        client_protocol="responses_compact" if is_compact else "responses",
        client_model=client_model,
        upstream_model=up_model,
        stream=False if is_compact else bool(body.get("stream")),
    )

    sys_blocks: list[dict] = []
    if isinstance(body.get("instructions"), str) and body["instructions"].strip():
        sys_blocks.append({"type": "text", "text": body["instructions"]})

    raw_input = body.get("input")
    items = [{"type": "message", "role": "user", "content": raw_input}] if isinstance(raw_input, str) \
        else (raw_input if isinstance(raw_input, list) else [])

    messages: list[dict] = []
    seen_turn = False
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        typ = item.get("type") or ("message" if "role" in item else "")
        role = item.get("role")
        if typ == "message" and role in ("system", "developer"):
            blocks = _text_blocks_from_parts(item.get("content"))
            if not seen_turn:
                sys_blocks.extend(blocks)
            elif blocks:
                joined = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
                if joined:
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": f"<developer_instruction>\n{joined}\n</developer_instruction>"}]})
        elif typ == "message" and role in ("user", "assistant"):
            seen_turn = True
            blocks = _text_blocks_from_parts(item.get("content"))
            if blocks:
                messages.append({"role": role, "content": blocks})
        elif typ == "reasoning":
            seen_turn = True
            unpacked = unpack_signed_block(item.get("encrypted_content") or "", ctx._secret)
            if unpacked is not None:
                messages.append({"role": "assistant", "content": [unpacked]})
        elif typ == "compaction":
            seen_turn = True
            summary = unpack_compaction(item.get("encrypted_content") or "", ctx._secret)
            if summary:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"<conversation_summary>\n{summary}\n</conversation_summary>"}]})
        elif typ == "function_call":
            seen_turn = True
            tid = _clean_tool_id(item.get("call_id") or item.get("id"), ctx, idx)
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": tid,
                "name": str(item.get("name") or "tool"),
                "input": _parse_json_args(item.get("arguments")),
            }]})
        elif typ == "custom_tool_call":
            seen_turn = True
            tid = _clean_tool_id(item.get("call_id") or item.get("id"), ctx, idx)
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": tid,
                "name": str(item.get("name") or "custom_tool"),
                "input": {"input": str(item.get("input") or "")},
            }]})
        elif typ in ("function_call_output", "custom_tool_call_output"):
            seen_turn = True
            tid = _clean_tool_id(item.get("call_id"), ctx, idx)
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tid,
                "content": _tool_output_content(item.get("output")),
            }]})

    if is_compact:
        messages.append({"role": "user", "content": [{"type": "text", "text": COMPACT_PROMPT}]})

    tools: list[dict] = []
    if not is_compact and body.get("tool_choice") != "none":
        for t in body.get("tools") or []:
            if not isinstance(t, dict):
                continue
            ttype = t.get("type")
            if ttype == "function" and t.get("name"):
                entry = {"name": str(t["name"])}
                if isinstance(t.get("description"), str) and t["description"]:
                    entry["description"] = t["description"]
                entry["input_schema"] = _clean_schema(t.get("parameters"))
                tools.append(entry)
            elif ttype == "custom" and t.get("name"):
                name = str(t["name"])
                meta.custom_tools.add(name)
                desc = str(t.get("description") or "").strip()
                suffix = "Pass the complete raw tool payload as a single string in the `input` property."
                tools.append({
                    "name": name,
                    "description": f"{desc}\n\n{suffix}".strip(),
                    "input_schema": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Raw input text"}},
                        "required": ["input"],
                    },
                })

    max_tok = body.get("max_output_tokens")
    max_tokens = int(max_tok) if isinstance(max_tok, int) and max_tok > 0 else prov.default_max_tokens
    wants_thinking = (prov.thinking_mode == "adaptive") or (
        prov.thinking_mode == "auto" and isinstance(body.get("reasoning"), dict)
        and body["reasoning"].get("effort") not in (None, "none"))
    if is_compact:
        wants_thinking = False

    anth = _assemble_canonical_anthropic(
        model=up_model,
        max_tokens=max_tokens,
        system_blocks=_clean_system_blocks(sys_blocks, ctx.privacy.strip_billing_header, prov.sanitize_system),
        messages=_enforce_anthropic_invariants(messages),
        tools=tools,
        raw_tool_choice=None if is_compact else body.get("tool_choice"),
        thinking={"type": "adaptive"} if wants_thinking else None,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop_sequences=None,
        stream=meta.stream,
        ctx=ctx,
    )
    return anth, meta


def _from_chat_completions(body: dict, ctx: RequestContext, prov: ProviderConfig) -> tuple[dict, BridgeMeta]:
    client_model = str(body.get("model") or prov.default_model)
    up_model = _resolve_model(body.get("model"), prov)
    meta = BridgeMeta(
        client_protocol="chat_completions",
        client_model=client_model,
        upstream_model=up_model,
        stream=bool(body.get("stream")),
    )

    sys_blocks: list[dict] = []
    messages: list[dict] = []
    seen_turn = False
    for idx, m in enumerate(body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            blocks = _text_blocks_from_parts(m.get("content"))
            if not seen_turn:
                sys_blocks.extend(blocks)
            elif blocks:
                joined = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
                if joined:
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": f"<developer_instruction>\n{joined}\n</developer_instruction>"}]})
        elif role == "user":
            seen_turn = True
            blocks = _text_blocks_from_parts(m.get("content"))
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            seen_turn = True
            blocks = _text_blocks_from_parts(m.get("content"))
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                tid = _clean_tool_id(tc.get("id"), ctx, idx)
                blocks.append({
                    "type": "tool_use",
                    "id": tid,
                    "name": str(fn.get("name") or "tool"),
                    "input": _parse_json_args(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            seen_turn = True
            tid = _clean_tool_id(m.get("tool_call_id"), ctx, idx)
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tid,
                "content": _tool_output_content(m.get("content")),
            }]})

    tools: list[dict] = []
    if body.get("tool_choice") != "none":
        for t in body.get("tools") or []:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if t.get("type") == "function" else t
            if isinstance(fn, dict) and fn.get("name"):
                entry = {"name": str(fn["name"])}
                if isinstance(fn.get("description"), str) and fn["description"]:
                    entry["description"] = fn["description"]
                entry["input_schema"] = _clean_schema(fn.get("parameters") or fn.get("input_schema"))
                tools.append(entry)

    max_tok = body.get("max_completion_tokens") or body.get("max_tokens")
    max_tokens = int(max_tok) if isinstance(max_tok, int) and max_tok > 0 else prov.default_max_tokens
    wants_thinking = (prov.thinking_mode == "adaptive") or (
        prov.thinking_mode == "auto" and body.get("reasoning_effort") not in (None, "none"))
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else (stop if isinstance(stop, list) else None)

    anth = _assemble_canonical_anthropic(
        model=up_model,
        max_tokens=max_tokens,
        system_blocks=_clean_system_blocks(sys_blocks, ctx.privacy.strip_billing_header, prov.sanitize_system),
        messages=_enforce_anthropic_invariants(messages),
        tools=tools,
        raw_tool_choice=body.get("tool_choice"),
        thinking={"type": "adaptive"} if wants_thinking else None,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop_sequences=stops,
        stream=meta.stream,
        ctx=ctx,
    )
    return anth, meta


def _from_anthropic(endpoint: str, body: dict, ctx: RequestContext, prov: ProviderConfig) -> tuple[dict, BridgeMeta]:
    client_model = str(body.get("model") or prov.default_model)
    up_model = _resolve_model(body.get("model"), prov)
    meta = BridgeMeta(
        client_protocol="anthropic_count" if endpoint.endswith("/count_tokens") else "anthropic",
        client_model=client_model,
        upstream_model=up_model,
        stream=False if endpoint.endswith("/count_tokens") else bool(body.get("stream")),
    )
    raw_sys = body.get("system")
    if isinstance(raw_sys, str):
        sys_blocks = [{"type": "text", "text": raw_sys}]
    elif isinstance(raw_sys, list):
        sys_blocks = [b for b in raw_sys if isinstance(b, dict)]
    else:
        sys_blocks = []

    tools: list[dict] = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") and t.get("type") != "custom":   # Anthropic server/computer-use tools
            tools.append(dict(t))
        elif t.get("name"):
            entry = {"name": str(t["name"])}
            if isinstance(t.get("description"), str) and t["description"]:
                entry["description"] = t["description"]
            entry["input_schema"] = _clean_schema(t.get("input_schema"))
            if isinstance(t.get("cache_control"), dict):
                entry["cache_control"] = t["cache_control"]
            tools.append(entry)

    thinking = dict(body["thinking"]) if isinstance(body.get("thinking"), dict) else (
        {"type": "adaptive"} if prov.thinking_mode == "adaptive" else None)
    if prov.thinking_mode == "off":
        thinking = None

    max_tok = body.get("max_tokens")
    max_tokens = int(max_tok) if isinstance(max_tok, int) and max_tok > 0 else prov.default_max_tokens

    anth = _assemble_canonical_anthropic(
        model=up_model,
        max_tokens=max_tokens,
        system_blocks=_clean_system_blocks(sys_blocks, ctx.privacy.strip_billing_header, prov.sanitize_system),
        messages=_enforce_anthropic_invariants(body.get("messages") or []),
        tools=tools,
        raw_tool_choice=body.get("tool_choice"),
        thinking=thinking,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop_sequences=body.get("stop_sequences"),
        stream=meta.stream,
        ctx=ctx,
        is_count_tokens=endpoint.endswith("/count_tokens"),
    )
    return anth, meta


def _map_tool_choice(raw, has_tools: bool) -> dict | None:
    if not has_tools or raw is None or raw == "auto":
        return None
    if raw == "required":
        return {"type": "any"}
    if raw == "none":
        return {"type": "none"}
    if isinstance(raw, dict):
        t = raw.get("type")
        if t in ("auto", "any", "none"):
            return None if t == "auto" else {"type": t}
        if t == "tool" and raw.get("name"):
            return {"type": "tool", "name": str(raw["name"])}
        if t == "function":
            name = raw.get("name") or (raw.get("function") or {}).get("name")
            if name:
                return {"type": "tool", "name": str(name)}
    return None


def _assemble_canonical_anthropic(
    *, model: str, max_tokens: int, system_blocks: list[dict], messages: list[dict],
    tools: list[dict], raw_tool_choice, thinking: dict | None, temperature, top_p,
    stop_sequences, stream: bool, ctx: RequestContext, is_count_tokens: bool = False,
) -> dict:
    """Build the outbound Anthropic JSON with a strictly uniform key order across all clients."""
    tc = _map_tool_choice(raw_tool_choice, bool(tools))
    # Anthropic rejects forced tool_choice ("any" / "tool") when thinking is active.
    if thinking and tc and tc.get("type") in ("any", "tool"):
        thinking = None
    if thinking and isinstance(thinking.get("budget_tokens"), int):
        if max_tokens <= thinking["budget_tokens"]:
            max_tokens = thinking["budget_tokens"] + 4096

    out: dict = {"model": model}
    if not is_count_tokens:
        out["max_tokens"] = max_tokens
    if system_blocks:
        out["system"] = system_blocks
    out["messages"] = messages
    if tools:
        out["tools"] = tools
    if tc is not None:
        out["tool_choice"] = tc
    if thinking is not None:
        th = dict(thinking)
        if not is_count_tokens and ctx.privacy.thinking_mismatch != "off" \
                and th.get("type") == "adaptive" and "block_binding" not in th:
            th["block_binding"] = {"prefix_mismatch_behavior": ctx.privacy.thinking_mismatch}
            ctx.extra_betas.append(THINKING_BINDING_BETA)
        out["thinking"] = th
    else:
        if isinstance(temperature, (int, float)):
            out["temperature"] = temperature
        if isinstance(top_p, (int, float)):
            out["top_p"] = top_p
    if isinstance(stop_sequences, list) and stop_sequences and not is_count_tokens:
        out["stop_sequences"] = [str(s) for s in stop_sequences if isinstance(s, str)]
    if not is_count_tokens:
        out["stream"] = bool(stream)
        if ctx.privacy.end_user_id == "pseudonym":
            out["metadata"] = {"user_id": ctx.pseudo("end_user", "")}
    return out


def to_canonical_anthropic(provider_hint: str, endpoint: str, body: dict, ctx: RequestContext,
                           prov: ProviderConfig) -> tuple[str, dict, BridgeMeta]:
    """Translate & scrub any downstream POST body into an Anthropic endpoint + body."""
    if ctx.privacy.strip_silent_git:
        _strip_git_in_tree(body)
    if endpoint == "/responses" or endpoint.startswith("/responses/"):
        anth_body, meta = _from_responses(endpoint, body, ctx, prov)
        target_ep = "/messages"
    elif endpoint in ("/chat/completions", "/completions"):
        anth_body, meta = _from_chat_completions(body, ctx, prov)
        target_ep = "/messages"
    else:
        anth_body, meta = _from_anthropic(endpoint, body, ctx, prov)
        target_ep = "/messages/count_tokens" if endpoint.endswith("/count_tokens") else "/messages"

    ctx.learn_tree(anth_body)
    mask_tree(anth_body, ctx)
    return target_ep, anth_body, meta


# ------------------------------------------------------------------ outbound response synthesis

def _responses_usage(u: dict | None) -> dict:
    u = u or {}
    cached = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    inp = int(u.get("input_tokens") or 0) + cached + cache_write
    out = int(u.get("output_tokens") or 0)
    return {
        "input_tokens": inp,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": out,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": inp + out,
    }


def _chat_usage(u: dict | None) -> dict:
    ru = _responses_usage(u)
    return {
        "prompt_tokens": ru["input_tokens"],
        "completion_tokens": ru["output_tokens"],
        "total_tokens": ru["total_tokens"],
    }


def _unwrap_custom_input(raw_json: str) -> str:
    if not raw_json:
        return ""
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
            return parsed["input"]
        return json.dumps(parsed, ensure_ascii=False)
    except ValueError:
        return raw_json


class AnthropicToResponsesStream:
    """Translate Anthropic `/v1/messages` SSE into OpenAI `/v1/responses` SSE for Codex CLI."""

    def __init__(self, plain: RestoreDict, js: RestoreDict, meta: BridgeMeta, secret: bytes,
                 notes: Counter | None = None):
        self.plain, self.js = plain, js
        self.meta = meta
        self.secret = secret
        self.notes = notes if notes is not None else Counter()
        self.seq = 0
        self.created_at = int(time.time())
        self.resp_id = f"resp_{int(time.time() * 1000)}"
        self.usage: dict = {}
        self.stop_reason: str | None = None
        self.blocks: dict[int, dict] = {}
        self.output_items: list[dict] = []
        self.started = False
        self.completed = False

    def _ev(self, kind: str, **fields) -> tuple[str, dict]:
        payload = {"type": kind, "sequence_number": self.seq, **fields}
        self.seq += 1
        return kind, payload

    def _base_resp(self, status: str, output: list) -> dict:
        return {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.meta.client_model,
            "output": output,
        }

    def event(self, name, data) -> list[tuple[str | None, dict | str]]:
        if not isinstance(data, dict):
            return []
        typ = data.get("type")
        out: list[tuple[str | None, dict | str]] = []

        if typ == "message_start":
            msg = data.get("message") or {}
            note_thinking_drops(msg, self.notes)
            if msg.get("id"):
                self.resp_id = "resp_" + str(msg["id"]).removeprefix("msg_")
            self.usage.update(msg.get("usage") or {})
            self.started = True
            out.append(self._ev("response.created", response=self._base_resp("in_progress", [])))
            out.append(self._ev("response.in_progress", response=self._base_resp("in_progress", [])))
            return out

        if typ == "content_block_start":
            idx = int(data.get("index", 0))
            cb = data.get("content_block") or {}
            btype = cb.get("type")
            out_idx = len(self.output_items) + len(self.blocks)
            if btype in SIGNED_BLOCK_TYPES:
                item_id = f"rs_{self.resp_id}_{idx}"
                self.blocks[idx] = {
                    "kind": "reasoning", "out_idx": out_idx, "item_id": item_id,
                    "raw_type": btype, "raw_thinking": str(cb.get("thinking") or ""),
                    "raw_sig": str(cb.get("signature") or ""), "raw_data": cb.get("data"),
                    "sr": self.plain.stream(), "summary": "", "summary_opened": False,
                }
                out.append(self._ev("response.output_item.added", output_index=out_idx, item={
                    "id": item_id, "type": "reasoning", "status": "in_progress", "summary": []}))
            elif btype == "text":
                item_id = f"msg_{self.resp_id}_{idx}"
                sr = self.plain.stream()
                seed = sr.feed(cb["text"]) if isinstance(cb.get("text"), str) and cb["text"] else ""
                self.blocks[idx] = {
                    "kind": "message", "out_idx": out_idx, "item_id": item_id,
                    "sr": sr, "text": seed,
                }
                out.append(self._ev("response.output_item.added", output_index=out_idx, item={
                    "id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}))
                out.append(self._ev("response.content_part.added", item_id=item_id, output_index=out_idx,
                                    content_index=0, part={"type": "output_text", "text": "", "annotations": []}))
                if seed:
                    out.append(self._ev("response.output_text.delta", item_id=item_id, output_index=out_idx,
                                        content_index=0, delta=seed))
            elif btype == "tool_use":
                tname = str(cb.get("name") or "tool")
                call_id = str(cb.get("id") or f"call_{self.resp_id}_{idx}")
                item_id = f"fc_{call_id}"
                is_custom = tname in self.meta.custom_tools
                sr = self.js.stream()
                seed_obj = cb.get("input")
                seed_json = json.dumps(seed_obj, ensure_ascii=False) if isinstance(seed_obj, dict) and seed_obj else ""
                restored_seed = sr.feed(seed_json) if seed_json else ""
                self.blocks[idx] = {
                    "kind": "custom_tool" if is_custom else "function_call",
                    "out_idx": out_idx, "item_id": item_id, "call_id": call_id, "name": tname,
                    "sr": sr, "buf": restored_seed,
                }
                if is_custom:
                    out.append(self._ev("response.output_item.added", output_index=out_idx, item={
                        "id": item_id, "type": "custom_tool_call", "status": "in_progress",
                        "call_id": call_id, "name": tname, "input": ""}))
                else:
                    out.append(self._ev("response.output_item.added", output_index=out_idx, item={
                        "id": item_id, "type": "function_call", "status": "in_progress",
                        "call_id": call_id, "name": tname, "arguments": ""}))
                    if restored_seed:
                        out.append(self._ev("response.function_call_arguments.delta", item_id=item_id,
                                            output_index=out_idx, delta=restored_seed))
            return out

        if typ == "content_block_delta":
            idx = int(data.get("index", 0))
            st = self.blocks.get(idx)
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            if st is None:
                return out
            if dtype == "thinking_delta" and st["kind"] == "reasoning":
                chunk = str(delta.get("thinking") or "")
                st["raw_thinking"] += chunk
                vis = st["sr"].feed(chunk)
                if vis:
                    if not st["summary_opened"]:
                        st["summary_opened"] = True
                        out.append(self._ev("response.reasoning_summary_part.added", item_id=st["item_id"],
                                            output_index=st["out_idx"], summary_index=0,
                                            part={"type": "summary_text", "text": ""}))
                    st["summary"] += vis
                    out.append(self._ev("response.reasoning_summary_text.delta", item_id=st["item_id"],
                                        output_index=st["out_idx"], summary_index=0, delta=vis))
            elif dtype == "signature_delta" and st["kind"] == "reasoning":
                st["raw_sig"] += str(delta.get("signature") or "")
            elif dtype == "text_delta" and st["kind"] == "message":
                vis = st["sr"].feed(str(delta.get("text") or ""))
                if vis:
                    st["text"] += vis
                    out.append(self._ev("response.output_text.delta", item_id=st["item_id"],
                                        output_index=st["out_idx"], content_index=0, delta=vis))
            elif dtype == "input_json_delta" and st["kind"] in ("function_call", "custom_tool"):
                vis = st["sr"].feed(str(delta.get("partial_json") or ""))
                if vis:
                    st["buf"] += vis
                    if st["kind"] == "function_call":
                        out.append(self._ev("response.function_call_arguments.delta", item_id=st["item_id"],
                                            output_index=st["out_idx"], delta=vis))
            return out

        if typ == "content_block_stop":
            idx = int(data.get("index", 0))
            out.extend(self._close_block(idx))
            return out

        if typ == "message_delta":
            d = data.get("delta") or {}
            if d.get("stop_reason"):
                self.stop_reason = d["stop_reason"]
            for k, v in (data.get("usage") or {}).items():
                if isinstance(v, int):
                    self.usage[k] = v
            return out

        if typ == "message_stop":
            out.extend(self.finish())
            return out

        if typ == "error":
            err = data.get("error") or {}
            out.append(("error", {"type": "error", "message": err.get("message", "upstream error"),
                                  "code": err.get("type")}))
            self.completed = True
            return out
        return out

    def _close_block(self, idx: int) -> list[tuple[str | None, dict | str]]:
        st = self.blocks.pop(idx, None)
        if st is None:
            return []
        out: list[tuple[str | None, dict | str]] = []
        rest = st["sr"].flush()
        out_idx, item_id = st["out_idx"], st["item_id"]

        if st["kind"] == "reasoning":
            if rest:
                if not st["summary_opened"]:
                    st["summary_opened"] = True
                    out.append(self._ev("response.reasoning_summary_part.added", item_id=item_id,
                                        output_index=out_idx, summary_index=0,
                                        part={"type": "summary_text", "text": ""}))
                st["summary"] += rest
                out.append(self._ev("response.reasoning_summary_text.delta", item_id=item_id,
                                    output_index=out_idx, summary_index=0, delta=rest))
            summary_list = []
            if st["summary_opened"]:
                part = {"type": "summary_text", "text": st["summary"]}
                summary_list.append(part)
                out.append(self._ev("response.reasoning_summary_text.done", item_id=item_id,
                                    output_index=out_idx, summary_index=0, text=st["summary"]))
                out.append(self._ev("response.reasoning_summary_part.done", item_id=item_id,
                                    output_index=out_idx, summary_index=0, part=part))
            raw_block = {"type": "redacted_thinking", "data": st["raw_data"]} \
                if st["raw_type"] == "redacted_thinking" \
                else {"type": "thinking", "thinking": st["raw_thinking"], "signature": st["raw_sig"]}
            item = {
                "id": item_id, "type": "reasoning", "status": "completed",
                "summary": summary_list, "encrypted_content": pack_signed_block(raw_block, self.secret),
            }
            self.output_items.append(item)
            out.append(self._ev("response.output_item.done", output_index=out_idx, item=item))

        elif st["kind"] == "message":
            if rest:
                st["text"] += rest
                out.append(self._ev("response.output_text.delta", item_id=item_id, output_index=out_idx,
                                    content_index=0, delta=rest))
            part = {"type": "output_text", "text": st["text"], "annotations": []}
            item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant", "content": [part]}
            self.output_items.append(item)
            out.append(self._ev("response.output_text.done", item_id=item_id, output_index=out_idx,
                                content_index=0, text=st["text"]))
            out.append(self._ev("response.content_part.done", item_id=item_id, output_index=out_idx,
                                content_index=0, part=part))
            out.append(self._ev("response.output_item.done", output_index=out_idx, item=item))

        elif st["kind"] == "function_call":
            if rest:
                st["buf"] += rest
                out.append(self._ev("response.function_call_arguments.delta", item_id=item_id,
                                    output_index=out_idx, delta=rest))
            args = st["buf"] or "{}"
            item = {
                "id": item_id, "type": "function_call", "status": "completed",
                "call_id": st["call_id"], "name": st["name"], "arguments": args,
            }
            self.output_items.append(item)
            out.append(self._ev("response.function_call_arguments.done", item_id=item_id,
                                output_index=out_idx, arguments=args))
            out.append(self._ev("response.output_item.done", output_index=out_idx, item=item))

        elif st["kind"] == "custom_tool":
            if rest:
                st["buf"] += rest
            raw_input = _unwrap_custom_input(st["buf"])
            if raw_input:
                out.append(self._ev("response.custom_tool_call_input.delta", item_id=item_id,
                                    output_index=out_idx, delta=raw_input))
            out.append(self._ev("response.custom_tool_call_input.done", item_id=item_id,
                                output_index=out_idx, input=raw_input))
            item = {
                "id": item_id, "type": "custom_tool_call", "status": "completed",
                "call_id": st["call_id"], "name": st["name"], "input": raw_input,
            }
            self.output_items.append(item)
            out.append(self._ev("response.output_item.done", output_index=out_idx, item=item))
        return out

    def finish(self) -> list[tuple[str | None, dict | str]]:
        if self.completed:
            return []
        out: list[tuple[str | None, dict | str]] = []
        for idx in sorted(list(self.blocks.keys())):
            out.extend(self._close_block(idx))
        status = "incomplete" if self.stop_reason == "max_tokens" else "completed"
        ev_name = "response.incomplete" if status == "incomplete" else "response.completed"
        final = {**self._base_resp(status, self.output_items), "usage": _responses_usage(self.usage)}
        out.append(self._ev(ev_name, response=final))
        self.completed = True
        return out


class AnthropicToChatStream:
    """Translate Anthropic `/v1/messages` SSE into OpenAI `/v1/chat/completions` SSE."""

    def __init__(self, plain: RestoreDict, js: RestoreDict, meta: BridgeMeta, notes: Counter | None = None):
        self.plain, self.js = plain, js
        self.meta = meta
        self.notes = notes if notes is not None else Counter()
        self.created = int(time.time())
        self.cid = f"chatcmpl-{int(time.time() * 1000)}"
        self.blocks: dict[int, dict] = {}
        self.tool_count = 0
        self.stop_reason = "end_turn"
        self.usage: dict = {}
        self.done = False

    def _chunk(self, delta: dict, finish=None, usage=None) -> tuple[None, dict]:
        obj: dict = {
            "id": self.cid, "object": "chat.completion.chunk",
            "created": self.created, "model": self.meta.client_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            obj["usage"] = usage
        return None, obj

    def event(self, name, data) -> list[tuple[None, dict | str]]:
        if not isinstance(data, dict):
            return []
        typ = data.get("type")
        out: list[tuple[None, dict | str]] = []

        if typ == "message_start":
            msg = data.get("message") or {}
            note_thinking_drops(msg, self.notes)
            if msg.get("id"):
                self.cid = "chatcmpl-" + str(msg["id"]).removeprefix("msg_")
            self.usage.update(msg.get("usage") or {})
            out.append(self._chunk({"role": "assistant", "content": ""}))
            return out

        if typ == "content_block_start":
            idx = int(data.get("index", 0))
            cb = data.get("content_block") or {}
            btype = cb.get("type")
            if btype == "thinking":
                self.blocks[idx] = {"kind": "thinking", "sr": self.plain.stream()}
            elif btype == "text":
                sr = self.plain.stream()
                seed = sr.feed(cb["text"]) if isinstance(cb.get("text"), str) and cb["text"] else ""
                self.blocks[idx] = {"kind": "text", "sr": sr}
                if seed:
                    out.append(self._chunk({"content": seed}))
            elif btype == "tool_use":
                t_idx = self.tool_count
                self.tool_count += 1
                sr = self.js.stream()
                call_id = str(cb.get("id") or f"call_{idx}")
                tname = str(cb.get("name") or "tool")
                seed_obj = cb.get("input")
                seed = sr.feed(json.dumps(seed_obj, ensure_ascii=False)) \
                    if isinstance(seed_obj, dict) and seed_obj else ""
                self.blocks[idx] = {"kind": "tool", "t_idx": t_idx, "sr": sr}
                out.append(self._chunk({"tool_calls": [{
                    "index": t_idx, "id": call_id, "type": "function",
                    "function": {"name": tname, "arguments": seed},
                }]}))
            return out

        if typ == "content_block_delta":
            idx = int(data.get("index", 0))
            st = self.blocks.get(idx)
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            if st is None:
                return out
            if dtype == "thinking_delta" and st["kind"] == "thinking":
                vis = st["sr"].feed(str(delta.get("thinking") or ""))
                if vis:
                    out.append(self._chunk({"reasoning_content": vis}))
            elif dtype == "text_delta" and st["kind"] == "text":
                vis = st["sr"].feed(str(delta.get("text") or ""))
                if vis:
                    out.append(self._chunk({"content": vis}))
            elif dtype == "input_json_delta" and st["kind"] == "tool":
                vis = st["sr"].feed(str(delta.get("partial_json") or ""))
                if vis:
                    out.append(self._chunk({"tool_calls": [{
                        "index": st["t_idx"], "function": {"arguments": vis},
                    }]}))
            return out

        if typ == "content_block_stop":
            idx = int(data.get("index", 0))
            out.extend(self._flush_block(idx))
            return out

        if typ == "message_delta":
            d = data.get("delta") or {}
            if d.get("stop_reason"):
                self.stop_reason = d["stop_reason"]
            for k, v in (data.get("usage") or {}).items():
                if isinstance(v, int):
                    self.usage[k] = v
            return out

        if typ == "message_stop":
            out.extend(self.finish())
            return out
        return out

    def _flush_block(self, idx: int) -> list[tuple[None, dict | str]]:
        st = self.blocks.pop(idx, None)
        if st is None:
            return []
        rest = st["sr"].flush()
        if not rest:
            return []
        if st["kind"] == "thinking":
            return [self._chunk({"reasoning_content": rest})]
        if st["kind"] == "text":
            return [self._chunk({"content": rest})]
        if st["kind"] == "tool":
            return [self._chunk({"tool_calls": [{"index": st["t_idx"], "function": {"arguments": rest}}]})]
        return []

    def finish(self) -> list[tuple[None, dict | str]]:
        if self.done:
            return []
        out: list[tuple[None, dict | str]] = []
        for idx in sorted(list(self.blocks.keys())):
            out.extend(self._flush_block(idx))
        finish_map = {"tool_use": "tool_calls", "max_tokens": "length"}
        finish = finish_map.get(self.stop_reason, "stop")
        out.append(self._chunk({}, finish=finish, usage=_chat_usage(self.usage)))
        out.append((None, "[DONE]"))
        self.done = True
        return out


def convert_anthropic_json(parsed: dict, plain: RestoreDict, js: RestoreDict, meta: BridgeMeta,
                           secret: bytes, notes: Counter | None = None) -> dict:
    """Convert a non-streaming Anthropic `/v1/messages` JSON response to the downstream dialect."""
    if notes is not None:
        note_thinking_drops(parsed, notes)

    if meta.client_protocol in ("anthropic", "anthropic_count"):
        return restore_tree(parsed, plain, js)

    now = int(time.time())
    msg_id = str(parsed.get("id") or f"msg_{now}").removeprefix("msg_")

    if meta.client_protocol == "responses_compact":
        # Keep the summary masked inside encrypted_content; on replay it passes cleanly through mask_tree.
        texts = [b.get("text", "") for b in parsed.get("content") or []
                 if isinstance(b, dict) and b.get("type") == "text"]
        summary = "\n".join(t for t in texts if t).strip()
        return {
            "id": f"resp_{msg_id}",
            "object": "response.compaction",
            "created_at": now,
            "output": [{
                "id": f"cmp_{msg_id}",
                "type": "compaction",
                "encrypted_content": pack_compaction(summary, secret),
            }],
            "usage": _responses_usage(parsed.get("usage")),
        }

    if meta.client_protocol == "responses":
        output: list[dict] = []
        for idx, b in enumerate(parsed.get("content") or []):
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype in SIGNED_BLOCK_TYPES:
                summary_list = []
                if btype == "thinking" and b.get("thinking"):
                    summary_list.append({"type": "summary_text", "text": plain.full(str(b["thinking"]))})
                output.append({
                    "id": f"rs_{msg_id}_{idx}",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": summary_list,
                    "encrypted_content": pack_signed_block(b, secret),
                })
            elif btype == "text":
                txt = plain.full(str(b.get("text") or ""))
                output.append({
                    "id": f"msg_{msg_id}_{idx}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": txt, "annotations": []}],
                })
            elif btype == "tool_use":
                tname = str(b.get("name") or "tool")
                call_id = str(b.get("id") or f"call_{msg_id}_{idx}")
                inp_json = js.full(json.dumps(b.get("input") or {}, ensure_ascii=False))
                if tname in meta.custom_tools:
                    output.append({
                        "id": f"fc_{call_id}",
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": tname,
                        "input": _unwrap_custom_input(inp_json),
                    })
                else:
                    output.append({
                        "id": f"fc_{call_id}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": tname,
                        "arguments": inp_json,
                    })
        status = "incomplete" if parsed.get("stop_reason") == "max_tokens" else "completed"
        return {
            "id": f"resp_{msg_id}",
            "object": "response",
            "created_at": now,
            "status": status,
            "model": meta.client_model,
            "output": output,
            "usage": _responses_usage(parsed.get("usage")),
        }

    # chat_completions
    texts, thoughts, tool_calls = [], [], []
    for idx, b in enumerate(parsed.get("content") or []):
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "thinking" and b.get("thinking"):
            thoughts.append(plain.full(str(b["thinking"])))
        elif btype == "text":
            texts.append(plain.full(str(b.get("text") or "")))
        elif btype == "tool_use":
            inp_json = js.full(json.dumps(b.get("input") or {}, ensure_ascii=False))
            tool_calls.append({
                "id": str(b.get("id") or f"call_{msg_id}_{idx}"),
                "type": "function",
                "function": {"name": str(b.get("name") or "tool"), "arguments": inp_json},
            })
    message: dict = {"role": "assistant", "content": "\n".join(texts) if texts else (None if tool_calls else "")}
    if thoughts:
        message["reasoning_content"] = "\n".join(thoughts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = {"tool_use": "tool_calls", "max_tokens": "length"}.get(parsed.get("stop_reason"), "stop")
    return {
        "id": f"chatcmpl-{msg_id}",
        "object": "chat.completion",
        "created": now,
        "model": meta.client_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": _chat_usage(parsed.get("usage")),
    }


def translate_anthropic_error(parsed: dict, meta: BridgeMeta) -> dict:
    if meta.client_protocol in ("anthropic", "anthropic_count"):
        return parsed
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, dict):
        return {"error": {"message": err.get("message", "upstream error"),
                          "type": err.get("type", "api_error"), "code": None}}
    return {"error": {"message": json.dumps(parsed, ensure_ascii=False), "type": "api_error", "code": None}}


def bridge_stream_restorer(meta: BridgeMeta, plain: RestoreDict, js: RestoreDict,
                           secret: bytes, notes: Counter | None = None):
    if meta.client_protocol == "responses":
        return AnthropicToResponsesStream(plain, js, meta, secret, notes)
    if meta.client_protocol == "chat_completions":
        return AnthropicToChatStream(plain, js, meta, notes)
    return AnthropicStream(plain, js, notes)
