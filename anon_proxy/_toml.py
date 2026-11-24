"""TOML loader: uses stdlib tomllib on Python 3.11+, with a minimal fallback for Python 3.9/3.10."""
from __future__ import annotations

import ast
import re

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None


def load_toml_bytes(raw: bytes) -> dict:
    if tomllib is not None:
        return tomllib.loads(raw.decode("utf-8-sig"))
    return _parse_minimal_toml(raw.decode("utf-8-sig"))


def _strip_comment(line: str) -> str:
    in_str = None
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if in_str == '"' and ch == "\\":
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_str is None:
                in_str = ch
            elif in_str == ch:
                in_str = None
        elif ch == "#" and in_str is None:
            return line[:i]
    return line


def _split_top(s: str, sep: str = ",") -> list[str]:
    parts, cur, depth_b, depth_c, in_str, esc = [], [], 0, 0, None, False
    for ch in s:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if in_str == '"' and ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if ch in ('"', "'"):
            if in_str is None:
                in_str = ch
            elif in_str == ch:
                in_str = None
            cur.append(ch)
        elif in_str is None:
            if ch == "[":
                depth_b += 1
            elif ch == "]":
                depth_b -= 1
            elif ch == "{":
                depth_c += 1
            elif ch == "}":
                depth_c -= 1
            if ch == sep and depth_b == 0 and depth_c == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
            cur.append(ch)
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_key(k: str) -> str:
    k = k.strip()
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        return ast.literal_eval(k) if k.startswith('"') else k[1:-1]
    return k


def _parse_val(v: str):
    v = v.strip()
    if v in ("true", "false"):
        return v == "true"
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return ast.literal_eval(v) if v.startswith('"') else v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_parse_val(x) for x in _split_top(inner, ",")] if inner else []
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        out = {}
        for item in (_split_top(inner, ",") if inner else []):
            ik, _, iv = _split_kv(item)
            out[_parse_key(ik)] = _parse_val(iv)
        return out
    if re.fullmatch(r"[+-]?\d[\d_]*", v):
        return int(v.replace("_", ""))
    if re.fullmatch(r"[+-]?\d[\d_]*\.[\d_]*([eE][+-]?\d+)?", v):
        return float(v.replace("_", ""))
    raise ValueError(f"unsupported TOML value: {v!r}")


def _split_kv(line: str) -> tuple[str, str, str]:
    in_str, esc = None, False
    for i, ch in enumerate(line):
        if esc:
            esc = False
            continue
        if in_str == '"' and ch == "\\":
            esc = True
            continue
        if ch in ('"', "'"):
            if in_str is None:
                in_str = ch
            elif in_str == ch:
                in_str = None
        elif ch == "=" and in_str is None:
            return line[:i].strip(), "=", line[i + 1:].strip()
    raise ValueError(f"expected key = value in TOML line: {line!r}")


def _parse_minimal_toml(text: str) -> dict:
    root: dict = {}
    cur = root
    lines = [_strip_comment(ln).strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        if not ln:
            continue
        if ln.startswith("[") and ln.endswith("]") and "=" not in ln:
            sec = ln[1:-1].strip()
            cur = root
            for part in _split_top(sec, "."):
                key = _parse_key(part)
                cur = cur.setdefault(key, {})
            continue
        while ln.count("[") > ln.count("]") or ln.count("{") > ln.count("}"):
            if i >= len(lines):
                break
            ln += " " + lines[i]
            i += 1
        k, _, v = _split_kv(ln)
        cur[_parse_key(k)] = _parse_val(v)
    return root
