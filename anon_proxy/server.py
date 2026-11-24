"""HTTP front end: authenticate the downstream user, scrub the request, forward it with the
owner's own API key, and restore the response on the way back."""
from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import http.client
import itertools
import json
import os
import re
import secrets
import ssl
import sys
import threading
import time
import urllib.parse
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Python 3.14+
    from compression import zstd
except ImportError:  # pragma: no cover
    zstd = None

from ._toml import load_toml_bytes
from .bridge import (BridgeMeta, ProviderConfig, bridge_stream_restorer, convert_anthropic_json,
                     to_canonical_anthropic, translate_anthropic_error)
from .masking import MaskConfig, Masker, RestoreDict, Vault, restore_tree
from .protocols import (Privacy, RequestContext, note_thinking_drops, scrub_anthropic, scrub_openai,
                        stream_restorer)

DEFAULT_OPENAI_UA = "codex_cli_rs/0.170.0 (Windows 10.0.26200; x86_64) WindowsTerminal"
DEFAULT_ANTHROPIC_UA = "Anthropic/Python 0.52.0"
DEFAULT_KEEP = ("git@github.com", "noreply@github.com", "noreply@anthropic.com", "8.8.8.8", "8.8.4.4",
                "1.1.1.1", "1.0.0.1")
PASS_RESPONSE_HEADERS = frozenset({"x-request-id", "request-id", "retry-after", "openai-processing-ms"})
PASS_RESPONSE_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")
MAX_BODY = 100 * 1024 * 1024
COMMENT = "\x00comment"


# ------------------------------------------------------------------ settings

@dataclass
class Upstream:
    base_url: str
    api_key_env: str
    user_agent: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    api_key_file: str | None = None   # read on every request, so it can be created or replaced while running


def _parse_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                        v = v[1:-1]
                    elif " #" in v:
                        v = v.split(" #", 1)[0].strip()
                    out[k] = v
    except OSError:
        pass
    return out


def upstream_key(up: Upstream) -> str:
    """Environment variable first, then api_key_file / .env (re-read on every request)."""
    key = os.environ.get(up.api_key_env, "").strip()
    if key:
        return key
    candidates = [up.api_key_file] if up.api_key_file else []
    if ".env" not in [os.path.basename(p) for p in candidates if p]:
        base = os.path.dirname(up.api_key_file) if up.api_key_file else os.getcwd()
        candidates.append(os.path.join(base, ".env"))
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        kv = _parse_env_file(path)
        if kv.get(up.api_key_env):
            return kv[up.api_key_env]
        try:
            with open(path, encoding="utf-8-sig") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        except OSError:
            continue
        if len(lines) == 1 and "=" not in lines[0]:
            return lines[0].strip("\"'")
    return ""


def key_status(up: Upstream) -> str:
    if os.environ.get(up.api_key_env, "").strip():
        return f"from ${up.api_key_env}"
    if upstream_key(up):
        return f"from {up.api_key_file or '.env'}"
    where = f"${up.api_key_env}" + (f" or {up.api_key_file}" if up.api_key_file else " or .env")
    return f"NOT SET ({where})"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    openai: Upstream = field(default_factory=lambda: Upstream(
        "https://api.openai.com/v1", "OPENAI_API_KEY", DEFAULT_OPENAI_UA))
    anthropic: Upstream = field(default_factory=lambda: Upstream(
        "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", DEFAULT_ANTHROPIC_UA))
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    upstream_proxy: str | None = None
    timeout: float = 900.0
    users: dict[str, str] = field(default_factory=dict)
    privacy: Privacy = field(default_factory=Privacy)
    mask: MaskConfig = field(default_factory=lambda: MaskConfig(keep=frozenset(DEFAULT_KEEP)))
    secret: bytes = b""
    audit_dir: str | None = None
    vault_dir: str | None = None


def _upstream(raw: dict, default: Upstream, base_dir: str) -> Upstream:
    key_file = raw.get("api_key_file") or None
    if key_file and not os.path.isabs(key_file):
        key_file = os.path.join(base_dir, key_file)
    return Upstream(
        base_url=str(raw.get("base_url", default.base_url)).rstrip("/"),
        api_key_env=str(raw.get("api_key_env", default.api_key_env)),
        user_agent=str(raw.get("user_agent", default.user_agent)),
        extra_headers={str(k): str(v) for k, v in (raw.get("extra_headers") or {}).items()},
        api_key_file=key_file or default.api_key_file,
    )


def _load_secret(secret_file: str, base_dir: str) -> bytes:
    env = os.environ.get("ANON_PROXY_SECRET")
    if env:
        return env.encode()
    path = secret_file if os.path.isabs(secret_file) else os.path.join(base_dir, secret_file)
    if os.path.exists(path):
        with open(path, encoding="ascii") as f:
            return bytes.fromhex(f.read().strip())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    value = secrets.token_hex(32)
    with open(path, "w", encoding="ascii") as f:
        f.write(value + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return bytes.fromhex(value)


def load_settings(path: str) -> Settings:
    with open(path, "rb") as f:
        data = load_toml_bytes(f.read())
    base = os.path.dirname(os.path.abspath(path))
    s = Settings()
    host, _, port = str(data.get("listen", "127.0.0.1:8787")).rpartition(":")
    s.host, s.port = host or "127.0.0.1", int(port)
    ups = data.get("upstream") or {}
    s.openai = _upstream(ups.get("openai") or {}, s.openai, base)
    s.anthropic = _upstream(ups.get("anthropic") or {}, s.anthropic, base)
    dotenv = _parse_env_file(os.path.join(base, ".env"))
    pr = data.get("provider") or {}
    default_model = dotenv.get("ANTHROPIC_MODEL") or str(pr.get("default_model", "claude-sonnet-4-20250514"))
    s.provider = ProviderConfig(
        target=str(pr.get("target", "")).strip().lower(),
        default_model=default_model,
        force_model=bool(pr.get("force_model", False)),
        model_map={str(k): str(v) for k, v in (pr.get("model_map") or {}).items()},
        default_max_tokens=int(pr.get("default_max_tokens", 16384)),
        thinking_mode=str(pr.get("thinking_mode", "auto")),
        strip_client_betas=bool(pr.get("strip_client_betas", True)),
        sanitize_system=bool(pr.get("sanitize_system", True)),
    )
    if s.provider.target not in ("", "anthropic"):
        raise ValueError('provider.target must be "" or "anthropic"')
    if s.provider.thinking_mode not in ("auto", "adaptive", "off"):
        raise ValueError('provider.thinking_mode must be "auto", "adaptive" or "off"')
    net = data.get("network") or {}
    s.upstream_proxy = net.get("upstream_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    s.timeout = float(net.get("timeout_seconds", 900))
    pv = data.get("privacy") or {}
    s.privacy = Privacy(end_user_id=str(pv.get("end_user_id", "none")),
                        force_store_false=bool(pv.get("force_store_false", True)),
                        strip_billing_header=bool(pv.get("strip_billing_header", True)),
                        strip_silent_git=bool(pv.get("strip_silent_git", True)),
                        thinking_mismatch=str(pv.get("thinking_mismatch", "drop_block")))
    if s.privacy.end_user_id not in ("none", "pseudonym"):
        raise ValueError('privacy.end_user_id must be "none" or "pseudonym"')
    if s.privacy.thinking_mismatch not in ("drop_block", "error", "off"):
        raise ValueError('privacy.thinking_mismatch must be "drop_block", "error" or "off"')
    mk = data.get("masking") or {}
    style = str(mk.get("surrogate_style", "tagged"))
    if style not in ("tagged", "natural"):
        raise ValueError('masking.surrogate_style must be "tagged" or "natural"')
    s.mask = MaskConfig(
        secrets=bool(mk.get("secrets", True)),
        entropy_secrets=bool(mk.get("entropy_secrets", False)),
        emails=bool(mk.get("emails", True)),
        ips=bool(mk.get("ips", True)),
        path_usernames=bool(mk.get("path_usernames", True)),
        mask_cwd=bool(mk.get("mask_cwd", False)),
        normalize_os_version=bool(mk.get("normalize_os_version", True)),
        hostnames=bool(mk.get("hostnames", True)),
        git_remotes=bool(mk.get("git_remotes", True)),
        literals=tuple(str(x) for x in mk.get("literals", [])),
        keep=frozenset(str(x) for x in mk.get("keep", DEFAULT_KEEP)),
        timezone=str(pv.get("timezone", "UTC")) or None,
        surrogate_style=style,
    )
    s.users = {str(k): str(v) for k, v in (data.get("users") or {}).items()}
    s.secret = _load_secret(str(pv.get("secret_file", "data/secret.key")), base)
    audit = str(pv.get("audit_dir", "") or "")
    s.audit_dir = (audit if os.path.isabs(audit) else os.path.join(base, audit)) if audit else None
    vdir = str(pv.get("vault_dir", "") or "")
    s.vault_dir = (vdir if os.path.isabs(vdir) else os.path.join(base, vdir)) if vdir else None
    return s


# ------------------------------------------------------------------ helpers

def route(path: str, headers) -> tuple[str, str] | None:
    """-> (downstream-dialect-provider, endpoint-after-/v1) or None."""
    p = path.rstrip("/") or "/"
    rest = p[3:] if p.startswith("/v1/") else p
    if rest == "/responses" or rest.startswith("/responses/") or rest in (
            "/chat/completions", "/completions", "/embeddings"):
        return "openai", rest
    if rest in ("/messages", "/messages/count_tokens"):
        return "anthropic", rest
    if rest == "/models" or rest.startswith("/models/"):
        return ("anthropic" if headers.get("anthropic-version") else "openai"), rest
    return None


def decode_body(raw: bytes, encoding: str | None) -> bytes:
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


def upstream_headers(provider: str, up: Upstream, key: str, client_headers, stream: bool,
                     data: bytes | None, extra_betas=(), *, ignore_client_headers: bool = False) -> dict[str, str]:
    """Built from scratch: nothing the client sent is forwarded except what the API needs."""
    h = {"User-Agent": up.user_agent, "Accept": "text/event-stream" if stream else "application/json",
         "Accept-Encoding": "identity"}
    if data is not None:
        h["Content-Type"] = "application/json"
        h["Content-Length"] = str(len(data))
    if provider == "anthropic":
        h["x-api-key"] = key
        h["anthropic-version"] = "2023-06-01" if ignore_client_headers \
            else (client_headers.get("anthropic-version") or "2023-06-01")
        betas = [] if ignore_client_headers else [
            b.strip() for b in (client_headers.get("anthropic-beta") or "").split(",") if b.strip()]
        betas += [b for b in extra_betas if b not in betas]
        if betas:
            h["anthropic-beta"] = ",".join(betas)
    else:
        h["Authorization"] = f"Bearer {key}"
    h.update(up.extra_headers)
    return h


def iter_sse(resp):
    name, data = None, []
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        if not text:
            if data or name:
                yield name, "\n".join(data)
            name, data = None, []
            continue
        if text.startswith(":"):
            yield COMMENT, text
            continue
        fld, _, val = text.partition(":")
        if val.startswith(" "):
            val = val[1:]
        if fld == "event":
            name = val
        elif fld == "data":
            data.append(val)
    if data:
        yield name, "\n".join(data)


def format_sse(name, data) -> bytes:
    if name == COMMENT:
        return (data + "\n\n").encode()
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    head = f"event: {name}\n" if name else ""
    return (head + "".join(f"data: {line}\n" for line in payload.split("\n")) + "\n").encode()


def _normalize_models_for_openai(parsed: dict) -> dict:
    items = []
    for m in parsed.get("data") or []:
        if isinstance(m, dict) and m.get("id"):
            items.append({
                "id": m["id"],
                "object": "model",
                "created": int(m.get("created") or 0),
                "owned_by": str(m.get("owned_by") or "anthropic"),
            })
    return {"object": "list", "data": items}


# ------------------------------------------------------------------ proxy core

class Proxy:
    def __init__(self, settings: Settings):
        self.s = settings
        self.masker = Masker(settings.mask)
        self._vaults: dict[str, Vault] = {}
        self._lock = threading.Lock()
        self._ssl = ssl.create_default_context()
        self._seq = itertools.count(1)
        if settings.audit_dir:
            os.makedirs(settings.audit_dir, exist_ok=True)
        if settings.vault_dir:
            os.makedirs(settings.vault_dir, exist_ok=True)

    def vault(self, user: str) -> Vault:
        with self._lock:
            v = self._vaults.get(user)
            if v is None:
                key = hmac.new(self.s.secret, b"vault\x00" + user.encode(), hashlib.sha256).digest()
                persist = None
                if self.s.vault_dir:
                    safe_user = re.sub(r"[^\w.-]", "_", user)
                    persist = os.path.join(self.s.vault_dir, f"{safe_user}.vault")
                v = self._vaults[user] = Vault(key, style=self.s.mask.surrogate_style, persist_path=persist)
            return v

    def authenticate(self, key: str) -> str | None:
        found = None
        for k, alias in self.s.users.items():
            if hmac.compare_digest(k.encode(), key.encode()):
                found = alias
        return found

    def connect(self, up: Upstream) -> tuple[http.client.HTTPConnection, str]:
        u = urllib.parse.urlsplit(up.base_url)
        if u.scheme == "https":
            if self.s.upstream_proxy:
                p = urllib.parse.urlsplit(self.s.upstream_proxy)
                conn = http.client.HTTPSConnection(p.hostname, p.port or 80, timeout=self.s.timeout,
                                                   context=self._ssl)
                tunnel_headers = {}
                if p.username:
                    cred = f"{urllib.parse.unquote(p.username)}:{urllib.parse.unquote(p.password or '')}"
                    tunnel_headers["Proxy-Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
                conn.set_tunnel(u.hostname, u.port or 443, headers=tunnel_headers)
            else:
                conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=self.s.timeout,
                                                   context=self._ssl)
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.s.timeout)
        return conn, u.path.rstrip("/")

    def audit(self, user: str, method: str, url: str, headers: dict, data: bytes | None) -> None:
        if not self.s.audit_dir:
            return
        safe = {k: ("<redacted>" if k.lower() in ("authorization", "x-api-key") else v) for k, v in headers.items()}
        try:
            body = json.loads(data) if data else None
        except ValueError:
            body = None
        safe_user = re.sub(r"[^\w.-]", "_", user)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{next(self._seq):05d}-{safe_user}.json"
        with open(os.path.join(self.s.audit_dir, name), "w", encoding="utf-8") as f:
            json.dump({"method": method, "url": url, "headers": safe, "body": body}, f, ensure_ascii=False, indent=1)


def make_handler(proxy: Proxy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "anon-proxy"
        sys_version = ""

        def log_message(self, fmt, *args):   # one line per request is printed by _log instead
            pass

        # -- small helpers
        def _send_json(self, code: int, obj) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str, provider: str = "openai") -> None:
            if provider == "anthropic":
                kind = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error"}
                self._send_json(code, {"type": "error", "error": {"type": kind.get(code, "api_error"),
                                                                  "message": message}})
            else:
                self._send_json(code, {"error": {"message": message, "code": None,
                                                 "type": "invalid_request_error" if code < 500 else "api_error"}})

        def _read_body(self) -> bytes:
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                data = bytearray()
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return bytes(data)
                    if len(data) + size > MAX_BODY:
                        raise ValueError("request body too large")
                    data += self.rfile.read(size)
                    self.rfile.readline()
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("request body too large")
            return self.rfile.read(n) if n else b""

        def _user(self) -> str | None:
            auth = self.headers.get("Authorization") or ""
            key = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
            key = key or (self.headers.get("x-api-key") or "").strip()
            return proxy.authenticate(key) if key else None

        def _log(self, user, path, status, t0, ctx, bridged: str = "") -> None:
            stats = ",".join(f"{k}={v}" for k, v in sorted(ctx.stats.items())) if ctx and ctx.stats else "-"
            notes = f" thinking_dropped={ctx.notes['thinking_dropped']}" \
                if ctx and ctx.notes.get("thinking_dropped") else ""
            via = f" [{bridged}]" if bridged else ""
            print(f"{time.strftime('%H:%M:%S')} {user} {self.command} {path}{via} -> {status} "
                  f"{time.monotonic() - t0:.1f}s masked[{stats}]{notes}", file=sys.stderr, flush=True)

        # -- verbs
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path.rstrip("/") in ("", "/health", "/healthz"):
                return self._send_json(200, {"status": "ok"})
            r = route(path, self.headers)
            if r is None:
                return self._error(404, f"unknown endpoint {path}")
            client_prov, endpoint = r
            user = self._user()
            if user is None:
                return self._error(401, "missing or unknown gateway key", client_prov)
            if proxy.s.provider.target == "anthropic":
                self._forward("anthropic", endpoint, user, None, None,
                              client_provider=client_prov, normalize_models=(client_prov == "openai"))
            else:
                self._forward(client_prov, endpoint, user, None, None, client_provider=client_prov)

        def do_POST(self):
            t0 = time.monotonic()
            path, _, query = self.path.partition("?")
            r = route(path, self.headers)
            if r is None:
                return self._error(404, f"unknown endpoint {path}")
            client_prov, endpoint = r
            user = self._user()
            if user is None:
                return self._error(401, "missing or unknown gateway key", client_prov)
            try:
                body = json.loads(decode_body(self._read_body(), self.headers.get("Content-Encoding")))
                if not isinstance(body, dict):
                    raise ValueError("not an object")
            except Exception:
                return self._error(400, "request body must be a JSON object", client_prov)

            v = proxy.vault(user)
            ctx = RequestContext(user, v, proxy.masker, proxy.s.privacy, proxy.s.secret)
            try:
                if proxy.s.provider.target == "anthropic":
                    if endpoint == "/embeddings":
                        return self._error(400, "embeddings endpoint is not supported on Anthropic target", client_prov)
                    target_ep, out_body, bmeta = to_canonical_anthropic(
                        client_prov, endpoint, body, ctx, proxy.s.provider)
                    ignore_hdrs = proxy.s.provider.strip_client_betas or client_prov != "anthropic"
                    status = self._forward("anthropic", target_ep, user, out_body, ctx,
                                           client_provider=client_prov, bridge_meta=bmeta,
                                           ignore_client_headers=ignore_hdrs)
                    self._log(user, path, status, t0, ctx, f"->anthropic:{bmeta.upstream_model}")
                else:
                    (scrub_anthropic if client_prov == "anthropic" else scrub_openai)(endpoint, body, ctx)
                    status = self._forward(client_prov, endpoint + (f"?{query}" if query else ""), user, body, ctx,
                                           client_provider=client_prov)
                    self._log(user, path, status, t0, ctx)
            finally:
                v.flush_disk()

        # -- forwarding
        def _forward(self, provider: str, endpoint_qs: str, user: str, body, ctx, *,
                     client_provider: str | None = None, bridge_meta: BridgeMeta | None = None,
                     ignore_client_headers: bool = False, normalize_models: bool = False) -> int:
            err_prov = client_provider or provider
            up = proxy.s.anthropic if provider == "anthropic" else proxy.s.openai
            key = upstream_key(up)
            if not key:
                self._error(500, f"upstream API key missing: {key_status(up)}", err_prov)
                return 500
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode() if body is not None else None
            headers = upstream_headers(provider, up, key, self.headers, bool(body and body.get("stream")), data,
                                       ctx.extra_betas if ctx else (), ignore_client_headers=ignore_client_headers)
            conn, base_path = proxy.connect(up)
            target = base_path + endpoint_qs
            origin = urllib.parse.urlsplit(up.base_url)
            proxy.audit(user, self.command, f"{origin.scheme}://{origin.netloc}{target}", headers, data)
            try:
                conn.request(self.command, target, body=data, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                self._error(502, f"upstream unreachable: {e.__class__.__name__}: {e}", err_prov)
                return 502
            return self._relay(provider, endpoint_qs.split("?", 1)[0], resp, conn, ctx,
                               bridge_meta=bridge_meta, normalize_models=normalize_models)

        def _relay(self, provider: str, endpoint: str, resp, conn, ctx, *,
                   bridge_meta: BridgeMeta | None = None, normalize_models: bool = False) -> int:
            if ctx is not None:
                plain, js = ctx.restore_dicts()
            else:
                plain = js = RestoreDict({})
            passthru = [(k, v) for k, v in resp.getheaders()
                        if k.lower() in PASS_RESPONSE_HEADERS or k.lower().startswith(PASS_RESPONSE_PREFIXES)]
            ctype = resp.getheader("Content-Type") or ""
            try:
                if resp.status == 200 and "text/event-stream" in ctype:
                    self.send_response(200)
                    for k, v in passthru:
                        self.send_header(k, v)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    if bridge_meta is not None:
                        restorer = bridge_stream_restorer(
                            bridge_meta, plain, js, proxy.s.secret, ctx.notes if ctx else None)
                    else:
                        restorer = stream_restorer(provider, endpoint, plain, js, ctx.notes if ctx else None)
                    for name, raw in iter_sse(resp):
                        if name == COMMENT:
                            self.wfile.write(format_sse(COMMENT, raw))
                            continue
                        data = raw
                        if raw and raw != "[DONE]":
                            try:
                                data = json.loads(raw)
                            except ValueError:
                                pass
                        for out_name, out in restorer.event(name, data):
                            self.wfile.write(format_sse(out_name, out))
                    for out_name, out in restorer.finish():
                        self.wfile.write(format_sse(out_name, out))
                    return 200
                raw = resp.read()
                if "json" in ctype:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            if bridge_meta is not None:
                                if resp.status < 400:
                                    parsed = convert_anthropic_json(
                                        parsed, plain, js, bridge_meta, proxy.s.secret, ctx.notes if ctx else None)
                                else:
                                    parsed = translate_anthropic_error(restore_tree(parsed, plain, js), bridge_meta)
                            elif normalize_models and resp.status < 400:
                                parsed = _normalize_models_for_openai(parsed)
                            else:
                                if ctx is not None:
                                    note_thinking_drops(parsed, ctx.notes)
                                parsed = restore_tree(parsed, plain, js)
                            raw = json.dumps(parsed, ensure_ascii=False).encode()
                    except ValueError:
                        pass
                self.send_response(resp.status)
                for k, v in passthru:
                    self.send_header(k, v)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return resp.status
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return 499   # client went away; closing the upstream connection stops generation
            finally:
                conn.close()

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # clients closing keep-alive connections are normal, not worth a traceback
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                                          TimeoutError)):
            return
        super().handle_error(request, client_address)


def build_server(settings: Settings) -> ThreadingHTTPServer:
    return _Server((settings.host, settings.port), make_handler(Proxy(settings)))


def serve(settings: Settings) -> None:
    server = build_server(settings)
    print(f"anon-proxy listening on http://{settings.host}:{server.server_address[1]}", file=sys.stderr)
    if settings.provider.target == "anthropic":
        print(f"  mode      -> virtual provider (all endpoints unified to Anthropic {settings.provider.default_model})",
              file=sys.stderr)
        print(f"  anthropic -> {settings.anthropic.base_url}  key {key_status(settings.anthropic)}", file=sys.stderr)
    else:
        print(f"  openai    -> {settings.openai.base_url}  key {key_status(settings.openai)}", file=sys.stderr)
        print(f"  anthropic -> {settings.anthropic.base_url}  key {key_status(settings.anthropic)}", file=sys.stderr)
    print(f"  users: {len(settings.users)}  end_user_id: {settings.privacy.end_user_id}  "
          f"surrogates: {settings.mask.surrogate_style}  thinking_mismatch: {settings.privacy.thinking_mismatch}  "
          f"vaults: {settings.vault_dir or 'memory'}  audit: {settings.audit_dir or 'off'}  "
          f"upstream proxy: {settings.upstream_proxy or 'none'}",
          file=sys.stderr)
    if not settings.users:
        print("  WARNING: no [users] configured, every request will get 401", file=sys.stderr)
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
