"""Reversible masking: identifying strings -> stable surrogates, and back again.

Surrogates are deterministic per downstream user (HMAC with the server secret): the
same original always gets the same stand-in, which keeps prompt caching working and
multi-turn conversations consistent, while two downstream users get different
stand-ins for the same value.

Masking works on a list of segments. Every replacement is "frozen" so later patterns
never rewrite the inside of a surrogate that an earlier pattern produced.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

# ------------------------------------------------------------------ patterns

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),                       # OpenAI / Anthropic style keys
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),   # JWT
    re.compile(r"(?i)\bbearer\s+(?P<v>[A-Za-z0-9_\-.=+/]{20,})"),
]

# key = value / "key": "value" where the key name says it is a secret; only the value is masked.
# Every quantifier is bounded: unbounded ones around the keyword made long "a.a.a..." input quadratic.
ASSIGNMENT = re.compile(
    r"(?i)\b[a-z0-9_.\-]{0,40}?(?:secret|token|passw(?:or)?d|pwd|api[_\-]?key|access[_\-]?key|private[_\-]?key|"
    r"auth[_\-]?key|credentials?)[a-z0-9_.\-]{0,40}[\"']?[ \t]{0,8}[:=][ \t]{0,8}(?P<q>[\"']?)"
    r"(?P<v>[^\s\"'`,;#)}\]]{8,200})(?P=q)")

ASSIGNMENT_HINT = re.compile(
    r"(?i)secret|token|passw|pwd|api[_\-]?key|access[_\-]?key|private[_\-]?key|auth[_\-]?key|credential")

EMAIL = re.compile(
    r"(?<![\w.+\-])[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+(?P<tld>[A-Za-z]{2,24})(?![\w\-])")

GIT_REMOTE = re.compile(
    r"(?i)\b(?:github\.com|gitlab\.com|bitbucket\.org|gitee\.com|codeberg\.org)[:/]"
    r"(?P<org>[A-Za-z0-9][A-Za-z0-9_.\-]{0,38})/(?P<repo>[A-Za-z0-9_.\-]{1,100}?)(?:\.git)?"
    r"(?=$|[\s\"'`)\]>#?/,;:])")

_NAME = r"[^\\/\s\"'<>:|?*`;,()\[\]{}=]+"
PATH_USER_PATTERNS = [
    # C:\Users\name\...  (also C:/Users/name and JSON-escaped C:\\Users\\name; "John Smith\" allowed)
    re.compile(r"(?i)(?<![\w])[a-z]:(?:\\|/)+users(?:\\|/)+"
               r"(?P<v>[a-z0-9_.\-]+(?: [a-z0-9_.\-]+){1,3}(?=\\)|" + _NAME + ")"),
    re.compile(r"(?<![\w.\-~/\\])/(?:Users|home)/(?P<v>" + _NAME + ")"),     # /Users/name, /home/name
    re.compile(r"(?i)/mnt/[a-z]/users/(?P<v>" + _NAME + ")"),                # WSL
]

HOSTNAME = re.compile(
    r"(?i)\b(?:desktop|laptop)-[a-z0-9]{5,15}\b"
    r"|\b[a-z0-9][\w\-]{0,40}-(?:macbook(?:-pro|-air)?|mbp|imac|mac-mini)(?:-\d+)?(?:\.local)?\b")

IPV4 = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w]|\.\d)")

TIMEZONE_TAG = re.compile(r"(<timezone>)([^<]{1,64})(</timezone>)")

# ------------------------------------------------------------------ exclusions

FILE_EXT_TLDS = frozenset(
    "png jpg jpeg gif svg webp ico bmp avif js mjs cjs ts tsx jsx css scss less json md txt py rs go java kt "
    "rb php html htm xml yaml yml toml lock map wasm pdf zip gz tar so dll exe sh ps1 vue svelte".split())
EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net", "example.invalid")
EXCLUDED_PATH_NAMES = frozenset(
    "public default all users default user shared guest user username name you yourname your-name me "
    "xxx runner defaultapppool".split()) | {"all users", "default user"}
COMMON_WORDS = frozenset(
    "admin administrator user users root dev test ubuntu runner vagrant pi ec2-user guest owner home app "
    "node docker data work code main default public shared mac pc desktop laptop".split())
GIT_RESERVED = frozenset(
    "settings orgs features login logout about pricing marketplace topics explore issues pulls notifications "
    "sponsors apps site security enterprise collections trending new organizations search users contact join "
    "codespaces copilot dashboard".split())
TEST_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
IP_POOLS = ("203.0.113", "198.51.100", "192.0.2")   # surrogates use .100-.254, all the same length per pool
SECRET_PLACEHOLDER_HINTS = ("example", "your", "placeholder", "xxxx", "changeme", "dummy", "redacted", "****")


def _plausible_secret(value: str) -> bool:
    if not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
        return False
    if value[0] in "$%<{(" or "(" in value or "[" in value:
        return False
    low = value.lower()
    if low.startswith(("process.", "os.", "env.", "config.", "settings.", "self.", "this.", "getenv")):
        return False
    return not any(h in low for h in SECRET_PLACEHOLDER_HINTS)


# ------------------------------------------------------------------ config / vault

@dataclass
class MaskConfig:
    secrets: bool = True
    emails: bool = True
    ips: bool = True
    path_usernames: bool = True
    hostnames: bool = True
    git_remotes: bool = True
    literals: tuple[str, ...] = ()
    keep: frozenset[str] = field(default_factory=frozenset)
    timezone: str | None = "UTC"


class Vault:
    """One per downstream user: original <-> surrogate, learned usernames, and a cache."""

    def __init__(self, key: bytes, cache_chars: int = 32_000_000):
        self._key = key
        self._fwd: dict[tuple[str, str], str] = {}
        self._rev: dict[str, str] = {}
        self._learned: dict[str, None] = {}
        self._learned_rx: re.Pattern | None = None
        self.generation = 0
        self._lock = threading.RLock()
        self._cache: OrderedDict = OrderedDict()
        self._cache_chars = 0
        self._cache_limit = cache_chars

    # -- surrogates
    def _digest(self, kind: str, original: str) -> str:
        return hmac.new(self._key, f"{kind}\x00{original}".encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _format(kind: str, h: str) -> str:
        if kind == "ip":
            slot = int(h[:8], 16) % (len(IP_POOLS) * 155)
            pool, last = divmod(slot, 155)
            return f"{IP_POOLS[pool]}.{100 + last}"
        if kind == "email":
            return f"mail{h[:8]}@example.com"
        if kind == "secret":
            return f"SECRET_{h[:12]}"
        prefix = {"user": "user", "host": "host-", "org": "org", "repo": "repo"}.get(kind, "name")
        return f"{prefix}{h[:8]}"

    def surrogate(self, kind: str, original: str) -> str:
        with self._lock:
            s = self._fwd.get((kind, original))
            if s is None:
                s, n = self._format(kind, self._digest(kind, original)), 0
                while s in self._rev and self._rev[s] != original:   # hash collision: probe
                    n += 1
                    s = self._format(kind, self._digest(kind, f"{original}\x00{n}"))
                self._fwd[(kind, original)] = s
                self._rev[s] = original
            return s

    def is_surrogate(self, s: str) -> bool:
        return s in self._rev

    def mapping(self, surrogates) -> dict[str, str]:
        with self._lock:
            return {s: self._rev[s] for s in surrogates if s in self._rev}

    # -- learned literals (usernames seen in home-directory paths)
    def learn(self, value: str) -> None:
        v = value.strip()
        low = v.lower()
        if len(v) < 3 or low in COMMON_WORDS or low in EXCLUDED_PATH_NAMES or v in self._rev:
            return
        with self._lock:
            if low in self._learned:
                return
            self._learned[low] = None
            names = sorted(self._learned, key=len, reverse=True)
            # distinctive names (6+ chars) also match when glued to other text, e.g. after a shell
            # strips the backslashes out of C:\Users\name; short ones need word boundaries
            long_alts = "|".join(re.escape(x) for x in names if len(x) >= 6)
            short_alts = "|".join(re.escape(x) for x in names if len(x) < 6)
            parts = ([f"(?:{long_alts})"] if long_alts else []) + \
                ([rf"(?<![A-Za-z0-9_])(?:{short_alts})(?![A-Za-z0-9_])"] if short_alts else [])
            self._learned_rx = re.compile("|".join(parts), re.IGNORECASE)
            self.generation += 1

    @property
    def learned_regex(self) -> re.Pattern | None:
        return self._learned_rx

    # -- frozen masking: once a string has been masked, later requests reuse exactly that result,
    # even if a username learned since then would now match inside it. Conversation history is
    # resent on every turn and must stay byte-identical (prompt caching, and the history check on
    # replayed thinking blocks); whatever an old string revealed was already sent upstream anyway.
    def cache_get(self, text: str):
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
            return hit

    def cache_put(self, text: str, masked: str, used: set, kinds: Counter) -> None:
        with self._lock:
            if text in self._cache:
                return
            self._cache[text] = (masked, tuple(used), dict(kinds))
            self._cache_chars += len(text) + len(masked) + 64
            while self._cache_chars > self._cache_limit and self._cache:
                old, (old_masked, _, _) = self._cache.popitem(last=False)
                self._cache_chars -= len(old) + len(old_masked) + 64


# ------------------------------------------------------------------ masker

Segment = tuple[bool, str]   # (frozen, text)


def _apply(segments: list[Segment], rx: re.Pattern, repl) -> list[Segment]:
    out: list[Segment] = []
    for frozen, s in segments:
        if frozen or not s:
            out.append((frozen, s))
            continue
        pos = 0
        for m in rx.finditer(s):
            pieces = repl(m)
            if pieces is None:
                continue
            if m.start() > pos:
                out.append((False, s[pos:m.start()]))
            out.extend(pieces)
            pos = m.end()
        if pos < len(s):
            out.append((False, s[pos:]))
    return out


def _literal_regex(words) -> re.Pattern | None:
    words = sorted({w for w in words if w and len(w) >= 2}, key=len, reverse=True)
    if not words:
        return None
    parts = []
    for w in words:
        e = re.escape(w)
        parts.append(rf"(?<![A-Za-z0-9_]){e}(?![A-Za-z0-9_])" if w.isascii() else e)
    return re.compile("|".join(parts), re.IGNORECASE)


class Masker:
    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg
        self.keep = frozenset(k.lower() for k in cfg.keep)
        self.literal_rx = _literal_regex(cfg.literals)

    def learn(self, text: str, vault: Vault) -> None:
        if not self.cfg.path_usernames or not isinstance(text, str) or len(text) < 8:
            return
        for rx in PATH_USER_PATTERNS:
            for m in rx.finditer(text):
                name = m.group("v")
                if self._username_ok(name, vault):
                    vault.learn(name)

    def _username_ok(self, name: str, vault: Vault) -> bool:
        return bool(name) and name.lower() not in EXCLUDED_PATH_NAMES and name[0] not in "$%<{~." \
            and not vault.is_surrogate(name) and name.lower() not in self.keep

    def mask(self, text: str, vault: Vault, used: set, stats: Counter) -> str:
        if not isinstance(text, str) or len(text) < 3:
            return text
        hit = vault.cache_get(text)
        if hit is not None:
            masked, surs, kinds = hit
            used.update(surs)
            stats.update(kinds)
            return masked

        cfg, keep = self.cfg, self.keep
        local_used: set[str] = set()
        local_stats: Counter = Counter()

        def freeze(kind: str, original: str) -> Segment:
            s = vault.surrogate(kind, original)
            local_used.add(s)
            local_stats[kind] += 1
            return True, s

        def swap(m: re.Match, group, kind: str, check=None):
            val = m.group(group)
            if not val or val.lower() in keep or vault.is_surrogate(val) or (check and not check(val)):
                return None
            (s, e), (ms, me) = m.span(group), m.span()
            pieces: list[Segment] = []
            if s > ms:
                pieces.append((False, m.string[ms:s]))
            pieces.append(freeze(kind, val))
            if me > e:
                pieces.append((False, m.string[e:me]))
            return pieces

        def email(m: re.Match):
            val = m.group(0)
            domain = val.rsplit("@", 1)[1].lower()
            if m.group("tld").lower() in FILE_EXT_TLDS or domain.endswith(EXAMPLE_DOMAINS):
                return None
            return swap(m, 0, "email")

        def git(m: re.Match):
            org, repo = m.group("org"), m.group("repo")
            if org.lower() in GIT_RESERVED or org.lower() in keep or f"{org}/{repo}".lower() in keep \
                    or vault.is_surrogate(org):
                return None
            return [(False, m.string[m.start():m.start("org")]), freeze("org", org), (False, "/"),
                    freeze("repo", repo), (False, m.string[m.end("repo"):m.end()])]

        def path(m: re.Match):
            name = m.group("v")
            if not self._username_ok(name, vault):
                return None
            vault.learn(name)
            return swap(m, "v", "user")

        def ip(m: re.Match):
            try:
                addr = ipaddress.ip_address(m.group(0))
            except ValueError:
                return None
            if addr.is_loopback or addr.is_unspecified or addr.is_multicast or addr.is_link_local \
                    or addr.is_reserved or any(addr in n for n in TEST_NETS):
                return None
            return swap(m, 0, "ip")

        segs: list[Segment] = [(False, text)]
        if cfg.timezone:
            segs = _apply(segs, TIMEZONE_TAG, lambda m: None if m.group(2) == cfg.timezone else
                          [(True, m.group(1) + cfg.timezone + m.group(3))])
        if cfg.secrets:
            for rx in SECRET_PATTERNS:
                segs = _apply(segs, rx, lambda m: swap(m, "v" if "v" in m.re.groupindex else 0, "secret"))
            if ASSIGNMENT_HINT.search(text):   # cheap pre-check; ASSIGNMENT is the costliest pattern
                segs = _apply(segs, ASSIGNMENT, lambda m: swap(m, "v", "secret", _plausible_secret))
        if cfg.git_remotes:
            segs = _apply(segs, GIT_REMOTE, git)
        if cfg.emails:
            segs = _apply(segs, EMAIL, email)
        if cfg.path_usernames:
            for rx in PATH_USER_PATTERNS:
                segs = _apply(segs, rx, path)
            if vault.learned_regex is not None:
                segs = _apply(segs, vault.learned_regex, lambda m: swap(m, 0, "user"))
        if self.literal_rx is not None:
            segs = _apply(segs, self.literal_rx, lambda m: swap(m, 0, "name"))
        if cfg.hostnames:
            segs = _apply(segs, HOSTNAME, lambda m: swap(m, 0, "host"))
        if cfg.ips:
            segs = _apply(segs, IPV4, ip)

        masked = "".join(s for _, s in segs)
        used.update(local_used)
        stats.update(local_stats)
        vault.cache_put(text, masked, local_used, local_stats)
        return masked


# ------------------------------------------------------------------ restoring

class RestoreDict:
    """surrogate -> original for one request. json_escape=True for text that lives inside a
    JSON string (tool-call arguments), where originals must be JSON-escaped."""

    def __init__(self, mapping: dict[str, str], json_escape: bool = False):
        if json_escape:
            mapping = {s: json.dumps(o, ensure_ascii=False)[1:-1] for s, o in mapping.items()}
        self.map = mapping
        self.rx = re.compile("|".join(re.escape(s) for s in sorted(mapping, key=len, reverse=True))) \
            if mapping else None
        self.prefixes = frozenset(s[:i] for s in mapping for i in range(1, len(s)))
        self.maxlen = max((len(s) for s in mapping), default=0)

    def full(self, text: str) -> str:
        if self.rx is None or not text:
            return text
        return self.rx.sub(lambda m: self.map[m.group(0)], text)

    def stream(self) -> "StreamRestorer":
        return StreamRestorer(self)


class StreamRestorer:
    """Restores surrogates in a stream of deltas, holding back a tail that might be the
    start of a surrogate until the next delta (or flush) decides it."""

    def __init__(self, d: RestoreDict):
        self.d = d
        self.pending = ""

    def feed(self, chunk: str) -> str:
        d = self.d
        if d.rx is None:
            return chunk
        buf = self.pending + chunk
        out, last = [], 0
        for m in d.rx.finditer(buf):
            out.append(buf[last:m.start()])
            out.append(d.map[m.group(0)])
            last = m.end()
        tail = buf[last:]
        hold = 0
        for k in range(min(len(tail), d.maxlen - 1), 0, -1):
            if tail[-k:] in d.prefixes:
                hold = k
                break
        out.append(tail[:len(tail) - hold])
        self.pending = tail[len(tail) - hold:]
        return "".join(out)

    def flush(self) -> str:
        rest, self.pending = self.pending, ""
        return self.d.full(rest)


RESTORE_SKIP_KEYS = frozenset({"encrypted_content", "signature", "thinking", "data", "obfuscation"})
JSON_TEXT_KEYS = frozenset({"arguments", "partial_json"})


def restore_tree(obj, plain: RestoreDict, js: RestoreDict):
    """Restore surrogates in every string of a parsed JSON value (in place)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if k not in RESTORE_SKIP_KEYS:
                    obj[k] = (js if k in JSON_TEXT_KEYS else plain).full(v)
            elif isinstance(v, (dict, list)):
                restore_tree(v, plain, js)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = plain.full(v)
            elif isinstance(v, (dict, list)):
                restore_tree(v, plain, js)
    return obj
