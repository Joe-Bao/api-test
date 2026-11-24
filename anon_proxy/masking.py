"""Reversible masking: identifying strings -> stable surrogates, and back again.

Surrogates are deterministic per downstream user (HMAC with the server secret): the
same original always gets the same stand-in, which keeps prompt caching working and
multi-turn conversations consistent, while two downstream users get different
stand-ins for the same value.

Masking works on a list of segments. Every replacement is "frozen" so later patterns
never rewrite the inside of a surrogate that an earlier pattern produced.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import threading
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

try:
    import zoneinfo
except ImportError:  # pragma: no cover
    zoneinfo = None  # type: ignore[assignment]

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

# Unlabeled high-entropy token fallback (inspired by packyme/privacy-filter): bounded, requires mixed classes.
HIGH_ENTROPY_TOKEN = re.compile(r"(?<![A-Za-z0-9_/\-.\\])[A-Za-z0-9_\-]{24,128}(?![A-Za-z0-9_/\-.\\])")

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
CURRENT_DATE_TAG = re.compile(r"(<current_date>)(\d{4}-\d{2}-\d{2})(</current_date>)")
TODAYS_DATE_LINE = re.compile(r"((?:Today's date|Current date):[ \t]*)(\d{4}-\d{2}-\d{2})")
OS_VERSION_LINE = re.compile(r"(OS Version:[ \t]*)(Darwin|Linux|Windows|MINGW\S*)[ \t]+[^\r\n<]+", re.IGNORECASE)
# macOS per-UID/UUID temporary directory bucket: (/private)/var/folders/<2ch>/<hash>/
DARWIN_VAR_FOLDER = re.compile(r"(?<=/var/folders/)(?P<v>[a-z0-9]{2}/[a-z0-9_]{16,64})(?=/)", re.IGNORECASE)

CWD_EXTRACT_PATTERNS = [
    re.compile(r"<cwd>([^<\r\n]{2,300})</cwd>"),
    re.compile(r"<root>([^<\r\n]{2,300})</root>"),
    re.compile(r"(?:Primary working directory|Working directory):[ \t]*([^\r\n<]{2,300})"),
]
SYSTEM_PATH_ROOTS = frozenset({
    "users", "home", "mnt", "private", "var", "folders", "tmp", "opt", "srv", "etc", "usr",
    "bin", "sbin", "lib", "volumes", "root", "dev", "proc", "sys", "program files", "windows",
})

# ------------------------------------------------------------------ exclusions & naturalistic pools

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
SECRET_PLACEHOLDER_HINTS = ("example", "your", "placeholder", "xxxx", "changeme", "dummy", "redacted",
                            "****", "tok_test_", "secret_")

_FIRST = ("alex", "jordan", "taylor", "morgan", "casey", "riley", "quinn", "avery",
          "cameron", "dakota", "skyler", "reese", "parker", "hayden", "logan", "rowan")
_LAST = ("mercer", "chen", "reed", "hayes", "brooks", "bennett", "foster", "porter",
         "coleman", "perry", "powell", "long", "patterson", "hughes", "flores", "ward")
_WORDS = ("nimbus", "atlas", "vertex", "beacon", "cobalt", "harbor", "summit", "vector",
          "prism", "horizon", "meridian", "keystone", "solstice", "pioneer", "anchor", "cascade")
_CITIES = ("austin", "denver", "boston", "seattle", "chicago", "portland", "atlanta", "phoenix")


def _canonical_os_version(m: re.Match):
    prefix, family = m.group(1), m.group(2).lower()
    if family == "darwin":
        norm = "Darwin 24.0.0"
    elif family == "linux":
        norm = "Linux 6.6.0"
    else:
        norm = "Windows 10.0.22631"
    full = prefix + norm
    return None if m.group(0) == full else [(True, full)]


def _extract_cwd_segments(raw_path: str) -> list[tuple[str, str]]:
    """Return [(kind, segment), ...] for non-system directory segments after the home/root prefix."""
    p = raw_path.strip().strip("\"'`")
    p_norm = re.sub(r"\\+", "/", p).rstrip("/")
    if not p_norm:
        return []
    parts = [seg for seg in p_norm.split("/") if seg and not re.fullmatch(r"[A-Za-z]:", seg)]
    if not parts:
        return []
    start = 0
    low0 = parts[0].lower()
    if low0 in ("users", "home") and len(parts) >= 3:
        start = 2
    elif low0 == "mnt" and len(parts) >= 5 and parts[2].lower() == "users":
        start = 4
    elif low0 == "volumes" and len(parts) >= 3:
        start = 2
    elif low0 in ("opt", "srv", "var", "tmp", "private") and len(parts) >= 2:
        start = 1
        while start < len(parts) and parts[start].lower() in SYSTEM_PATH_ROOTS:
            start += 1
    else:
        return []
    tail = parts[start:]
    out: list[tuple[str, str]] = []
    for idx, seg in enumerate(tail):
        if len(seg) < 2 or seg.lower() in SYSTEM_PATH_ROOTS or seg[0] in "$%<{~.":
            continue
        kind = "cwd" if idx == len(tail) - 1 else "cwddir"
        out.append((kind, seg))
    return out


def _date_for_timezone(tz_name: str) -> str | None:
    if tz_name.upper() == "UTC":
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    if zoneinfo is None:
        return None
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        return None
    return datetime.datetime.now(tz).strftime("%Y-%m-%d")


def _shannon_entropy(s: str) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _plausible_secret(value: str) -> bool:
    if not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
        return False
    if value[0] in "$%<{(" or "(" in value or "[" in value:
        return False
    low = value.lower()
    if low.startswith(("process.", "os.", "env.", "config.", "settings.", "self.", "this.", "getenv")):
        return False
    return not any(h in low for h in SECRET_PLACEHOLDER_HINTS)


def _high_entropy_secret(value: str) -> bool:
    """Catch unlabeled random API tokens without touching git SHAs, UUIDs, or camelCase identifiers."""
    low = value.lower()
    if any(h in low for h in SECRET_PLACEHOLDER_HINTS):
        return False
    digits = sum(c.isdigit() for c in value)
    if digits < 4:
        return False
    has_upper = any(c.isupper() for c in value)
    has_lower = any(c.islower() for c in value)
    if not (has_upper and has_lower):   # skips pure hex SHAs, UUIDs, and CONSTANT_NAMES
        return False
    if "_" in value and sum(c == "_" for c in value) >= 3:   # snake_case identifier with numbers
        return False
    return _shannon_entropy(value) >= 4.2


# ------------------------------------------------------------------ authenticated vault persistence

VAULT_MAGIC = b"APV1"


def _seal(key: bytes, payload: bytes) -> bytes:
    enc_key = hmac.new(key, b"vault-enc", hashlib.sha256).digest()
    mac_key = hmac.new(key, b"vault-mac", hashlib.sha256).digest()
    nonce = secrets.token_bytes(16)
    compressed = zlib.compress(payload, level=6)
    stream = hashlib.shake_256(enc_key + nonce).digest(len(compressed))
    cipher = bytes(a ^ b for a, b in zip(compressed, stream))
    tag = hmac.new(mac_key, VAULT_MAGIC + nonce + cipher, hashlib.sha256).digest()
    return VAULT_MAGIC + nonce + tag + cipher


def _unseal(key: bytes, blob: bytes) -> bytes | None:
    if len(blob) < 4 + 16 + 32 or blob[:4] != VAULT_MAGIC:
        return None
    nonce, tag, cipher = blob[4:20], blob[20:52], blob[52:]
    enc_key = hmac.new(key, b"vault-enc", hashlib.sha256).digest()
    mac_key = hmac.new(key, b"vault-mac", hashlib.sha256).digest()
    expected = hmac.new(mac_key, VAULT_MAGIC + nonce + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        return None
    stream = hashlib.shake_256(enc_key + nonce).digest(len(cipher))
    plain = bytes(a ^ b for a, b in zip(cipher, stream))
    try:
        return zlib.decompress(plain)
    except zlib.error:
        return None


# ------------------------------------------------------------------ config / vault

@dataclass
class MaskConfig:
    secrets: bool = True
    entropy_secrets: bool = False       # opt-in or enabled via config.toml to catch unlabeled random tokens
    emails: bool = True
    ips: bool = True
    path_usernames: bool = True
    mask_cwd: bool = False              # mask project directory segments extracted from silent <cwd> / Working directory
    normalize_os_version: bool = True   # normalize silent "OS Version: Darwin 25.6.0" kernel builds
    hostnames: bool = True
    git_remotes: bool = True
    literals: tuple[str, ...] = ()
    keep: frozenset[str] = field(default_factory=frozenset)
    timezone: str | None = "UTC"
    surrogate_style: str = "tagged"     # "tagged" (user1a2b3c4d / SECRET_...) | "natural" (alex_mercer4b / tok_test_...)


class Vault:
    """One per downstream user: original <-> surrogate, learned usernames, and a frozen cache."""

    def __init__(self, key: bytes, cache_chars: int = 32_000_000, style: str = "tagged",
                 persist_path: str | None = None):
        if style not in ("tagged", "natural"):
            raise ValueError('surrogate style must be "tagged" or "natural"')
        self._key = key
        self._style = style
        self._persist_path = persist_path
        self._fwd: dict[tuple[str, str], str] = {}
        self._rev: dict[str, str] = {}
        self._learned: dict[str, None] = {}
        self._learned_rx: re.Pattern | None = None
        self._cwd_segs: dict[str, str] = {}   # lowercase segment -> kind ("cwd" | "cwddir")
        self._cwd_rx: re.Pattern | None = None
        self.generation = 0
        self._dirty = False
        self._lock = threading.RLock()
        self._cache: OrderedDict = OrderedDict()
        self._cache_chars = 0
        self._cache_limit = cache_chars
        if persist_path:
            self._load_disk()

    # -- surrogates
    def _digest(self, kind: str, original: str) -> str:
        return hmac.new(self._key, f"{kind}\x00{original}".encode(), hashlib.sha256).hexdigest()

    def _format(self, kind: str, h: str) -> str:
        if kind == "ip":
            slot = int(h[:8], 16) % (len(IP_POOLS) * 155)
            pool, last = divmod(slot, 155)
            return f"{IP_POOLS[pool]}.{100 + last}"
        if kind == "varfolder":
            return f"{h[:2]}/{h[2:26]}0000gn"
        if kind == "cwd":
            return f"workspace_{h[:6]}"
        if kind == "cwddir":
            return f"dir_{h[:6]}"
        if self._style == "natural":
            v = int(h[:4], 16)
            f_name, l_name = _FIRST[v & 15], _LAST[(v >> 4) & 15]
            w1, w2 = _WORDS[v & 15], _WORDS[(v >> 4) & 15]
            if kind == "email":
                return f"{f_name}.{l_name}{h[4:8]}@example.com"
            if kind == "secret":
                return f"tok_test_{h[:16]}"
            if kind == "user":
                return f"{f_name}_{l_name}{h[4:8]}"
            if kind == "host":
                return f"dev-{_CITIES[v & 7]}-{h[4:8]}"
            if kind == "org":
                return f"{w1}-labs-{h[4:7]}"
            if kind == "repo":
                return f"{w2}-svc-{h[4:7]}"
            return f"{w1.capitalize()}Group_{h[4:8]}"
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
                self._dirty = True
            return s

    def is_surrogate(self, s: str) -> bool:
        return s in self._rev

    def mapping(self, surrogates) -> dict[str, str]:
        with self._lock:
            return {s: self._rev[s] for s in surrogates if s in self._rev}

    # -- learned literals (usernames seen in home-directory paths)
    def _rebuild_learned_rx(self) -> None:
        if not self._learned:
            self._learned_rx = None
            return
        names = sorted(self._learned, key=len, reverse=True)
        # distinctive names (6+ chars) also match when glued to other text, e.g. after a shell
        # strips the backslashes out of C:\Users\name; short ones need word boundaries
        long_alts = "|".join(re.escape(x) for x in names if len(x) >= 6)
        short_alts = "|".join(re.escape(x) for x in names if len(x) < 6)
        parts = ([f"(?:{long_alts})"] if long_alts else []) + \
            ([rf"(?<![A-Za-z0-9_])(?:{short_alts})(?![A-Za-z0-9_])"] if short_alts else [])
        self._learned_rx = re.compile("|".join(parts), re.IGNORECASE)

    def learn(self, value: str) -> None:
        v = value.strip()
        low = v.lower()
        if len(v) < 3 or low in COMMON_WORDS or low in EXCLUDED_PATH_NAMES or v in self._rev:
            return
        with self._lock:
            if low in self._learned:
                return
            self._learned[low] = None
            self._rebuild_learned_rx()
            self.generation += 1
            self._dirty = True

    @property
    def learned_regex(self) -> re.Pattern | None:
        return self._learned_rx

    def _rebuild_cwd_rx(self) -> None:
        if not self._cwd_segs:
            self._cwd_rx = None
            return
        alts = "|".join(re.escape(x) for x in sorted(self._cwd_segs, key=len, reverse=True))
        # Match in path context (preceded by / or \) and followed by any non-word-or-dash char (including ` . ] })
        self._cwd_rx = re.compile(rf"(?<=[/\\])(?P<v>{alts})(?![A-Za-z0-9_\-])", re.IGNORECASE)

    def learn_cwd_segment(self, kind: str, seg: str) -> None:
        s = seg.strip()
        low = s.lower()
        if len(s) < 2 or low in SYSTEM_PATH_ROOTS or s in self._rev:
            return
        with self._lock:
            if self._cwd_segs.get(low) == "cwd" or (low in self._cwd_segs and kind != "cwd"):
                return
            self._cwd_segs[low] = kind
            self._rebuild_cwd_rx()
            self.generation += 1
            self._dirty = True

    def cwd_kind(self, seg: str) -> str:
        return self._cwd_segs.get(seg.lower(), "cwd")

    @property
    def cwd_regex(self) -> re.Pattern | None:
        return self._cwd_rx

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
            self._dirty = True

    # -- encrypted disk persistence (prevents thinking_dropped after gateway restarts)
    def _load_disk(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "rb") as f:
                raw = _unseal(self._key, f.read())
            if raw is None:
                return
            state = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError):
            return
        with self._lock:
            for kind, orig, sur in state.get("fwd") or []:
                self._fwd[(kind, orig)] = sur
                self._rev[sur] = orig
            for low in state.get("learned") or []:
                self._learned[low] = None
            self._rebuild_learned_rx()
            for low, kind in (state.get("cwd_segs") or {}).items():
                self._cwd_segs[low] = kind
            self._rebuild_cwd_rx()
            for orig_text, masked, surs, kinds in state.get("cache") or []:
                if orig_text not in self._cache:
                    self._cache[orig_text] = (masked, tuple(surs), dict(kinds))
                    self._cache_chars += len(orig_text) + len(masked) + 64
            self._dirty = False

    def flush_disk(self, max_persist_chars: int = 4_000_000) -> None:
        if not self._persist_path:
            return
        with self._lock:
            if not self._dirty:
                return
            cache_items = []
            budget = max_persist_chars
            for orig_text, (masked, surs, kinds) in reversed(self._cache.items()):
                cost = len(orig_text) + len(masked)
                if cost > budget:
                    continue
                budget -= cost
                cache_items.append([orig_text, masked, list(surs), kinds])
            cache_items.reverse()
            payload = json.dumps({
                "fwd": [[k, o, s] for (k, o), s in self._fwd.items()],
                "learned": list(self._learned.keys()),
                "cwd_segs": self._cwd_segs,
                "cache": cache_items,
            }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            blob = _seal(self._key, payload)
            self._dirty = False
        os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
        tmp = f"{self._persist_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(blob)
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self._persist_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass


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
        if not isinstance(text, str) or len(text) < 8:
            return
        if self.cfg.path_usernames:
            for rx in PATH_USER_PATTERNS:
                for m in rx.finditer(text):
                    name = m.group("v")
                    if self._username_ok(name, vault):
                        vault.learn(name)
        if self.cfg.mask_cwd and ("cwd>" in text or "root>" in text or "working directory:" in text.lower()):
            for rx in CWD_EXTRACT_PATTERNS:
                for m in rx.finditer(text):
                    for kind, seg in _extract_cwd_segments(m.group(1)):
                        if seg.lower() not in self.keep and not vault.is_surrogate(seg):
                            vault.learn_cwd_segment(kind, seg)

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
            tz_date = _date_for_timezone(cfg.timezone)
            if tz_date:
                segs = _apply(segs, CURRENT_DATE_TAG, lambda m: None if m.group(2) == tz_date else
                              [(True, m.group(1) + tz_date + m.group(3))])
                if "date:" in text.lower():
                    segs = _apply(segs, TODAYS_DATE_LINE, lambda m: None if m.group(2) == tz_date else
                                  [(True, m.group(1) + tz_date)])
        if cfg.normalize_os_version and "os version:" in text.lower():
            segs = _apply(segs, OS_VERSION_LINE, _canonical_os_version)
        if cfg.secrets:
            for rx in SECRET_PATTERNS:
                segs = _apply(segs, rx, lambda m: swap(m, "v" if "v" in m.re.groupindex else 0, "secret"))
            if ASSIGNMENT_HINT.search(text):   # cheap pre-check; ASSIGNMENT is the costliest pattern
                segs = _apply(segs, ASSIGNMENT, lambda m: swap(m, "v", "secret", _plausible_secret))
        if cfg.entropy_secrets and len(text) >= 24:
            segs = _apply(segs, HIGH_ENTROPY_TOKEN, lambda m: swap(m, 0, "secret", _high_entropy_secret))
        if cfg.git_remotes:
            segs = _apply(segs, GIT_REMOTE, git)
        if cfg.emails:
            segs = _apply(segs, EMAIL, email)
        if cfg.path_usernames:
            if "/var/folders/" in text.lower():
                segs = _apply(segs, DARWIN_VAR_FOLDER, lambda m: swap(m, "v", "varfolder"))
            for rx in PATH_USER_PATTERNS:
                segs = _apply(segs, rx, path)
            if vault.learned_regex is not None:
                segs = _apply(segs, vault.learned_regex, lambda m: swap(m, 0, "user"))
        if cfg.mask_cwd and vault.cwd_regex is not None:
            segs = _apply(segs, vault.cwd_regex, lambda m: swap(m, "v", vault.cwd_kind(m.group("v"))))
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
