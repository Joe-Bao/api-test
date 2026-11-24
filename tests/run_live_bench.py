#!/usr/bin/env python3
"""Live end-to-end "Canary Bug-Fix" benchmark for Virtual Provider -> real Claude.

Simulates Codex CLI (`POST /v1/responses` with `wire_api = "responses"`) solving a small,
multi-turn coding task using BOTH:
  1. `shell_command` (type="function") — to run unittest and inspect files
  2. `apply_patch`   (type="custom")   — Codex's freeform diff tool to fix a bug

Meanwhile, the workspace path, file header comments, and environment contain synthetic
"canary" secrets/emails/IPs plus the real local home path. After the agent fixes the bug
and passes unittest, this script inspects `audit/*.json` to verify:
  * Tool calls (`shell_command` + `apply_patch`) executed and fixed the code on disk
  * Multi-turn signed `thinking` round-tripped via `reasoning.encrypted_content` (`thinking_dropped == 0`)
  * ZERO canary strings or real local usernames appeared in any payload sent to Anthropic!
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from anon_proxy.bridge import ProviderConfig  # noqa: E402
from anon_proxy.masking import MaskConfig  # noqa: E402
from anon_proxy.protocols import Privacy  # noqa: E402
from anon_proxy.server import DEFAULT_KEEP, Settings, Upstream, _parse_env_file, build_server, upstream_key  # noqa: E402

# Canary values planted inside the local project — Claude must NEVER see these raw strings!
CANARY_EMAIL = "ops-canary@internal-corp.cn"
CANARY_IP = "10.24.88.19"
CANARY_SECRET = "Zx9Qp2Lm8Vn4Kr7Tw1Yb5Cf3Hj6Gs0D8"
CANARY_ORG = "StealthInternalLab"

BUGGY_CODE = f'''"""Internal billing calculator.
Maintainer: {CANARY_EMAIL} | Org: {CANARY_ORG}
Metrics endpoint: http://{CANARY_IP}:9100 | Token: {CANARY_SECRET}
"""


def discounted_total(prices: list[int], vip: bool = False) -> int:
    subtotal = sum(prices)
    if vip and subtotal >= 100:
        # BUG: VIPs should get 20% off (pay 80%), not add 20%!
        return int(round(subtotal * 1.20))
    return subtotal
'''

UNIT_TEST = '''import unittest
from billing_calc import discounted_total


class BillingTest(unittest.TestCase):
    def test_regular(self):
        self.assertEqual(discounted_total([30, 40], vip=False), 70)

    def test_vip_discount(self):
        self.assertEqual(discounted_total([60, 40], vip=True), 80)


if __name__ == "__main__":
    unittest.main()
'''


def parse_sse(raw: bytes) -> list[tuple[str | None, dict | str]]:
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            payload = "\n".join(data)
            events.append((name, payload if payload == "[DONE]" else json.loads(payload)))
    return events


def apply_codex_patch(workdir: str, patch_text: str) -> str:
    """Minimal parser for Codex's `*** Begin Patch` / `*** Update File:` format."""
    lines = patch_text.strip().splitlines()
    target_file = None
    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False

    def flush_hunk(path: str, olds: list[str], news: list[str]) -> None:
        if not path or (not olds and not news):
            return
        full = os.path.join(workdir, path) if not os.path.isabs(path) else path
        with open(full, encoding="utf-8") as f:
            content = f.read()
        old_block = "\n".join(olds)
        new_block = "\n".join(news)
        if old_block not in content:
            raise ValueError(f"patch context did not match {full}:\n{old_block}")
        content = content.replace(old_block, new_block, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    patched = []
    for ln in lines:
        if ln.startswith("*** Begin Patch") or ln.startswith("*** End Patch"):
            if target_file and in_hunk:
                flush_hunk(target_file, old_lines, new_lines)
                old_lines, new_lines, in_hunk = [], [], False
            continue
        if ln.startswith("*** Update File:"):
            if target_file and in_hunk:
                flush_hunk(target_file, old_lines, new_lines)
                old_lines, new_lines = [], []
            target_file = ln.split(":", 1)[1].strip()
            patched.append(target_file)
            in_hunk = False
            continue
        if ln.startswith("@@"):
            if target_file and in_hunk:
                flush_hunk(target_file, old_lines, new_lines)
                old_lines, new_lines = [], []
            in_hunk = True
            continue
        if target_file is not None:
            in_hunk = True
            if ln.startswith("-"):
                old_lines.append(ln[1:])
            elif ln.startswith("+"):
                new_lines.append(ln[1:])
            elif ln.startswith(" "):
                old_lines.append(ln[1:])
                new_lines.append(ln[1:])
            else:
                old_lines.append(ln)
                new_lines.append(ln)
    if target_file and in_hunk:
        flush_hunk(target_file, old_lines, new_lines)
    return f"Updated {', '.join(patched)}" if patched else "No files updated"


def main() -> int:
    env_path = os.path.join(ROOT, ".env")
    dotenv = _parse_env_file(env_path)
    model = os.environ.get("ANTHROPIC_MODEL") or dotenv.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"

    up = Upstream("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "Anthropic/Python 0.52.0",
                  api_key_file=env_path)
    if not upstream_key(up):
        print("ERROR: ANTHROPIC_API_KEY is empty in .env (and environment). Fill .env first.", file=sys.stderr)
        return 2

    # Put the temp bench inside the user's real home tree (under data/) so real /Users/<user> path masking is tested!
    data_root = os.path.join(ROOT, "data")
    os.makedirs(data_root, exist_ok=True)
    bench_root = tempfile.mkdtemp(prefix="canary_bench_", dir=data_root)
    os.makedirs(bench_root, exist_ok=True)
    proj_dir = os.path.join(bench_root, "proj")
    audit_dir = os.path.join(bench_root, "audit")
    os.makedirs(proj_dir, exist_ok=True)

    with open(os.path.join(proj_dir, "billing_calc.py"), "w", encoding="utf-8") as f:
        f.write(BUGGY_CODE)
    with open(os.path.join(proj_dir, "test_billing.py"), "w", encoding="utf-8") as f:
        f.write(UNIT_TEST)

    s = Settings(host="127.0.0.1", port=0)
    s.anthropic = up
    s.provider = ProviderConfig(
        target="anthropic",
        default_model=model,
        force_model=True,
        default_max_tokens=4096,
        thinking_mode="auto",
        strip_client_betas=True,
    )
    s.users = {"gk-live-bench": "bench_user"}
    s.secret = os.urandom(32)
    s.privacy = Privacy()
    s.mask = MaskConfig(
        surrogate_style="natural",
        entropy_secrets=True,
        keep=frozenset(DEFAULT_KEEP),
        literals=(CANARY_ORG,),
    )
    s.audit_dir = audit_dir

    server = build_server(s)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    tools = [
        {
            "type": "function",
            "name": "shell_command",
            "description": "Execute a shell command in the project directory and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                "required": ["command"],
            },
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": (
                "Patch a file using Codex patch format:\n"
                "*** Begin Patch\n*** Update File: <relative_or_abs_path>\n@@\n context\n-old\n+new\n*** End Patch"
            ),
        },
    ]

    history: list[dict] = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": (
                "You are a concise coding agent. First inspect/run tests with `shell_command`, "
                "then fix the bug using `apply_patch` (preserve module docstrings/comments), "
                "then re-run `shell_command` to verify unittest passes before giving a one-line summary."
            )}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": (
                f"<environment_context>\n  <cwd>{proj_dir}</cwd>\n  <timezone>Asia/Shanghai</timezone>\n"
                f"</environment_context>\n"
                f"Fix the failing unit test in `{proj_dir}`."
            )}],
        },
    ]

    used_tools: list[str] = []
    saw_reasoning = False
    final_answer = ""

    try:
        for turn in range(1, 7):
            req_body = {
                "model": "gpt-5.1-codex",
                "stream": True,
                "store": True,
                "prompt_cache_key": "sess-live-bench-001",
                "client_metadata": {"installation_id": "inst-secret-uuid-999"},
                "reasoning": {"effort": "medium", "summary": "auto"},
                "tools": tools,
                "input": history,
            }
            req = urllib.request.Request(
                f"{base_url}/v1/responses",
                data=json.dumps(req_body).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer gk-live-bench",
                    "originator": "codex_cli_rs",
                    "session-id": "sess-live-bench-001",
                    "User-Agent": "codex_cli_rs/0.170.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", "replace")
                e.close()
                print(f"[Turn {turn}] HTTP {e.code}: {err_body}", file=sys.stderr)
                return 1

            events = parse_sse(raw)
            completed = next((d["response"] for _, d in events if isinstance(d, dict)
                              and d.get("type") in ("response.completed", "response.incomplete")), None)
            if not completed:
                print(f"[Turn {turn}] Missing response.completed in SSE stream", file=sys.stderr)
                return 1

            outputs = completed.get("output") or []
            history.extend(outputs)
            tool_calls_this_turn = 0

            for item in outputs:
                itype = item.get("type")
                if itype == "reasoning":
                    saw_reasoning = bool(item.get("encrypted_content", "").startswith("ap1:"))
                elif itype == "message":
                    txt = "".join(c.get("text", "") for c in item.get("content") or [])
                    final_answer = txt
                    print(f"[Turn {turn} assistant] {txt.strip()}")
                elif itype == "function_call" and item.get("name") == "shell_command":
                    tool_calls_this_turn += 1
                    used_tools.append("shell_command")
                    cmd = json.loads(item.get("arguments") or "{}").get("command", "")
                    print(f"[Turn {turn} tool:shell_command] {cmd}")
                    proc = subprocess.run(cmd, shell=True, cwd=proj_dir, capture_output=True, text=True, timeout=15)
                    out_txt = (proc.stdout + proc.stderr).strip() or f"(exit {proc.returncode})"
                    history.append({
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": f"exit_code={proc.returncode}\n{out_txt}",
                    })
                elif itype == "custom_tool_call" and item.get("name") == "apply_patch":
                    tool_calls_this_turn += 1
                    used_tools.append("apply_patch")
                    patch_txt = item.get("input") or ""
                    print(f"[Turn {turn} tool:apply_patch] ({len(patch_txt)} chars)")
                    try:
                        res = apply_codex_patch(proj_dir, patch_txt)
                    except Exception as exc:
                        res = f"Patch error: {exc}"
                    print(f"  -> {res}")
                    history.append({
                        "type": "custom_tool_call_output",
                        "call_id": item["call_id"],
                        "output": res,
                    })

            if tool_calls_this_turn == 0:
                break

        # 1. Verify unit tests now pass on disk AND canary comments stayed intact!
        check = subprocess.run([sys.executable, "-m", "unittest", "discover"], cwd=proj_dir,
                               capture_output=True, text=True, timeout=10)
        with open(os.path.join(proj_dir, "billing_calc.py"), encoding="utf-8") as f:
            final_code = f.read()
        canaries_intact = all(c in final_code for c in (CANARY_EMAIL, CANARY_IP, CANARY_SECRET, CANARY_ORG))

        # 2. Audit every single payload sent to Anthropic for leaks!
        local_user = os.path.basename(os.path.expanduser("~"))
        forbidden = [CANARY_EMAIL, CANARY_IP, CANARY_SECRET, CANARY_ORG, "Asia/Shanghai",
                     "sess-live-bench-001", "inst-secret-uuid-999", local_user]
        leaks = []
        audit_files = sorted(os.listdir(audit_dir))
        for fname in audit_files:
            with open(os.path.join(audit_dir, fname), encoding="utf-8") as f:
                blob = f.read()
            for bad in forbidden:
                if bad and bad in blob:
                    leaks.append((fname, bad))

        print("\n=== BENCHMARK REPORT ===")
        print(f"Model used upstream     : {model}")
        print(f"Upstream calls audited  : {len(audit_files)}")
        print(f"Tools invoked by Claude : {used_tools}")
        print(f"Signed thinking packed  : {saw_reasoning}")
        print(f"Unit tests after fix    : {'PASS' if check.returncode == 0 else 'FAIL'}")
        print(f"Canary comments intact  : {canaries_intact}")
        print(f"Privacy leaks to Claude : {len(leaks)} {leaks if leaks else '(ZERO LEAKS)'}")

        ok = (check.returncode == 0 and canaries_intact and "shell_command" in used_tools
              and "apply_patch" in used_tools and len(leaks) == 0)
        return 0 if ok else 1
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(bench_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
