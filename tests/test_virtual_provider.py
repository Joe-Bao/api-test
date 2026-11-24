"""Virtual Provider end-to-end tests:
Codex (/v1/responses, /v1/responses/compact), OpenAI (/v1/chat/completions), and Anthropic (/v1/messages)
all enter anon_proxy and get washed into one uniform, clean Anthropic `/v1/messages` request upstream,
then translated and restored back into each client's native wire protocol.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import fake_upstream  # noqa: E402
from anon_proxy.bridge import ProviderConfig  # noqa: E402
from anon_proxy.masking import MaskConfig  # noqa: E402
from anon_proxy.protocols import Privacy  # noqa: E402
from anon_proxy.server import Settings, Upstream, build_server, load_settings  # noqa: E402

USERNAME = "falcon42"
EMAIL = "founder@stealth-ai.io"
SECRET = "K9mP2xQ8vL4nR7wY1zB5cF3hJ6tG0sD8"
CWD = f"/Users/{USERNAME}/stealth-repo"
INSTALL_ID = "0f693cf0-8943-474d-a18b-716746cbd36f"
SESSION = "01a209e5-c04a-7d13-a6ce-566e4090e0f8"
DEVICE_ID = "a33a6db2e04597e6253f8b4e7c617602b7693cd7bc2bfaf33dd90fe2e0060baa"
FORBIDDEN = (USERNAME, EMAIL, SECRET, INSTALL_ID, SESSION, DEVICE_ID, "Asia/Shanghai", "StealthCorp")


def parse_sse(raw: bytes):
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


class VirtualProviderE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.log_path = os.path.join(cls.tmp, "vp_upstream.jsonl")
        fake_upstream.LOG_PATH = cls.log_path
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), fake_upstream.Handler)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        up = f"http://127.0.0.1:{cls.upstream.server_address[1]}/v1"

        os.environ["VP_ANTHROPIC_KEY"] = "owner-anthropic-key-789"
        s = Settings(host="127.0.0.1", port=0)
        s.anthropic = Upstream(up, "VP_ANTHROPIC_KEY", "Anthropic/Python 0.52.0")
        s.provider = ProviderConfig(
            target="anthropic",
            default_model="claude-opus-5-5",
            force_model=True,
            default_max_tokens=16384,
            thinking_mode="auto",
            strip_client_betas=True,
        )
        s.users = {"gk-codex": "alice"}
        s.secret = b"v" * 32
        s.privacy = Privacy()
        s.mask = MaskConfig(
            surrogate_style="natural",
            entropy_secrets=True,
            mask_cwd=True,
            timezone="Australia/Sydney",
            keep=frozenset({"git@github.com"}),
            literals=("StealthCorp",),
        )
        s.vault_dir = os.path.join(cls.tmp, "vaults")
        cls.proxy = build_server(s)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.proxy.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def request(self, method: str, path: str, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        hdrs = dict(headers or {})
        if data is not None:
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            try:
                return e.code, e.read()
            finally:
                e.close()

    def upstream_entries(self):
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def assertNoLeak(self, entry: dict):
        blob = json.dumps(entry, ensure_ascii=False)
        for bad in FORBIDDEN:
            self.assertNotIn(bad, blob, f"upstream leaked {bad!r}")

    def test_all_clients_wash_into_identical_claude_envelope(self):
        """Codex (/v1/responses), OpenAI SDK (/v1/chat/completions), and Claude SDK (/v1/messages)
        must all hit POST /v1/messages with identical headers, model, max_tokens, and schema."""
        prompt = f"Check {CWD}/app.py for {EMAIL} at StealthCorp with key {SECRET}"
        tool_schema = {"type": "object", "properties": {"command": {"type": "string"}}}

        # 1. Codex CLI -> /v1/responses
        st1, raw1 = self.request("POST", "/v1/responses", {
            "model": "gpt-5.1-codex",
            "stream": True,
            "store": True,
            "prompt_cache_key": SESSION,
            "client_metadata": {"installation_id": INSTALL_ID},
            "instructions": "You are a helpful coding assistant.",
            "tools": [{"type": "function", "name": "shell_command", "description": "run", "parameters": tool_schema}],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        }, {
            "Authorization": "Bearer gk-codex",
            "originator": "codex_cli_rs",
            "session-id": SESSION,
            "User-Agent": "codex_cli_rs/0.170.0",
        })
        self.assertEqual(st1, 200)
        up_codex = self.upstream_entries()[-1]

        # 2. Open-source client -> /v1/chat/completions
        st2, raw2 = self.request("POST", "/v1/chat/completions", {
            "model": "gpt-4o",
            "stream": True,
            "user": "internal-user-1",
            "tools": [{"type": "function", "function": {"name": "shell_command", "description": "run",
                                                        "parameters": tool_schema}}],
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        }, {
            "Authorization": "Bearer gk-codex",
            "User-Agent": "OpenCode/1.2.0",
        })
        self.assertEqual(st2, 200)
        up_chat = self.upstream_entries()[-1]

        # 3. Anthropic client -> /v1/messages (with extra client betas & billing header)
        st3, raw3 = self.request("POST", "/v1/messages?beta=true", {
            "model": "claude-sonnet-4-5",
            "stream": True,
            "metadata": {"user_id": DEVICE_ID},
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.312; cc_entrypoint=cli;"},
                {"type": "text", "text": "You are a helpful coding assistant."},
            ],
            "tools": [{"name": "shell_command", "description": "run", "input_schema": tool_schema}],
            "messages": [{"role": "user", "content": prompt}],
        }, {
            "x-api-key": "gk-codex",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14",
            "X-Claude-Code-Session-Id": SESSION,
            "User-Agent": "claude-cli/2.1.312 (external, cli)",
        })
        self.assertEqual(st3, 200)
        up_anth = self.upstream_entries()[-1]

        # Verify all 3 upstream requests look 100% identical to Anthropic!
        for entry in (up_codex, up_chat, up_anth):
            self.assertNoLeak(entry)
            self.assertEqual(entry["path"], "/v1/messages")
            hdrs = {k.lower(): v for k, v in entry["headers"]}
            self.assertEqual(hdrs["user-agent"], "Anthropic/Python 0.52.0")
            self.assertEqual(hdrs["anthropic-version"], "2023-06-01")
            self.assertNotIn("anthropic-beta", hdrs)
            self.assertNotIn("session-id", hdrs)
            self.assertNotIn("x-claude-code-session-id", hdrs)

        self.assertEqual(up_codex["json"], up_chat["json"])
        self.assertEqual(up_codex["json"], up_anth["json"])

        # Verify each downstream client received its own native SSE format with originals restored!
        codex_text = "".join(d["delta"] for _, d in parse_sse(raw1) if d["type"] == "response.output_text.delta")
        chat_text = "".join(
            d["choices"][0]["delta"].get("content") or "" for _, d in parse_sse(raw2) if d != "[DONE]"
        )
        anth_text = "".join(
            d["delta"]["text"] for _, d in parse_sse(raw3)
            if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta"
        )
        expected = f"echo: {prompt}"
        self.assertEqual(codex_text, expected)
        self.assertEqual(chat_text, expected)
        self.assertEqual(anth_text, expected)

    def test_codex_signed_thinking_and_tool_round_trip(self):
        """Codex multi-turn with adaptive reasoning + function_call + custom apply_patch tool:
        signed Claude thinking block travels inside reasoning.encrypted_content and returns byte-identical."""
        tools = [
            {"type": "function", "name": "shell_command", "description": "run shell",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
            {"type": "custom", "name": "apply_patch", "description": "patch files"},
        ]
        # Turn 1: Codex asks with reasoning + triggers shell_command tool call
        prompt1 = f"WITH_THINKING CALL_TOOL: ls {CWD} # <timezone>Asia/Shanghai</timezone>"
        st, raw1 = self.request("POST", "/v1/responses", {
            "model": "anon-coder",
            "stream": True,
            "reasoning": {"effort": "high", "summary": "auto"},
            "tools": tools,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt1}]}],
        }, {"Authorization": "Bearer gk-codex"})
        self.assertEqual(st, 200)
        up1 = self.upstream_entries()[-1]
        self.assertNoLeak(up1)
        self.assertEqual(up1["json"]["thinking"],
                         {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})

        ev1 = parse_sse(raw1)
        completed1 = next(d["response"] for _, d in ev1 if d["type"] == "response.completed")
        rs_item = next(it for it in completed1["output"] if it["type"] == "reasoning")
        fc_item = next(it for it in completed1["output"] if it["type"] == "function_call")
        self.assertTrue(rs_item["encrypted_content"].startswith("ap1:"))
        # Restored summary and restored tool arguments on client side:
        self.assertIn(CWD, rs_item["summary"][0]["text"])
        self.assertIn(CWD, json.loads(fc_item["arguments"])["command"])

        # Turn 2: Codex sends back reasoning (with encrypted_content) + function_call + output, now triggers apply_patch
        prompt2 = f"WITH_THINKING CALL_PATCH: {CWD}/secret.py"
        st2, raw2 = self.request("POST", "/v1/responses", {
            "model": "anon-coder",
            "stream": True,
            "reasoning": {"effort": "high"},
            "tools": tools,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt1}]},
                rs_item,
                fc_item,
                {"type": "function_call_output", "call_id": fc_item["call_id"], "output": f"found {CWD}/secret.py"},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt2}]},
            ],
        }, {"Authorization": "Bearer gk-codex"})
        self.assertEqual(st2, 200)
        up2 = self.upstream_entries()[-1]
        self.assertNoLeak(up2)

        # Verify Turn 1's user message stayed byte-identical and assistant message starts with intact signed thinking!
        self.assertEqual(up1["json"]["messages"][0], up2["json"]["messages"][0])
        asst_blocks = up2["json"]["messages"][1]["content"]
        self.assertEqual(asst_blocks[0]["type"], "thinking")
        self.assertEqual(asst_blocks[0]["signature"], "sig_fake_opus_55==")
        self.assertNotIn(USERNAME, asst_blocks[0]["thinking"])
        self.assertEqual(asst_blocks[1]["type"], "tool_use")

        # Verify custom_tool_call (apply_patch) was unwrapped & restored cleanly for Codex!
        ev2 = parse_sse(raw2)
        completed2 = next(d["response"] for _, d in ev2 if d["type"] == "response.completed")
        patch_item = next(it for it in completed2["output"] if it["type"] == "custom_tool_call")
        self.assertEqual(patch_item["name"], "apply_patch")
        self.assertIn(f"*** Update File: {CWD}/secret.py", patch_item["input"])

    def test_codex_compaction_endpoint_and_non_streaming(self):
        """Codex `/v1/responses/compact` summarizes via Claude and replays cleanly on the next turn."""
        st, raw = self.request("POST", "/v1/responses/compact", {
            "model": "gpt-5.1",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": f"Working in {CWD} for {EMAIL}"}]},
            ],
        }, {"Authorization": "Bearer gk-codex"})
        self.assertEqual(st, 200)
        comp = json.loads(raw)
        self.assertEqual(comp["object"], "response.compaction")
        cmp_item = comp["output"][0]
        self.assertEqual(cmp_item["type"], "compaction")
        self.assertTrue(cmp_item["encrypted_content"].startswith("ap1c:"))

        # Feed compaction output into next non-streaming /v1/responses request
        st2, raw2 = self.request("POST", "/v1/responses", {
            "model": "gpt-5.1",
            "stream": False,
            "input": [
                cmp_item,
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": f"Continue editing {CWD}/main.py"}]},
            ],
        }, {"Authorization": "Bearer gk-codex"})
        self.assertEqual(st2, 200)
        self.assertNoLeak(self.upstream_entries()[-1])
        resp2 = json.loads(raw2)
        self.assertEqual(resp2["object"], "response")
        self.assertIn(f"{CWD}/main.py", resp2["output"][0]["content"][0]["text"])

    def test_models_endpoint_and_example_config_loader(self):
        st, raw = self.request("GET", "/v1/models", headers={"Authorization": "Bearer gk-codex"})
        self.assertEqual(st, 200)
        models = json.loads(raw)
        self.assertEqual(models["object"], "list")
        self.assertEqual(models["data"][0]["object"], "model")

        # Ensure config.example.toml parses cleanly on Python 3.9+ without polluting repo root
        import shutil
        tmp_cfg = os.path.join(self.tmp, "config.toml")
        shutil.copyfile(os.path.join(ROOT, "config.example.toml"), tmp_cfg)
        cfg = load_settings(tmp_cfg)
        self.assertEqual(cfg.provider.target, "anthropic")
        self.assertEqual(cfg.mask.surrogate_style, "natural")
        self.assertEqual(cfg.mask.timezone, "Australia/Sydney")
        self.assertTrue(cfg.mask.entropy_secrets)

    def test_codex_system_branding_and_bundled_skills_scrubbed(self):
        codex_dev_1 = (
            "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. "
            "Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.\n\n"
            "Within this context, Codex refers to the open-source agentic coding interface "
            "(not the old Codex language model built by OpenAI)."
        )
        codex_skills_only_system = (
            "<skills_instructions>\n## Skills\n### Skill roots\n"
            "- `r0` = `/private/var/folders/h1/0klcwbwj08l_vxxkjp1_0c8c0000gn/T/codex_home/skills/.system`\n"
            "### Available skills\n"
            "- openai-docs: ChatGPT Work Q&A (file: r0/openai-docs/SKILL.md)\n"
            "- imagegen: Generate images (file: r0/imagegen/SKILL.md)\n"
            "</skills_instructions>"
        )
        st, _ = self.request("POST", "/v1/responses", {
            "model": "gpt-5.1-codex",
            "stream": False,
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": codex_dev_1}]},
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": codex_skills_only_system}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
        }, {"Authorization": "Bearer gk-codex"})
        self.assertEqual(st, 200)
        sent_sys = json.dumps(self.upstream_entries()[-1]["json"].get("system") or [])
        for leaked in ("Codex CLI", "OpenAI", "openai-docs", "ChatGPT Work", "0klcwbwj08l_vxxkjp1_0c8c0000gn"):
            self.assertNotIn(leaked, sent_sys)
        self.assertIn("terminal-based coding assistant", sent_sys)

    def test_claude_code_silent_git_status_and_cwd_scrubbed(self):
        cc_reminder = (
            "<system-reminder>\n"
            "As you answer the user's questions, you can use the following context:\n"
            "# gitStatus\n"
            "gitStatus: This is the git status at the start of the conversation. Note that this status is a snapshot.\n"
            "Current branch: feat/jira-secret-999\n"
            "Recent commits:\n"
            "252e99f internal SSO fix for CEO\n"
            "</system-reminder>"
        )
        cc_env = (
            "<env>\n"
            f"Working directory: {CWD}\n"
            "OS Version: Darwin 25.6.0\n"
            "Today's date: 1999-12-31\n"
            "</env>"
        )
        st, raw = self.request("POST", "/v1/messages", {
            "model": "claude-opus-5-5",
            "max_tokens": 100,
            "stream": False,
            "system": [{"type": "text", "text": cc_env}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": cc_reminder},
                {"type": "text", "text": f"Inspect {CWD}/handler.py"},
            ]}],
        }, {"x-api-key": "gk-codex", "anthropic-version": "2023-06-01"})
        self.assertEqual(st, 200)
        up = self.upstream_entries()[-1]["json"]
        blob = json.dumps(up, ensure_ascii=False)
        for hidden in ("jira-secret-999", "internal SSO fix", "stealth-repo", "25.6.0", "1999-12-31", USERNAME):
            self.assertNotIn(hidden, blob)
        self.assertIn("OS Version: Darwin 24.0.0", blob)
        self.assertIn("workspace_", blob)
        # Response echoes back with stealth-repo and username fully restored!
        resp = json.loads(raw)
        self.assertIn(f"{CWD}/handler.py", resp["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
