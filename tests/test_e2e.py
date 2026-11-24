"""End to end: client -> anon_proxy -> fake upstream, all in-process on random ports."""
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
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import fake_upstream  # noqa: E402
from anon_proxy.masking import MaskConfig  # noqa: E402
from anon_proxy.protocols import Privacy  # noqa: E402
from anon_proxy.server import Settings, Upstream, build_server  # noqa: E402

USERNAME = "kestrel77"
EMAIL = "alice.w@corp.cn"
SECRET = "ghp_" + "Ab3" * 12
CWD = rf"C:\Users\{USERNAME}\work\proj"
INSTALL_ID = "0f693cf0-8943-474d-a18b-716746cbd36f"
SESSION = "01a209e5-c04a-7d13-a6ce-566e4090e0f8"
DEVICE_ID = "a33a6db2e04597e6253f8b4e7c617602b7693cd7bc2bfaf33dd90fe2e0060baa"
FORBIDDEN = (USERNAME, EMAIL, SECRET, INSTALL_ID, SESSION, DEVICE_ID, "Europe/Berlin", "ACME Corp")


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


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        fake_upstream.LOG_PATH = os.path.join(cls.tmp, "upstream.jsonl")
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), fake_upstream.Handler)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        up = f"http://127.0.0.1:{cls.upstream.server_address[1]}/v1"
        os.environ["E2E_OWNER_OPENAI"] = "owner-openai-key-123"
        os.environ["E2E_OWNER_ANTHROPIC"] = "owner-anthropic-key-456"
        s = Settings(host="127.0.0.1", port=0)
        s.openai = Upstream(up, "E2E_OWNER_OPENAI", "uniform-agent/1.0")
        s.anthropic = Upstream(up, "E2E_OWNER_ANTHROPIC", "uniform-agent/1.0")
        s.users = {"gk-alice": "alice", "gk-bob": "bob", "gk-carol": "carol"}
        s.secret = b"s" * 32
        s.privacy = Privacy()
        s.mask = MaskConfig(keep=frozenset({"git@github.com"}), literals=("ACME Corp",))
        s.audit_dir = os.path.join(cls.tmp, "audit")
        cls.proxy = build_server(s)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.proxy.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.upstream.shutdown()

    # -- helpers
    def post(self, path, body, headers):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def upstream_entries(self):
        with open(fake_upstream.LOG_PATH, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def assertClean(self, entry):
        blob = json.dumps(entry, ensure_ascii=False)
        for bad in FORBIDDEN:
            self.assertNotIn(bad, blob, f"upstream saw {bad!r}")

    def codex_request(self, prompt, extra_input=()):
        headers = {"Authorization": "Bearer gk-alice", "session-id": SESSION, "thread-id": SESSION,
                   "originator": "codex_cli_rs", "x-codex-installation-id": INSTALL_ID,
                   "x-codex-turn-metadata": json.dumps({"installation_id": INSTALL_ID, "session_id": SESSION}),
                   "User-Agent": "codex_cli_rs/0.170.0 (Windows 10.0.26200; x86_64) WindowsTerminal"}
        body = {
            "model": "gpt-5.1", "stream": True, "store": True, "prompt_cache_key": SESSION,
            "client_metadata": {"x-codex-installation-id": INSTALL_ID, "session_id": SESSION},
            "tools": [{"type": "function", "name": "shell_command", "description": "run a command",
                       "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}],
            "input": [
                {"type": "message", "role": "developer", "id": f"msg_{SESSION}",
                 "content": [{"type": "input_text", "text": f"skills root C:/Users/{USERNAME}/.codex/skills"}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": f"<environment_context>\n  <cwd>{CWD}</cwd>\n"
                                                             "  <timezone>Europe/Berlin</timezone>\n"
                                                             "</environment_context>"}]},
                *extra_input,
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
        }
        return self.post("/v1/responses", body, headers)

    # -- tests
    def test_codex_responses_stream_is_scrubbed_and_restored(self):
        prompt = f"I am {EMAIL} at ACME Corp, token {SECRET}, project in {CWD}"
        status, raw = self.codex_request(prompt)
        self.assertEqual(status, 200)
        entry = self.upstream_entries()[-1]
        self.assertClean(entry)
        headers = {k.lower(): v for k, v in entry["headers"]}
        for h in ("session-id", "thread-id", "originator", "x-codex-installation-id", "x-codex-turn-metadata"):
            self.assertNotIn(h, headers)
        self.assertEqual(headers["user-agent"], "uniform-agent/1.0")
        self.assertEqual(headers["authorization"], f"Bearer <redacted len={len('owner-openai-key-123')}>")
        body = entry["json"]
        self.assertNotIn("client_metadata", body)
        self.assertIs(body["store"], False)
        self.assertNotEqual(body["prompt_cache_key"], SESSION)
        self.assertIn("<timezone>UTC</timezone>", json.dumps(body))
        self.assertTrue(body["input"][0]["id"].startswith("msg_"))
        # client side: deltas and the final item both carry the originals again
        events = parse_sse(raw)
        text = "".join(d["delta"] for n, d in events if d["type"] == "response.output_text.delta")
        self.assertEqual(text, "echo: " + prompt)
        done = next(d for n, d in events if d["type"] == "response.output_item.done")
        self.assertEqual(done["item"]["content"][0]["text"], "echo: " + prompt)

    def test_codex_tool_call_round_trip(self):
        status, raw = self.codex_request(f"CALL_TOOL: dir {CWD}")
        self.assertEqual(status, 200)
        self.assertClean(self.upstream_entries()[-1])
        events = parse_sse(raw)
        args = "".join(d["delta"] for n, d in events if d["type"] == "response.function_call_arguments.delta")
        self.assertEqual(json.loads(args)["command"], f"echo dir {CWD}")
        item = next(d for n, d in events if d["type"] == "response.output_item.done")["item"]
        self.assertEqual(json.loads(item["arguments"])["command"], f"echo dir {CWD}")
        # next turn: the tool output (with the real path) goes back and must be masked again
        follow = [item, {"type": "function_call_output", "call_id": item["call_id"], "output": f"dir {CWD}"}]
        status, raw = self.codex_request("continue", extra_input=follow)
        self.assertEqual(status, 200)

    def test_tool_result_turn(self):
        item = {"type": "function_call", "call_id": "call_1", "name": "shell_command",
                "arguments": json.dumps({"command": f"type {CWD}\\notes.txt"})}
        out = {"type": "function_call_output", "call_id": "call_1", "output": f"owner: {EMAIL}"}
        headers = {"Authorization": "Bearer gk-alice"}
        body = {"model": "m", "stream": True, "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read notes"}]}, item, out]}
        status, raw = self.post("/v1/responses", body, headers)
        self.assertEqual(status, 200)
        self.assertClean(self.upstream_entries()[-1])
        text = "".join(d["delta"] for n, d in parse_sse(raw) if d["type"] == "response.output_text.delta")
        self.assertEqual(text, f"tool said: owner: {EMAIL}")

    def test_chat_completions_stream_and_plain(self):
        msgs = [{"role": "user", "name": "alice", "content": f"my mail is {EMAIL}"}]
        status, raw = self.post("/v1/chat/completions", {"model": "m", "stream": True, "messages": msgs,
                                                         "user": "alice-internal-id"},
                                {"Authorization": "Bearer gk-alice"})
        self.assertEqual(status, 200)
        entry = self.upstream_entries()[-1]
        self.assertClean(entry)
        self.assertNotIn("user", entry["json"])
        self.assertNotIn("name", entry["json"]["messages"][0])
        chunks = [d for n, d in parse_sse(raw) if d != "[DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        self.assertEqual(text, f"echo: my mail is {EMAIL}")
        status, raw = self.post("/v1/chat/completions", {"model": "m", "messages": msgs},
                                {"Authorization": "Bearer gk-alice"})
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], f"echo: my mail is {EMAIL}")

    def test_claude_code_messages_stream(self):
        thinking = {"type": "thinking", "thinking": f"signed text mentioning {USERNAME}", "signature": "sig=="}
        body = {
            "model": "claude-x", "stream": True, "max_tokens": 100,
            "thinking": {"type": "adaptive", "display": "omitted"},
            "metadata": {"user_id": json.dumps({"device_id": DEVICE_ID, "session_id": SESSION})},
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.312; cc_entrypoint=cli;"},
                       {"type": "text", "text": f"Primary working directory: {CWD}"}],
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [thinking, {"type": "text", "text": "hello"}]},
                {"role": "user", "content": [{"type": "text", "text": f"CALL_TOOL: type {CWD}\\a.txt"}]},
            ],
        }
        headers = {"x-api-key": "gk-bob", "anthropic-version": "2023-06-01",
                   "anthropic-beta": "claude-code-20250219", "X-Claude-Code-Session-Id": SESSION,
                   "X-Stainless-OS": "Windows", "x-app": "cli", "User-Agent": "claude-cli/2.1.312 (external, cli)"}
        status, raw = self.post("/v1/messages?beta=true", body, headers)
        self.assertEqual(status, 200)
        entry = self.upstream_entries()[-1]
        sent = entry["json"]
        # the signed thinking block is the only place an original may remain, and it must be untouched
        self.assertEqual(sent["messages"][1]["content"][0], thinking)
        sent["messages"][1]["content"][0] = {}
        self.assertClean(entry)
        self.assertNotIn("metadata", sent)
        self.assertFalse(any("billing" in b["text"] for b in sent["system"]))
        self.assertEqual(sent["thinking"]["block_binding"], {"prefix_mismatch_behavior": "drop_block"})
        h = {k.lower(): v for k, v in entry["headers"]}
        self.assertEqual(h["anthropic-beta"], "claude-code-20250219,thinking-binding-controls-2026-08-01")
        for gone in ("x-claude-code-session-id", "x-stainless-os", "x-app"):
            self.assertNotIn(gone, h)
        self.assertEqual(entry["path"], "/v1/messages?beta=true")
        events = parse_sse(raw)
        partial = "".join(d["delta"]["partial_json"] for n, d in events
                          if d.get("type") == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(partial)["command"], f"echo type {CWD}\\a.txt")

    def test_history_prefix_stays_byte_identical(self):
        headers = {"x-api-key": "gk-carol", "anthropic-version": "2023-06-01"}
        first = {"role": "user", "content": f"hello, I am {USERNAME}"}
        self.post("/v1/messages", {"model": "m", "max_tokens": 10, "messages": [first]}, headers)
        sent1 = self.upstream_entries()[-1]["json"]["messages"]
        history = [first, {"role": "assistant", "content": "hi"},
                   {"role": "user", "content": f"my files are in C:\\Users\\{USERNAME}\\docs"}]
        self.post("/v1/messages", {"model": "m", "max_tokens": 10, "messages": history}, headers)
        sent2 = self.upstream_entries()[-1]["json"]["messages"]
        self.assertEqual(sent1[0], sent2[0])        # the earlier turn did not change
        self.assertNotIn(USERNAME, json.dumps(sent2[2]))
        history += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": f"thanks {USERNAME}"}]
        self.post("/v1/messages", {"model": "m", "max_tokens": 10, "messages": history}, headers)
        sent3 = self.upstream_entries()[-1]["json"]["messages"]
        self.assertEqual(sent2, sent3[:3])
        self.assertNotIn(USERNAME, json.dumps(sent3[4]))

    def test_users_get_different_surrogates(self):
        seen = []
        for key in ("gk-alice", "gk-bob"):
            self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": EMAIL}]},
                      {"Authorization": f"Bearer {key}"})
            seen.append(self.upstream_entries()[-1]["json"]["messages"][0]["content"])
        self.assertNotEqual(seen[0], seen[1])

    def test_unknown_key_is_rejected_before_upstream(self):
        before = len(self.upstream_entries()) if os.path.exists(fake_upstream.LOG_PATH) else 0
        status, raw = self.post("/v1/responses", {"model": "m", "input": "hi"}, {"Authorization": "Bearer nope"})
        self.assertEqual(status, 401)
        after = len(self.upstream_entries()) if os.path.exists(fake_upstream.LOG_PATH) else 0
        self.assertEqual(before, after)

    def test_audit_files_match_what_upstream_saw(self):
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": f"x {EMAIL}"}]},
                  {"Authorization": "Bearer gk-alice"})
        audit_dir = os.path.join(self.tmp, "audit")
        newest = sorted(os.listdir(audit_dir))[-1]
        with open(os.path.join(audit_dir, newest), encoding="utf-8") as f:
            audit = json.load(f)
        self.assertEqual(audit["body"], self.upstream_entries()[-1]["json"])
        self.assertEqual(audit["headers"]["Authorization"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
