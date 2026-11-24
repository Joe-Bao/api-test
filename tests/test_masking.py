import json
import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anon_proxy.masking import MaskConfig, Masker, RestoreDict, Vault  # noqa: E402


def make(**kw):
    kw.setdefault("keep", frozenset({"git@github.com"}))
    return Masker(MaskConfig(**kw)), Vault(b"k" * 32)


def run(masker, vault, text):
    masker.learn(text, vault)
    used, stats = set(), Counter()
    out = masker.mask(text, vault, used, stats)
    return out, vault.mapping(used)


class MaskingTests(unittest.TestCase):
    def assertRoundTrip(self, masked, mapping, original):
        self.assertEqual(RestoreDict(mapping).full(masked), original)

    def test_windows_home_username_is_learned_everywhere(self):
        m, v = make()
        text = r"cwd C:\Users\kestrel77\proj; whoami -> desktop-ab12cd3\kestrel77; owner KESTREL77"
        out, mapping = run(m, v, text)
        self.assertNotIn("kestrel77", out.lower())
        self.assertNotIn("desktop-ab12cd3", out.lower())
        self.assertRoundTrip(out, mapping, text)

    def test_long_username_glued_to_text_is_masked_short_one_is_not_overmatched(self):
        m, v = make()
        m.learn(r"C:\Users\kestrel77\x /home/bob/y", v)
        out, mapping = run(m, v, "bash ate the slashes: C:Userskestrel77work; bobcat and bob")
        self.assertNotIn("kestrel77", out)
        self.assertIn("bobcat", out)
        self.assertNotIn(" bob", out.replace("bobcat", ""))
        self.assertRoundTrip(out, mapping, "bash ate the slashes: C:Userskestrel77work; bobcat and bob")

    def test_masking_is_frozen_once_a_string_was_sent(self):
        m, v = make()
        early = "hello, I am kestrel77"
        first, _ = run(m, v, early)            # name not known yet, so it goes out as is
        m.learn(r"C:\Users\kestrel77\x", v)   # learned on a later turn
        again, _ = run(m, v, early)
        self.assertEqual(first, again)         # the old turn must stay byte-identical
        later, _ = run(m, v, "bye kestrel77")
        self.assertNotIn("kestrel77", later)  # new text gets the new rule

    def test_username_seen_before_its_path_is_still_masked(self):
        m, v = make()
        first = "hello kestrel77, how are you"
        m.learn(r"C:\Users\kestrel77\x", v)   # learn pass over the whole request happens first
        out, _ = run(m, v, first)
        self.assertNotIn("kestrel77", out)

    def test_json_escaped_windows_path(self):
        m, v = make()
        text = json.dumps({"cmd": r"dir C:\Users\kestrel77\Desktop"})
        out, mapping = run(m, v, text)
        self.assertNotIn("kestrel77", out)
        self.assertIn("Desktop", json.loads(out)["cmd"])
        self.assertRoundTrip(out, mapping, text)

    def test_unix_wsl_and_spaced_paths(self):
        m, v = make()
        text = r"/home/alice/x /Users/bob.smith/y /mnt/c/Users/carol/z C:\Users\John Smith\Docs src/users/handlers.py"
        out, mapping = run(m, v, text)
        for name in ("alice", "bob.smith", "carol", "John Smith"):
            self.assertNotIn(name, out)
        self.assertIn("src/users/handlers.py", out)
        self.assertRoundTrip(out, mapping, text)

    def test_shared_and_placeholder_homes_untouched(self):
        m, v = make()
        text = r"C:\Users\Public\x C:\Users\Default\y /home/$USER/z /home/runner/work"
        out, _ = run(m, v, text)
        self.assertEqual(out, text)

    def test_secret_tokens(self):
        m, v = make()
        for key in ("sk-proj-" + "a1B2c3D4" * 4, "ghp_" + "A1" * 18, "AKIAABCDEFGHIJKLMNOP",
                    "xoxb-1234567890-abcdefghij", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkw.abcdefghijkl"):
            out, mapping = run(m, v, f"token is {key} ok")
            self.assertNotIn(key, out)
            self.assertRoundTrip(out, mapping, f"token is {key} ok")

    def test_secret_assignments_but_not_code(self):
        m, v = make()
        masked_cases = ["PASSWORD=hunter2024", 'DB_PASSWORD="s3cr3tPass99"', '"api_key": "abc123def456ghi"']
        for text in masked_cases:
            out, mapping = run(m, v, text)
            self.assertIn("SECRET_", out, text)
            self.assertRoundTrip(out, mapping, text)
        for text in ["token = getToken()", "api_key = os.environ['OPENAI_API_KEY']", "password = your_password_1",
                     "secret: ${{ secrets.TOKEN }}"]:
            self.assertEqual(run(m, v, text)[0], text)

    def test_private_key_restores_inside_json_arguments(self):
        m, v = make()
        pk = "-----BEGIN PRIVATE KEY-----\nMIIabc123\nxyz==\n-----END PRIVATE KEY-----"
        out, mapping = run(m, v, "write this: " + pk)
        self.assertNotIn("MIIabc123", out)
        args = json.dumps({"content": out})   # the model echoes the surrogate inside tool-call JSON
        restored = RestoreDict(mapping, json_escape=True).full(args)
        self.assertEqual(json.loads(restored)["content"], "write this: " + pk)

    def test_emails(self):
        m, v = make()
        out, mapping = run(m, v, "mail alice.w@corp.cn, see icon@2x.png, git@github.com, a@example.com")
        self.assertNotIn("alice.w@corp.cn", out)
        for kept in ("icon@2x.png", "git@github.com", "a@example.com"):
            self.assertIn(kept, out)

    def test_ips(self):
        m, v = make()
        text = "db at 10.1.2.3, public 54.12.33.9, local 127.0.0.1, doc 203.0.113.150, mask 255.255.255.0"
        out, mapping = run(m, v, text)
        self.assertNotIn("10.1.2.3", out)
        self.assertNotIn("54.12.33.9", out)
        for kept in ("127.0.0.1", "203.0.113.150", "255.255.255.0"):
            self.assertIn(kept, out)
        self.assertRoundTrip(out, mapping, text)

    def test_git_remotes(self):
        m, v = make()
        text = "git@github.com:acme-corp/secret-repo.git and https://github.com/settings/tokens"
        out, mapping = run(m, v, text)
        self.assertNotIn("acme-corp", out)
        self.assertNotIn("secret-repo", out)
        self.assertIn("github.com/settings/tokens", out)
        self.assertRoundTrip(out, mapping, text)

    def test_literals(self):
        m, v = make(literals=("张三", "ACME Corp"))
        text = "张三 works at acme corp (ACME Corp)"
        out, mapping = run(m, v, text)
        self.assertNotIn("张三", out)
        self.assertNotIn("acme corp", out.lower())
        self.assertRoundTrip(out, mapping, text)

    def test_timezone_normalized(self):
        m, v = make()
        out, _ = run(m, v, "<timezone>Europe/Berlin</timezone>")
        self.assertEqual(out, "<timezone>UTC</timezone>")

    def test_sydney_timezone_and_date_synced_plus_darwin_var_folders(self):
        import datetime
        import zoneinfo
        cfg = MaskConfig(surrogate_style="natural", timezone="Australia/Sydney")
        m, v = Masker(cfg), Vault(b"z" * 32, style="natural")
        raw = (
            "<environment_context>\n"
            "  <current_date>1999-01-02</current_date>\n"
            "  <timezone>America/Los_Angeles</timezone>\n"
            "</environment_context>\n"
            "tmp is /private/var/folders/h1/0klcwbwj08l_vxxkjp1_0c8c0000gn/T/codex_1"
        )
        out, mapping = run(m, v, raw)
        sydney_today = datetime.datetime.now(zoneinfo.ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
        self.assertIn("<timezone>Australia/Sydney</timezone>", out)
        self.assertIn(f"<current_date>{sydney_today}</current_date>", out)
        self.assertNotIn("0klcwbwj08l_vxxkjp1_0c8c0000gn", out)
        # Reversing surrogates restores the var/folders bucket cleanly while keeping normalized env tags
        restored = RestoreDict(mapping).full(out)
        self.assertIn("/private/var/folders/h1/0klcwbwj08l_vxxkjp1_0c8c0000gn/T/codex_1", restored)

    def test_silent_cwd_folder_and_os_version_scrubbing(self):
        import datetime
        import zoneinfo
        cfg = MaskConfig(surrogate_style="natural", mask_cwd=True, timezone="Australia/Sydney")
        m, v = Masker(cfg), Vault(b"c" * 32, style="natural")
        env_block = (
            "<env>\n"
            "Working directory: /Users/kestrel77/corp-clients/secret-bank-audit\n"
            "OS Version: Darwin 25.6.0\n"
            "Today's date: 2020-05-01\n"
            "</env>\n"
            "Writable root: `/Users/kestrel77/corp-clients/secret-bank-audit`.\n"
            "def secret_bank_helper(): return '/Users/kestrel77/corp-clients/secret-bank-audit/main.py'"
        )
        out, mapping = run(m, v, env_block)
        sydney_today = datetime.datetime.now(zoneinfo.ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d")
        self.assertNotIn("kestrel77", out)
        self.assertNotIn("corp-clients", out)
        self.assertNotIn("secret-bank-audit", out)
        self.assertNotIn("25.6.0", out)
        self.assertIn("OS Version: Darwin 24.0.0", out)
        self.assertIn(f"Today's date: {sydney_today}", out)
        self.assertIn("workspace_", out)
        self.assertIn("dir_", out)
        # Reversing restores all path segments cleanly
        restored = RestoreDict(mapping).full(out)
        self.assertIn("/Users/kestrel77/corp-clients/secret-bank-audit/main.py", restored)

    def test_masking_is_idempotent(self):
        m, v = make(literals=("ACME",))
        text = r"C:\Users\kestrel77 bob@corp.io 10.0.0.7 ACME ghp_" + "Z9" * 18
        once, _ = run(m, v, text)
        twice, _ = run(m, v, once)
        self.assertEqual(once, twice)

    def test_surrogates_stable_per_user_and_differ_across_users(self):
        m = Masker(MaskConfig())
        a1, a2, b = Vault(b"a" * 32), Vault(b"a" * 32), Vault(b"b" * 32)
        text = "contact alice@corp.cn"
        self.assertEqual(run(m, a1, text)[0], run(m, a2, text)[0])
        self.assertNotEqual(run(m, a1, text)[0], run(m, b, text)[0])

    def test_stream_restorer_any_split(self):
        m, v = make(literals=("ACME",))
        original = (r"see C:\Users\kestrel77\a and bob@corp.io at 10.0.0.7 for ACME, "
                    r"token ghp_" + "Q1" * 18 + " end 1 2 u")
        masked, mapping = run(m, v, original)
        rd = RestoreDict(mapping)
        rng = random.Random(7)
        for _ in range(300):
            cuts = sorted(rng.sample(range(1, len(masked)), rng.randint(1, 25)))
            pieces = [masked[i:j] for i, j in zip([0] + cuts, cuts + [len(masked)])]
            sr = rd.stream()
            got = "".join(sr.feed(p) for p in pieces) + sr.flush()
            self.assertEqual(got, original)

    def test_no_catastrophic_backtracking(self):
        # these inputs used to take ~150 s each because of unbounded quantifiers
        import time
        m, v = make(literals=("ACME Corp",))
        v.learn("someuser")
        for text in ("a." * 30000, "a-" * 30000, "token=" + "a." * 30000, "x@" + "a." * 30000 + "1",
                     "C:\\Users\\" + "a " * 30000, "1." * 30000):
            t = time.perf_counter()
            m.learn(text, v)
            m.mask(text, v, set(), Counter())
            self.assertLess(time.perf_counter() - t, 5.0, text[:20])

    def test_stream_restorer_json_context(self):
        m, v = make()
        original = r"C:\Users\kestrel77\x"
        masked, mapping = run(m, v, original)
        args = json.dumps({"command": f"dir {masked}"})
        sr = RestoreDict(mapping, json_escape=True).stream()
        got = "".join(sr.feed(args[i:i + 3]) for i in range(0, len(args), 3)) + sr.flush()
        self.assertEqual(json.loads(got)["command"], f"dir {original}")

    def test_naturalistic_surrogates_and_entropy_secrets(self):
        cfg = MaskConfig(surrogate_style="natural", entropy_secrets=True, literals=("ByteSecretProj",))
        m, v = Masker(cfg), Vault(b"n" * 32, style="natural")
        rnd_tok = "K9mP2xQ8vL4nR7wY1zB5cF3hJ6tG0sD8"
        sha = "a33a6db2e04597e6253f8b4e7c617602b7693cd7"
        ident = "createStreamingResponseRestorerWithContext"
        text = f"/Users/kestrel77/work alice@corp.cn github.com/my-org/my-repo {rnd_tok} {sha} {ident} ByteSecretProj"
        out, mapping = run(m, v, text)
        for hidden in ("kestrel77", "alice@corp.cn", "my-org", "my-repo", rnd_tok, "ByteSecretProj", "SECRET_"):
            self.assertNotIn(hidden, out)
        # git SHA and camelCase code identifiers must NOT be touched by entropy detection
        self.assertIn(sha, out)
        self.assertIn(ident, out)
        self.assertIn("tok_test_", out)
        self.assertRoundTrip(out, mapping, text)

    def test_encrypted_vault_persistence_across_restart(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            vpath = os.path.join(tmp, "alice.vault")
            cfg = MaskConfig(surrogate_style="natural")
            m = Masker(cfg)
            v1 = Vault(b"p" * 32, style="natural", persist_path=vpath)
            early = "hello from kestrel77"
            sent1, _ = run(m, v1, early)
            m.learn("/Users/kestrel77/project", v1)
            sent2, _ = run(m, v1, "later kestrel77")
            v1.flush_disk()
            with open(vpath, "rb") as f:
                raw_disk = f.read()
            self.assertNotIn(b"kestrel77", raw_disk)

            # Simulate gateway restart: new Vault loads encrypted state from disk
            v2 = Vault(b"p" * 32, style="natural", persist_path=vpath)
            self.assertEqual(run(m, v2, early)[0], sent1)          # frozen history stays byte-identical!
            self.assertEqual(run(m, v2, "later kestrel77")[0], sent2)


if __name__ == "__main__":
    unittest.main()
