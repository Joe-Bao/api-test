"""python -m anon_proxy serve -c config.toml | keygen <alias> | mask "text" """
from __future__ import annotations

import argparse
import os
import secrets
import sys
from collections import Counter

from .masking import MaskConfig, Masker, Vault
from .server import DEFAULT_KEEP, load_settings, serve, upstream_key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m anon_proxy")
    sub = ap.add_subparsers(dest="cmd")
    p_serve = sub.add_parser("serve", help="run the gateway")
    p_serve.add_argument("-c", "--config", default="config.toml")
    p_serve.add_argument("--ask-keys", action="store_true",
                         help="prompt (without echo) for upstream API keys that are not set in the environment")
    p_key = sub.add_parser("keygen", help="print a new gateway key line for [users]")
    p_key.add_argument("alias", nargs="?", default="user")
    p_mask = sub.add_parser("mask", help="show how a piece of text would be masked")
    p_mask.add_argument("-c", "--config")
    p_mask.add_argument("text", nargs="?", help="text to mask (default: read stdin)")
    args = ap.parse_args(argv)

    if args.cmd == "keygen":
        print(f'"gk-{secrets.token_urlsafe(24)}" = "{args.alias}"')
        return 0

    if args.cmd == "mask":
        cfg = load_settings(args.config).mask if args.config else MaskConfig(keep=frozenset(DEFAULT_KEEP))
        text = args.text if args.text is not None else sys.stdin.read()
        masker, vault = Masker(cfg), Vault(secrets.token_bytes(32), style=cfg.surrogate_style)
        masker.learn(text, vault)
        used, stats = set(), Counter()
        print(masker.mask(text, vault, used, stats))
        for s, o in sorted(vault.mapping(used).items()):
            print(f"  {s}  <-  {o!r}", file=sys.stderr)
        return 0

    config = getattr(args, "config", None) or "config.toml"
    if not os.path.exists(config):
        print(f"config file {config!r} not found: copy config.example.toml to config.toml first", file=sys.stderr)
        return 2
    settings = load_settings(config)
    if getattr(args, "ask_keys", False):
        import getpass
        ups = (settings.anthropic,) if settings.provider.target == "anthropic" else (settings.openai, settings.anthropic)
        for up in ups:
            if not upstream_key(up):
                value = getpass.getpass(f"{up.api_key_env} (typing is hidden, Enter to skip): ").strip()
                if value:
                    os.environ[up.api_key_env] = value   # lives only in this process
    serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
