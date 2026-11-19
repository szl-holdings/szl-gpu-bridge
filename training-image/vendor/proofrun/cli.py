"""proofrun CLI.  proofrun verify <dir> [--trusted-key FP]... [--accept-retroactive] [--json]
                  proofrun demo <dir>     (writes a synthetic signed run with an ephemeral key)"""
from __future__ import annotations

import argparse
import json
import sys

from .verify import render, verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="proofrun")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify every claim in a run directory (exit 0 only if all pass)")
    v.add_argument("dir")
    v.add_argument("--trusted-key", action="append", default=[], help="signer fingerprint ed25519:<16 hex>")
    v.add_argument("--accept-retroactive", action="store_true",
                   help="evaluate runs that predate proofrun (no heartbeat chain or image digest)")
    v.add_argument("--json", action="store_true")
    d = sub.add_parser("demo", help="write a synthetic proof-carrying run (ephemeral key)")
    d.add_argument("dir")
    d.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        rep = verify(a.dir, set(a.trusted_key) or None, a.accept_retroactive)
        print(json.dumps(rep, indent=1) if a.json else render(rep))
        return 0 if rep["verdict"] == "PASS" else 1
    if a.cmd == "demo":
        from .synthetic import make_run
        info = make_run(a.dir, steps=a.steps)
        print(json.dumps(info, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
