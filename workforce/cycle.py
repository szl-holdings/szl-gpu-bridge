"""Run exactly one officer cycle and write one JSON record.

    python -B -m workforce.cycle --role eternal-auditor --out DIR

Prints ``record=<path>`` and ``sha256=<hex>`` (also to $GITHUB_OUTPUT when
set). Exits non-zero, writing nothing, if the record would carry a
credential-shaped string. Never uploads: publishing is a separate step that
requires a verified Sigstore attestation (see workforce/publish.py).
"""

import argparse
import os
import sys
from pathlib import Path

from . import core, duties


def gateway_from_env(env=None):
    env = os.environ if env is None else env
    gw = (env.get("COMPUTE_GATEWAY") or duties.DEFAULT_GATEWAY).rstrip("/")
    if not core.is_public_https_host(gw):
        raise SystemExit("COMPUTE_GATEWAY must be https on a DNS hostname")
    return gw


def run(role, out_dir, net, env=None, now=None):
    if role not in duties.ROLES:
        raise SystemExit(f"unknown role {role!r}")
    fn = duties.ROLES[role]
    if role in ("szl-chief-engineer", "szl-cyber-shield"):
        findings = fn(net, gateway=gateway_from_env(env))
    else:
        findings = fn(net)
    _, blob = core.build_record(role, findings, env=env, now=now)
    core.refuse_secrets(blob, env=env)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / core.record_name(role, blob)
    path.write_bytes(blob)
    return path, core.sha256_hex(blob)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=sorted(duties.ROLES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    net = core.Net(gh_read_token=os.environ.get("WORKFORCE_GH_READ_TOKEN", ""))
    try:
        path, digest = run(a.role, a.out, net)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    print(f"record={path}")
    print(f"sha256={digest}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"record={path}\nsha256={digest}\nname={path.name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
