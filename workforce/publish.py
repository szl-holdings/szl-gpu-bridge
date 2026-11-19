"""Publish one attested officer record to the Hugging Face receipts dataset.

    python -B -m workforce.publish --record R.json --bundle R.sigstore.json

Stop rules (fail closed):
  * no WORKFORCE_HF_TOKEN in the environment -> skip, upload nothing, exit 0;
  * bundle missing, not a Sigstore DSSE bundle, no signature, no signing
    certificate, or its in-toto subject digest != sha256(record) -> refuse,
    upload nothing, exit 2;
  * the bundle fails CRYPTOGRAPHIC verification (``gh attestation verify``
    against the Sigstore trust root, pinned to this repository, the
    workforce-cycle.yml signer workflow and refs/heads/main) -> refuse, exit 2.
    A missing ``gh``, a timeout or any non-zero exit is a refusal;
  * record carries a credential-shaped string, or the live value of a
    WORKFORCE_* secret -> refuse, exit 2.

``check_bundle`` is only a cheap structural pre-filter (shape and digest
binding). It proves nothing about who signed: a forged bundle with a dummy
certificate passes it, so it is never sufficient on its own. The trust
decision is ``verify_attestation``. The identity constants below are the only
copy: workforce-cycle.yml repeats them as literals and a test keeps the two
in step. The Hugging Face token is never passed to the ``gh`` subprocess.
Uses the Hub commit API directly (stdlib only, no huggingface_hub).
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import core

HF_COMMIT = "https://huggingface.co/api/datasets/{repo}/commit/main"

# Pinned signer identity. Hard-coded on purpose: never derived from the run's
# own environment, so a fork, a renamed repo or a different workflow cannot
# satisfy it. ``--signer-workflow`` takes [host/]<owner>/<repo>/<path>; for a
# reusable workflow the signer is the called workflow file (the certificate's
# job_workflow_ref), which is workforce-cycle.yml.
PINNED_REPO = "szl-holdings/szl-gpu-bridge"
PINNED_SIGNER_WORKFLOW = (
    "szl-holdings/szl-gpu-bridge/.github/workflows/workforce-cycle.yml"
)
PINNED_SOURCE_REF = "refs/heads/main"
GH_VERIFY_TIMEOUT_S = 90
# Only what ``gh`` needs. No WORKFORCE_* variable (the HF token) crosses.
_GH_ENV_KEYS = ("PATH", "HOME", "GH_TOKEN", "GITHUB_TOKEN", "GH_HOST", "TMPDIR", "LANG")


def check_bundle(record_bytes, bundle_bytes):
    """Return None if the bundle binds this record, else a refusal reason."""
    try:
        bundle = json.loads(bundle_bytes)
    except (ValueError, TypeError):
        return "bundle is not JSON"
    if not str(bundle.get("mediaType", "")).startswith(
        "application/vnd.dev.sigstore.bundle"
    ):
        return "not a Sigstore bundle"
    vm = bundle.get("verificationMaterial") or {}
    if not (vm.get("certificate") or vm.get("x509CertificateChain")):
        return "no signing certificate (keyless identity missing)"
    env = bundle.get("dsseEnvelope") or {}
    sigs = [s for s in env.get("signatures") or [] if s.get("sig")]
    if not sigs:
        return "unsigned envelope"
    if env.get("payloadType") != "application/vnd.in-toto+json":
        return "payload is not an in-toto statement"
    try:
        stmt = json.loads(base64.b64decode(env.get("payload", ""), validate=True))
    except (ValueError, TypeError):
        return "payload not decodable"
    want = core.sha256_hex(record_bytes)
    digests = {(s.get("digest") or {}).get("sha256") for s in stmt.get("subject") or []}
    if want not in digests:
        return "attestation subject digest does not match the record"
    return None


def verify_attestation(record_path, bundle_path, env=None, run=subprocess.run):
    """Cryptographically verify the bundle with ``gh attestation verify``,
    pinned to the identity above. Returns None on success, else a reason.
    Fails closed on anything other than a clean zero exit."""
    env = os.environ if env is None else env
    cmd = [
        "gh",
        "attestation",
        "verify",
        str(record_path),
        "--bundle",
        str(bundle_path),
        "--repo",
        PINNED_REPO,
        "--signer-workflow",
        PINNED_SIGNER_WORKFLOW,
        "--source-ref",
        PINNED_SOURCE_REF,
        "--deny-self-hosted-runners",
    ]
    sha = env.get("GITHUB_SHA", "")
    if re.fullmatch(r"[0-9a-f]{40}", sha):
        cmd += ["--source-digest", sha]
    clean_env = {k: env[k] for k in _GH_ENV_KEYS if env.get(k)}
    try:
        proc = run(
            cmd,
            env=clean_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GH_VERIFY_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError:
        return "gh is not available; cannot verify the attestation"
    except (subprocess.TimeoutExpired, OSError):
        return "attestation verification did not complete"
    if proc.returncode != 0:
        return "attestation failed cryptographic verification"
    return None


def _commit(repo, token, files, summary, opener=None):
    lines = [{"key": "header", "value": {"summary": summary, "description": ""}}]
    for path, data in files:
        lines.append(
            {
                "key": "file",
                "value": {
                    "content": base64.b64encode(data).decode(),
                    "path": path,
                    "encoding": "base64",
                },
            }
        )
    body = "\n".join(json.dumps(x) for x in lines).encode()
    req = urllib.request.Request(
        HF_COMMIT.format(repo=repo),
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/x-ndjson",
            "User-Agent": "szl-workforce/1.0",
        },
    )
    opener = opener or urllib.request.build_opener()
    with opener.open(req, timeout=60) as r:
        return r.status


def publish(
    record_path, bundle_path, env=None, committer=_commit, verifier=verify_attestation
):
    env = os.environ if env is None else env
    token = env.get("WORKFORCE_HF_TOKEN", "")
    if not token:
        return "SKIPPED_NO_SECRET", 0
    rp, bp = Path(record_path), Path(bundle_path)
    if not rp.is_file() or not bp.is_file():
        return "REFUSED: record or attestation bundle missing (unsigned)", 2
    record = rp.read_bytes()
    try:
        core.refuse_secrets(record, env=env)
    except ValueError as e:
        return f"REFUSED: {e}", 2
    reason = check_bundle(record, bp.read_bytes())
    if reason:
        return f"REFUSED: {reason}", 2
    reason = verifier(rp, bp, env=env)
    if reason:
        return f"REFUSED: {reason}", 2
    repo = env.get("WORKFORCE_DATASET") or core.DEFAULT_DATASET
    name = rp.name
    files = [
        (f"officers/{name}", record),
        (f"officers/{name[: -len('.json')]}.sigstore.json", bp.read_bytes()),
    ]
    committer(repo, token, files, f"officer record {name} (sigstore-attested)")
    return f"PUBLISHED officers/{name}", 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--bundle", required=True)
    a = ap.parse_args(argv)
    msg, code = publish(a.record, a.bundle)
    print(msg, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
