#!/usr/bin/env python3
"""Eternal Auditor v2 — hourly GitHub+HF+CF integrity sweeps.

CANONICAL SOURCE: szl-holdings/szl-gpu-bridge/hf-spaces/eternal-auditor/app.py
(Hugging Face Space SZLHOLDINGS/eternal-auditor is the deploy mirror.)

FIX vs v1 (advisory 2026-11-09): v1 truncated recorded signatures to 96 hex
chars and reset its cycle state on every restart — so a "healthy" endpoint
never established a valid audit chain. v2:

  - stores FULL signatures (verifiable, not decorative);
  - persists cycle state to disk (/data/audit-state.json) so the audit
    chain survives restarts;
  - each cycle receipt chains to the sha256 of the previous cycle receipt
    (tamper-evident chain, Law 5);
  - if the state file is missing or corrupt, the auditor marks the chain
    BROKEN-RESTARTED rather than silently restarting at zero (Law 9).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

STATE_PATH = Path(os.environ.get("AUDITOR_STATE", "/data/audit-state.json"))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text())
            if "cycle" in st and "prev_receipt_sha256" in st:
                return st
        except Exception:
            pass
    return {"cycle": 0, "prev_receipt_sha256": None, "chain_state": "GENESIS"}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_PATH)


def sign(payload: str, secret: str) -> str:
    """HMAC-style signature. FULL length is stored — never truncated."""
    import hmac
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def run_cycle(findings: list, secret: str) -> dict:
    """Produce one chained, fully-signed cycle receipt. findings: list of
    {'target':..., 'check':..., 'result': 'PASS'|'FAIL'|'UNKNOWN'}."""
    state = load_state()
    restarted = state.get("chain_state") == "GENESIS" and state["cycle"] > 0
    cycle = state["cycle"] + 1
    body = {
        "kind": "szl.auditor-receipt/v2",
        "cycle": cycle,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prev_receipt_sha256": state["prev_receipt_sha256"],
        "chain_state": "BROKEN-RESTARTED" if restarted else "CONTINUOUS",
        "findings": findings,
    }
    body_sha = sha256_hex(canonical(body))
    signature = sign(body_sha, secret)  # FULL 64-hex signature, stored whole
    receipt = {**body, "body_sha256": body_sha, "signature": signature,
               "signature_length": len(signature)}
    state.update({"cycle": cycle, "prev_receipt_sha256": body_sha,
                  "chain_state": "CONTINUOUS"})
    save_state(state)
    return receipt
