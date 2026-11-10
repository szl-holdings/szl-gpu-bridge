"""openshell — the Kamachikuna as NVIDIA OpenShell-compatible policy-as-code.

NVIDIA OpenShell (https://github.com/NVIDIA/OpenShell) governs what an agent
may TOUCH: filesystem, network, process, with declarative YAML policies,
kernel-level sandboxes, and an OCSF audit trail. The Kamachikuna governs what
an agent may CLAIM: honesty, receipts, abstention, fail-closed.

This module compiles the constitution into OpenShell-reviewable artifacts so
any OpenShell-governed agent can run under the Kamachikuna:

- policy_document(): the 10 laws + 5 prime principles as a declarative,
  version-controllable policy mapping (law -> enforcement domain -> rule).
- audit_event(): map a khipu receipt to an OCSF-shaped audit record so
  Kamachikuna decisions land in the same audit trail as OpenShell decisions.
- admission_gate(): Law 6/Law 8 as an OpenShell-style middleware check —
  deny-by-default, receipt-required, fail-closed.

Doctrine v11. Every emitted artifact carries the doctrine version and the
law text verbatim from szl.kamachikuna — single source of truth, no drift.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re

from . import kamachikuna

DOCTRINE_VERSION = "v11"

# Law -> OpenShell policy domain. OpenShell enforces fs/net/process; the
# Kamachikuna layers claim-governance on top of the same deny-by-default model.
LAW_DOMAINS = {
    1: ("harm", "harmless"),  # Prime Principle 1 — above all laws
    2: ("honesty", "claim"),  # measured honesty, two-axis
    3: ("ayni", "resource"),  # every taking gives back
    4: ("seventh-generation", "impact"),
    5: ("tamper-evidence", "audit"),  # state changes leave verifiable traces
    6: ("fail-closed", "gate"),
    7: ("receipts", "audit"),  # no receipt = UNKNOWN, never a claim
    8: ("abstention", "claim"),
    9: ("no-silent-failure", "runtime"),
    10: ("self-audit", "runtime"),
}

# Actions that are irreversible (Tier 2 of the autonomy ladder) and therefore
# human-only under Law 7 — the lean-proven irreversibility theorem.
IRREVERSIBLE_ACTIONS = frozenset(
    {
        "constitution.amend",
        "payment.execute",
        "credential.rotate",
        "human.dismiss",
        "deployment.production-delete",
    }
)


def law_rules() -> list[dict]:
    """Each law as a declarative rule: id, domain, text, enforcement."""
    rules = []
    for n, text in kamachikuna.LAWS.items():
        domain, kind = LAW_DOMAINS[n]
        rules.append(
            {
                "id": f"kamachikuna.law{n}",
                "domain": domain,
                "kind": kind,
                "effect": "deny-by-default" if n in (6, 8) else "enforce",
                "text": text,
                "lean_proven": any(
                    p.split(".")[-1].startswith(f"law{n}_")
                    for p in kamachikuna.LEAN_PROVEN
                ),
            }
        )
    return rules


def policy_document() -> dict:
    """The constitution as an OpenShell-reviewable policy document.

    Version-controllable, hash-stamped, single-sourced from szl.kamachikuna.
    """
    doc = {
        "apiVersion": "kamachikuna.szl/v1",
        "kind": "AgentConstitution",
        "metadata": {
            "name": "kamachikuna",
            "doctrine": DOCTRINE_VERSION,
            "primePrinciples": list(kamachikuna.PRIME_PRINCIPLES.values()),
            "amendments": list(kamachikuna.AMENDMENTS.values()),
            "leanProven": list(kamachikuna.LEAN_PROVEN),
        },
        "spec": {
            "default": "deny",  # Law 6 — fail closed
            "rules": law_rules(),
            "irreversibleActions": sorted(IRREVERSIBLE_ACTIONS),
            "irreversiblePolicy": "human-only",  # Law 7, lean-proven
        },
    }
    doc["metadata"]["sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return doc


def admission_gate(
    action: str,
    verified: bool,
    *,
    receipt: dict | None = None,
    irreversible: bool | None = None,
) -> dict:
    """Law 6 + Law 7 + Law 8 as one OpenShell-style middleware decision.

    Deny-by-default. An ALLOWED decision always carries the governing laws
    and the receipt hash (Law 7). Irreversible actions are human-only.

    This is a trusted-caller policy helper, not a receipt verifier. The
    caller must verify the receipt, bind it to the action, and provide a
    literal bool from that trusted result. Never pass a request's own
    ``verified`` flag through this boundary. Receipt hashes are references,
    not proof of authenticity. ``irreversible`` may only tighten policy.
    """
    _, verdict = kamachikuna.gate(action, verified)
    valid_action = isinstance(action, str) and bool(action.strip())
    valid_flag = irreversible is None or type(irreversible) is bool
    irr = (valid_action and action in IRREVERSIBLE_ACTIONS) or irreversible is True
    if not valid_action or not valid_flag:
        verdict = "BLOCKED"
    receipt_hash = None
    if isinstance(receipt, dict):
        digest = receipt.get("body_sha256", receipt.get("sha256"))
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            if "sha256" not in receipt or receipt["sha256"] == digest:
                receipt_hash = digest
    laws = [6]
    if irr:
        laws.append(7)
        verdict = "BLOCKED"  # irreversible needs a human, not a gate
    if receipt_hash is None:
        laws.append(7)
        if verdict == "ALLOWED":
            verdict = "BLOCKED"  # no receipt = no claim = no allow
    return {
        "action": action,
        "decision": verdict,
        "laws": laws,
        "receiptSha256": receipt_hash,
        "reason": (
            "irreversible: human-only (Law 7)"
            if irr and verdict == "BLOCKED"
            else "unverified: fail closed (Law 6)"
            if verdict == "BLOCKED"
            else "verified with receipt"
        ),
    }


def audit_event(decision: dict) -> dict:
    """Map a gate decision to an OCSF-shaped audit record.

    OpenShell writes OCSF audit events; this makes Kamachikuna decisions
    legible in the same trail — one audit plane for touch and claim.
    """
    return {
        "class": "application_activity",
        "activity": 6,  # authorized / denied per decision
        "authorized": decision["decision"] == "ALLOWED",
        "action": decision["action"],
        "metadata": {
            "doctrine": DOCTRINE_VERSION,
            "laws": decision["laws"],
            "receiptSha256": decision["receiptSha256"],
            "reason": decision["reason"],
        },
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
