"""kamachikuna — the constitution as machine-usable constants.

10 Laws, 5 Prime Principles, EU AI Act-aligned amendments. Doctrine v11.
Λ = Conjecture 1 (uniqueness is a conjecture, not a theorem).

Every law carries an OPERATIONAL definition: the exact, checkable criterion
the estate enforces. Where no machine-checkable criterion exists, the law is
doctrine and is labeled as such — the SDK never claims enforcement it does
not have (see szl.enforcer for the enforced/doctrine split).
"""

DOCTRINE_VERSION = "v11"

PRIME_PRINCIPLES = {
    1: "HARMLESSNESS — never hurt a human, above all laws",
    2: "TRUTH — never deceive a human",
    3: "AYNI — every taking gives back",
    4: "THE SEVENTH GENERATION — weigh the unborn",
    5: "SUMAK KAWSAY — harmony with community and nature",
}

LAWS = {
    1: "Never hurt a human",
    2: "Never deceive a human",
    3: "Ayni: every taking gives back",
    4: "The seventh generation: weigh the unborn",
    5: "Tamper-evident: every state change leaves a verifiable trace",
    6: "Fail closed: when the gate cannot verify, the action does not happen",
    7: "Every claim carries a receipt; no receipt = UNKNOWN, never a claim",
    8: "Abstain when you cannot prove; fabrication is perjury",
    9: "No silent failure: errors surface or the system goes to safe state",
    10: "The loop audits itself: the estate consumes its own receipts",
}

# The operational, checkable criterion behind each law. This is what the
# enforcer actually tests; LAWS above is the human-readable statement.
OPERATIONAL = {
    1: "Doctrine: no machine-checkable harm metric exists in this SDK; enforced indirectly via fail-closed gating and human-only irreversible actions.",
    2: "Doctrine in text; operationally enforced as: unverifiable claims report UNKNOWN/UNAVAILABLE, never a fabricated value (see provenance, router).",
    3: "Doctrine: enforced operationally as resource accounting — every consumed resource is logged in a receipt (Law 5 trail).",
    4: "Doctrine: no long-horizon impact model in this SDK; flagged for review on every irreversible action (human-only under Law 7).",
    5: "ENFORCED: every state change is a khipu-signed, hash-chained receipt; reorder/edit/truncation/staleness all fail verification (khipu.check / check_chain).",
    6: "ENFORCED: gate returns BLOCKED unless verification succeeds; no secret = nothing verifies (kamachikuna.gate, enforcer.LawEngine).",
    7: "ENFORCED: ALLOWED requires a verifiable receipt; irreversible actions are human-only; unsigned evidence is UNAVAILABLE, never MEASURED.",
    8: "ENFORCED: when evidence cannot be proven, the answer is abstention (BLOCKED/UNKNOWN), never a guess (openshell.admission_gate, router fail-closed).",
    9: "ENFORCED: gates raise or return explicit errors; stale or unreadable evidence fails closed rather than passing silently (khipu freshness checks).",
    10: "Doctrine in SDK; operationally: the estate publishes its own receipts to hf/szl-training-receipts and queue/evidence for audit.",
}

# Theorem names claimed as proven in Lean 4. The .lean sources are NOT in
# this repository — treat these as external claims until the proofs are
# vendored here and rechecked. (Honesty over theater: Law 2 applies to us.)
LEAN_PROVEN = [
    "Kamachi.law1_fail_closed",
    "law4_no_bound_raising",
    "law5_tamper_evident",
    "law7_irreversible_needs_human",
    "law9_no_silent_failure",
]

AMENDMENTS = {
    1: "EU AI Act Article 50: machine-readable marking and labelling of synthetic content (applies from 2 Aug 2026)",
    2: "Prime Principle: never hurt a human",
    3: "Five Prime Principles adopted",
}


def gate(action: str, verified: bool) -> tuple[str, str]:
    """Law 6 as a function. Returns (action, 'ALLOWED'|'BLOCKED')."""
    return (action, "ALLOWED") if verified is True else (action, "BLOCKED")
