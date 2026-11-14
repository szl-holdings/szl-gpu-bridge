"""enforcer — the Kamachikuna as an executable law engine.

Honest about what code can and cannot do: laws that are machine-checkable
(5, 6, 7, 8, 9) are enforced here; laws that govern judgment (1, 2, 3, 4,
10) are reported as DOCTRINE, never claimed as enforced. Every decision —
allowed or blocked — is minted as a signed khipu receipt (Law 5), so the
enforcement trail is itself tamper-evident.
"""
from __future__ import annotations

from typing import Any

from . import kamachikuna, khipu, openshell

ENFORCED_LAWS = frozenset({5, 6, 7, 8, 9})
DOCTRINE_LAWS = frozenset({1, 2, 3, 4, 10})

ENFORCEMENT_HOOKS = {
    5: "khipu hash-chained signed receipts; reorder/edit/truncation detectable",
    6: "fail-closed gate: unverified action never runs",
    7: "no receipt = no claim; irreversible actions are human-only",
    8: "unverifiable evidence abstains: UNKNOWN/BLOCKED, never a guess",
    9: "gate raises explicitly; no silent failure path",
}


class LawEngine:
    """Evaluates actions against the constitution. Fail closed: without a
    signing secret it can still BLOCK, but it will never mint an ALLOWED
    receipt — an unsigned allow would violate Law 7."""

    def __init__(self, secret: str | None = None) -> None:
        self._secret = secret

    def law_status(self) -> dict[int, dict[str, str]]:
        """Every law classified: enforced with its hook, or doctrine."""
        status = {}
        for n, text in kamachikuna.LAWS.items():
            if n in ENFORCED_LAWS:
                status[n] = {"status": "ENFORCED", "hook": ENFORCEMENT_HOOKS[n], "text": text}
            else:
                status[n] = {"status": "DOCTRINE", "hook": "not machine-checkable in this SDK", "text": text}
        return status

    def evaluate(
        self,
        action: str,
        verified: bool,
        *,
        receipt: dict[str, Any] | None = None,
        irreversible: bool | None = None,
    ) -> dict[str, Any]:
        """One governed decision, receipted. Wraps openshell.admission_gate
        and adds the tamper-evident receipt trail (Law 5)."""
        decision = openshell.admission_gate(
            action, verified, receipt=receipt, irreversible=irreversible
        )
        signed = None
        if self._secret is not None:
            signed = khipu.mint(
                {"kind": "szl.gate-decision/v1", "decision": decision},
                self._secret,
            )
        return {**decision, "receipt": signed}

    def verify_receipt(self, receipt: dict[str, Any]) -> bool:
        """Law 6 applied to evidence: no secret means nothing verifies."""
        if self._secret is None:
            return False
        return khipu.check(receipt, self._secret)

    def verify_chain(
        self,
        receipts: list[dict[str, Any]],
        *,
        expected_head: str | None = None,
        max_age_days: float | None = None,
    ) -> bool:
        if self._secret is None:
            return False
        return khipu.check_chain(
            receipts, self._secret,
            expected_head=expected_head, max_age_days=max_age_days,
        )
