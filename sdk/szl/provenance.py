"""provenance — the Korea/Naver bar as a machine-checkable badge.

A model claims provenance only when the evidence is present and bound.
States are honest: UNTRAINED, CARD_ONLY, MEASURED, UNAVAILABLE, NOT_CLAIMED.
Verification proves integrity and origin of the evidence — never accuracy
or performance of the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import khipu

BADGE_KIND = "szl.provenance-badge/v1"

STATES = ("UNTRAINED", "CARD_ONLY", "MEASURED", "UNAVAILABLE", "NOT_CLAIMED")


@dataclass
class ProvenanceClaim:
    """What a model card claims, with the evidence that must back it."""
    model_id: str
    source_sha: str | None = None          # canonical GitHub revision trained from
    training_receipt: dict[str, Any] | None = None  # signed khipu receipt
    datasets: dict[str, str] = field(default_factory=dict)  # name -> sha256

    def state(self, secret: str | None = None) -> str:
        if self.training_receipt is None:
            return "CARD_ONLY" if self.source_sha else "UNTRAINED"
        if secret is not None and not khipu.check(self.training_receipt, secret):
            return "UNAVAILABLE"  # receipt present but unverifiable
        if not self.source_sha or not self.datasets:
            return "UNAVAILABLE"
        return "MEASURED"

    def badge(self, secret: str | None = None) -> dict[str, Any]:
        """Mint the provenance badge. Honest states only, never a guess."""
        state = self.state(secret)
        body = {
            "kind": BADGE_KIND,
            "model_id": self.model_id,
            "state": state,
            "source_sha": self.source_sha,
            "datasets": self.datasets,
            "note": "Verification proves integrity and origin, never accuracy or performance.",
        }
        if secret is None:
            return body  # unsigned inspection badge
        return khipu.mint(body, secret)
