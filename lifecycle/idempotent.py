"""Idempotency-first job control for GPU lifecycle calls (propose-only).

Doctrine (leader-study card r1, docs/leader-study-r1.md aligning #337):
duplicate submissions with the same idempotency token return the prior
decision and NEVER re-execute; expired tokens are rejected (a fresh
token is required per lifecycle attempt); execution happens at most once
per accepted token; costs are recorded into a ledger per accepted job.

Pure Python, stdlib only. No cloud (SkyPilot, W&B) is contacted -
execution is an injected callback and cost an injected amount, and the
clock is injected (now_ts) for deterministic expiry semantics.

Honest limitation, verbatim: this idempotency registry is
in-process/local only and provides no distributed or cross-restart
guarantee unless explicitly implemented and disclosed (not the case
here).

Propose-only, unsigned; merges are owned by the estate signature wall.
No status claims are made about any other pull request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

EXECUTED = "EXECUTED"
DUPLICATE_NOT_REEXECUTED = "DUPLICATE_NOT_REEXECUTED"
TOKEN_EXPIRED_REJECTED = "TOKEN_EXPIRED_REJECTED"
CALLBACK_ERROR = "CALLBACK_ERROR"


class TokenError(ValueError):
    """A job spec or idempotency token is malformed."""


@dataclass
class JobSpec:
    token: str
    payload_fingerprint: str
    ttl_s: float
    submitted_at: float

    def __post_init__(self):
        if not self.token or not isinstance(self.token, str):
            raise TokenError("token must be a non-empty string")
        if not self.payload_fingerprint or not isinstance(self.payload_fingerprint, str):
            raise TokenError("payload_fingerprint must be a non-empty string")
        if not self.ttl_s or self.ttl_s <= 0:
            raise TokenError("ttl_s must be positive")

    def expires_at(self) -> float:
        return self.submitted_at + self.ttl_s


@dataclass
class JobDecision:
    token: str
    outcome: str
    executed: bool
    result: object = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    first_seen_at: float = 0.0

    def as_dict(self) -> dict:
        return {"token": self.token, "outcome": self.outcome, "executed": self.executed,
                "cost_usd": self.cost_usd, "first_seen_at": self.first_seen_at}


class IdempotentRegistry:
    """Token-keyed execution registry with terminal-state storage."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._decisions: Dict[str, JobDecision] = {}
        self._costs: List[Tuple[str, float]] = []

    def submit(self, spec: JobSpec, run: Callable[[], object], cost_usd: float = 0.0,
               now_ts: Optional[float] = None) -> JobDecision:
        """Submit a lifecycle job under idempotency-first semantics."""
        if not isinstance(spec, JobSpec):
            raise TokenError("submit requires a JobSpec")
        if cost_usd < 0:
            raise TokenError("cost_usd must be non-negative")
        now = now_ts if now_ts is not None else self.clock()
        prior = self._decisions.get(spec.token)
        if prior is not None:
            # terminal state per token: duplicates replay it, never re-run
            if now <= spec.expires_at():
                return JobDecision(prior.token, DUPLICATE_NOT_REEXECUTED, False,
                                   result=prior.result, error=prior.error,
                                   cost_usd=prior.cost_usd, first_seen_at=prior.first_seen_at)
            if now > (prior.first_seen_at + spec.ttl_s):
                return JobDecision(prior.token, TOKEN_EXPIRED_REJECTED, False,
                                   error="token expired: fresh token required per lifecycle attempt",
                                   first_seen_at=prior.first_seen_at)
        # accepted token: execute at most once, store terminal state
        try:
            result = run()
        except Exception as exc:  # noqa: BLE001 - terminal state, disclosed
            decision = JobDecision(spec.token, CALLBACK_ERROR, True, error=repr(exc),
                                   cost_usd=cost_usd, first_seen_at=now)
            self._decisions[spec.token] = decision
            self._costs.append((spec.token, cost_usd))
            return decision
        decision = JobDecision(spec.token, EXECUTED, True, result=result,
                               cost_usd=cost_usd, first_seen_at=now)
        self._decisions[spec.token] = decision
        self._costs.append((spec.token, cost_usd))
        return decision

    def decision_for(self, token: str) -> Optional[JobDecision]:
        return self._decisions.get(token)

    def total_cost_usd(self) -> float:
        return sum(c for _, c in self._costs)
