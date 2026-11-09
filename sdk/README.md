# szl — the receipts layer for governed AI

```python
import sys; sys.path.insert(0, ".")  # until published: pip install szl
from szl import khipu, kamachikuna, provenance, router

# 1. Every claim carries a receipt (Law 7)
r = khipu.mint({"claim": "training completed", "steps": 2000}, secret="…")
assert khipu.check(r, secret="…")            # integrity + origin
r2 = khipu.mint({"claim": "eval done"}, secret="…", prev_body_sha256=r["body_sha256"])
assert khipu.check_chain([r, r2], "…", expected_head=r2["body_sha256"])

# 2. Fail closed (Law 6)
assert kamachikuna.gate("deploy", verified=False)[1] == "BLOCKED"

# 3. Provenance badge — the Korea/Naver bar, machine-checked
c = provenance.ProvenanceClaim(model_id="m", source_sha="…", training_receipt=r,
                                datasets={"train": "sha256…"})
badge = c.badge(secret="…")   # state: MEASURED | CARD_ONLY | UNTRAINED | UNAVAILABLE

# 4. Two-axis Truth Router — honest cheapest, joules reported separately
pick = router.route("honesty-critical", models)  # None if no honest model: fail closed
```

## Modules
- **khipu** — signed, hash-chained receipts. Full-length HMAC-SHA256 signatures (never truncated — that was the v1 auditor bug). Tamper, reorder, and truncation detection (truncation via `expected_head`).
- **kamachikuna** — the constitution: 10 laws, 5 Prime Principles, 5 Lean-proven invariants, 3 amendments, and Law 6 as a callable gate.
- **provenance** — honest states only: UNTRAINED · CARD_ONLY · MEASURED · UNAVAILABLE · NOT_CLAIMED. Verification proves integrity and origin, never accuracy or performance.
- **router** — two-axis honesty routing: price axis decides, joules reported separately (never summed with dollars — different units), UNMEASURED energy never counts as zero, honesty-critical tasks fail closed.

## Design debts deliberately not hidden
- Chain prefixes are internally valid; completeness requires `expected_head` (tamper-evidence ≠ completeness).
- An unmeasured quantity is None/UNKNOWN everywhere — never zero.

Tests: `python3 sdk/tests/test_szl_sdk.py` — 11/11 offline, no network, no GPU.

Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.
