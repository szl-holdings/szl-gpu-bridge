# szl — the receipts layer for governed AI

Install the current source from the repository root with
`python -m pip install ./sdk`. PyPI releases are separate immutable artifacts;
check the installed version with `python -c "import szl; print(szl.__version__)"`.

```python
import os
from szl import khipu, kamachikuna, provenance, router

secret = os.environ["SZL_RECEIPT_SECRET"]  # configured by the trusted application

# 1. Every claim carries a receipt (Law 7)
r = khipu.mint({"claim": "training completed", "steps": 2000}, secret)
assert khipu.check(r, secret)            # integrity + origin
r2 = khipu.mint({"claim": "eval done"}, secret, prev_body_sha256=r["body_sha256"])
assert khipu.check_chain([r, r2], secret, expected_head=r2["body_sha256"])

# 2. Fail closed (Law 6)
assert kamachikuna.gate("deploy", verified=False)[1] == "BLOCKED"

# 3. Evidence must bind the exact model, source revision and dataset hashes.
# Illustrative hashes below must be replaced with measured artifact digests.
evidence = {"model_id": "m", "source_sha": "a" * 40,
            "datasets": {"train": "b" * 64}}
training = khipu.mint({"kind": "szl.training-receipt/v1", **evidence}, secret)
c = provenance.ProvenanceClaim(**evidence, training_receipt=training)
badge = c.badge(secret)   # MEASURED means bound signed evidence, not factual truth

# 4. Two-axis Truth Router — honest cheapest, joules reported separately
models = [router.Model("m", "UNMEASURED", None, None, None, 1.0)]
pick = router.route("honesty-critical", models)  # None if no honest model: fail closed
```

## Modules
- **khipu** — signed, hash-chained receipts. Full-length HMAC-SHA256 signatures (never truncated — that was the v1 auditor bug). Tamper, reorder, and truncation detection (truncation via `expected_head`).
- **kamachikuna** — the constitution: 10 laws, 5 Prime Principles, 5 Lean-proven invariants, 3 amendments, and Law 6 as a callable gate.
- **provenance** — honest states only: UNTRAINED · CARD_ONLY · MEASURED · UNAVAILABLE · NOT_CLAIMED. Verification proves integrity and origin, never accuracy or performance.
- **router** — two-axis honesty routing: price axis decides, joules reported separately (never summed with dollars — different units), UNMEASURED energy never counts as zero, honesty-critical tasks fail closed.
- **openshell** — declarative policy and a trusted-caller admission helper; see the authorization boundary below.

## Authorization boundary

`kamachikuna.gate` and `openshell.admission_gate` consume a trusted verification
result. They do **not** authenticate callers, verify receipt signatures, bind
receipts to requested actions, prevent replay, or authorize side effects.
Never expose either helper directly to an untrusted request's `verified` field.
The application must derive a literal boolean from its own verification and
authorization policy. A well-formed receipt hash is only an evidence reference.
Full action-bound cryptographic authorization remains required before using
this SDK as a production enforcement boundary.

Only `verified is True` can allow. Empty or malformed receipt references block;
references may use `sha256` or the khipu `body_sha256`, with matching values if
both are present. The built-in irreversible action set always blocks, including
when a caller supplies `irreversible=False`. A caller may tighten policy using
`irreversible=True`, but cannot relax that set.

Receipt verification proves possession of a shared HMAC secret and integrity,
not the signer's independent identity, factual accuracy, authority, or model
performance. Keep signing secrets outside untrusted request bodies.

The router likewise ranks trusted caller assertions. Numeric validation and a
`MEASURED` label do not authenticate badges, provenance, or freshness. The
application must verify that evidence before supplying measurements. Known
`FAILED` models and unknown state labels are ineligible for every task;
`UNMEASURED` models remain eligible only for non-critical fallback routing.

## The freshness rule
> "A line older than seven days is a claim, not a fact." — Vidal-Gadea Lab roadmap, Nov 2026

`khipu.check_chain(..., max_age_days=7)` fails a chain whose newest receipt is
older than the window, an empty chain, or an invalid freshness window.
A stale record that still verifies is worse than no
record, because it is believed — their lab's cached tracker sat 57 days untouched
and was read by a program with no error and no warning.

## Design debts deliberately not hidden
- Chain prefixes are internally valid; completeness requires `expected_head` (tamper-evidence ≠ completeness).
- An unmeasured quantity is None/UNKNOWN everywhere — never zero.
- Freshness is a separate axis from integrity: a chain can be both valid and stale.

## Release validation

From the repository root:

```sh
python -m pip install build twine
python -m build sdk --outdir /tmp/szl-dist
python sdk/tools/verify_release.py --dist /tmp/szl-dist --tag szl-v0.1.7
python -m twine check --strict /tmp/szl-dist/*
python -m pip install --no-deps --force-reinstall /tmp/szl-dist/*.whl
python -m unittest discover -s sdk/tests -v
```

Use an empty output directory. CI installs the wheel and runs tests outside
the source tree on Python 3.10 and 3.12. Release validation also rebuilds the
sdist and tests that installation. The release workflow publishes only after
all checks pass and the tag exactly matches the package version. Branch
dispatches validate without publishing. A release tag must resolve to the
checked-out commit and that commit must belong to fetched `origin/main` history.
PyPI must have the matching trusted
publisher registered; no token fallback is provided.

Version 0.1.7 is a source release candidate until the publish workflow succeeds.
Earlier `szl-v0.1.5` tags did not change the 0.1.0 package metadata; they are
not proof that 0.1.5 exists on PyPI. The conda-forge recipe remains a historical
0.1.0 recipe and must be separately updated with a verified published sdist hash.

Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.
