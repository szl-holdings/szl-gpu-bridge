# Finalized transfer observation, not revenue attestation

This first-party read-only watcher replaces a local polling pattern that advanced
its cursor after failed eth_getLogs calls and after exhausting its range budget.
It is not a payment client: it cannot sign, submit transactions, create cloud jobs,
change credentials, or prove that a transfer is an organic x402 settlement.

The operator supplies a private config with exactly rpc_url (HTTPS without embedded
credentials), chain_id, token_address, recipient_topic (32-byte padded address),
start_block, chunk_size (1..5000), max_chunks (1..24), max_polls (1..2880), and
poll_seconds (300..3600). No real wallet or service secrets are in this source.
The replay floor is a declared scope boundary, never an inferred verified history.
Legacy earlier history remains UNVERIFIED. Preserve the original script, status and
logs as immutable local evidence before migration. A previously fired event requires
owner review rather than resetting the flag.

Run with --state-dir PATH for one bounded pass, or --loop for the existing authorized
watch's remaining poll budget. Preserve its cumulative poll count. Do not create a
second scheduler or compete with another active watcher. An OS-held single-writer
lock is required. It releases on process exit; a PID file alone grants no authority.

Every pass checks chain identity, the previous block-hash anchor and the provider's
finalized head. Requests use the exact Transfer signature and recipient. Failed,
malformed, removed, contradictory or out-of-range logs do not advance coverage.
A capped scan retains the next unscanned block, instead of jumping to the chain tip.
A durable, byte-verified range receipt precedes each atomic cursor update. Restarts
may replay already-receipted ranges after a disk interruption; historical receipts
are not deleted. No empty response is extrapolated beyond its verified range.

There are no HTTP retries or alternate providers. 401/403/429 stop and persist an
access/rate-limit hold; any other failure also exits for review. Each pass has bounded
request count, duration and response bytes. A successful positive-value transfer is
an attribution candidate and stops the watch; it is never called revenue. An RPC
provider's finalized block is not an independent witness or a financial attestation.
Amounts remain raw integer token units: no decimals, fiat prices or ownership claims
are inferred. Zero-value transfers are retained but do not trigger a revenue candidate.

Acceptance: offline regression tests, normal exact-head source admission, no existing
legacy/replacement process, exact-byte installed source, one successful live pass,
and an independently re-read receipt/cursor before reporting restored coverage.
An active PID by itself is insufficient. The original history and all current holds
must be retained. This watcher does not repair a payment gateway or earn revenue.

Primary API contracts:
- https://ethereum.github.io/execution-apis/api/methods/eth_getLogs/
- https://ethereum.github.io/execution-apis/api/methods/eth_getBlockByNumber/

Rollback: stop only the positively identified replacement through its owner. Preserve
state and receipts; do not silently restore cursor-skipping or revenue-overclaim logic.
