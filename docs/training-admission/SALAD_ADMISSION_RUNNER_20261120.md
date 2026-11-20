# Salad Admission Runner v1 (single-writer executor core) - 2026-11-20

Companion to SALAD_ADMISSION_CONTROLLER_20261120.md and the merged gate
(tools/salad/admission.py, PR #307, squash 58303378). The gate maps one
intent to an ADMIT or DENY receipt; tools/salad/admission_runner.py is
the fail-closed executor that owns the queue, the execution ledger and,
one day, the credential seam.

## Status in the estate (honest)

The Salad API key 401s estate-wide (owner mint pending). The runner is
deliberately operational WITHOUT any credential:

* NO_CREDENTIAL mode (today): admitted intents record PENDING_EXECUTION
  ledger entries. The owner-pinned registry advances only its intent
  chain head; desired_state and budget are untouched.
* When the owner lands the fresh key, the one supervised process the
  owner runs constructs an applier closure holding the credential and
  replays the pending entries exactly once, in ledger order.

## Review provenance

Design brief reviewed 2026-11-20 by Mistral Large 4 (32.5s) and Mistral
Nemo 12b (63.6s); verdicts CONCERN with concrete amendments, all adopted:

1. INTENT OUTBOX ordering: a persisted EXECUTING ledger entry is written
   BEFORE the mutation and EXECUTED only after success. A crash between
   the two leaves CRASH_AMBIGUOUS, which scan_ledger refuses to run on;
   a human resolves it. The runner never guesses or auto-repairs.
2. LEDGER-SCAN IDEMPOTENCE: replay-exactly-once derives from the
   hash-chained ledger itself (gates.verify_chain over ledger_lines),
   so it is verifiable offline without trusting the runner.
3. CREDENTIAL BOUNDARY: exactly one supervised process may construct
   the applier closure. The runner core performs no Salad I/O, reads no
   secrets and never widens the key surface; env keys visible to more
   than one process weaken the single-writer moat and are forbidden.
4. Fail-closed halts: LISTED reason codes (REGISTRY_INVALID,
   EXECUTION_FAILED, SINK_FAILED, RECEIPT_CHAIN_SEED_INVALID,
   CLOCK_UNAVAILABLE); halts are terminal for the queue, never retried
   in-process (a fresh intent is required).

The ChatGPT rail was requested for this round and is recorded BLOCKED:
the desktop app UIA tree offered no reachable composer (settings pane
trap) and the account plan shows a 0 percent weekly allowance until
reset; no fabricated consultation occurred.

## Adversarial guarantees (tests/test_salad_admission_runner.py, 19)

* re-queued intent after execution: ALREADY_RECORDED, never re-executed;
* two-runner race: the second sees the executed sha in the ledger and
  records ALREADY_RECORDED without touching the applier;
* key arriving mid-replay: pending entries execute in ledger order,
  exactly once, with state and budget re-checked before each apply;
* registry edited mid-queue: REGISTRY_INVALID deny and halt;
* STATE_DRIFT and BUDGET_SHORTFALL at execution time deny fail-closed;
* tampered, duplicated or crash-ambiguous ledgers refuse to run;
* DENY receipts never advance the intent chain head (no queue poison);
* the naive local clock (no tzinfo) refuses to run.

## Rough edges kept explicit

* The ledger chain is tamper-evidence, NOT authentication (signing lands
  with szl-evidence-core next).
* The cross-process file lock for the ONE supervised process is a
  deployment concern, deliberately out of this PR; two processes must
  never be started against one ledger and one credential.
* An ADMIT receipt with sink persistence is not proof the sink is
  durable; the sink is operator-supplied and its durability contract is
  the operator's to verify (readback-or-it-did-not-happen).
