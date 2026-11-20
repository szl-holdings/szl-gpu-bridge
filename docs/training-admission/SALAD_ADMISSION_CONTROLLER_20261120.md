# Salad Admission Controller (SAC) - single-writer gate for Salad mutations

Recorded 2026-11-20. Provenance: the task-21 dream round on "make SZL training on
Salad fully operational, laptop = control room only" (Kimi kimi-k3.1-pro:cloud
02:39Z, Mistral Large 4 02:53Z, Mistral Nemo; receipts in the estate wave-logs,
commit 6660a3a4). All three brains converged on one rule:

    Exactly ONE writer to the Salad API. Every other actor - human, fleet
    doctor, watchdog, agent - may only propose a hash-chained intent.

This matched the measured incident history: the retired ungated controllers
(fleet_doctor_v2, kimi_fleet_doctor_v311, the chaperone loops; halt receipt
2026-11-20 02:20-02:33Z) could each request Salad mutations off weak evidence
(adapter filenames as completion, unknown statuses as stopped).

## v1 scope (this PR)

tools/salad/admission.py - a pure, fail-closed gate. No I/O, no credentials,
never touches Salad. It composes three existing components and adds nothing
weaker than any of them:

| Component | Role in the gate |
| --- | --- |
| controller_gates.verify_owner_lease | only the owner-pinned lease, at the pinned generation, naming the group, unexpired (lease_ok argument) |
| controller_gates.decide_adopted_lane | fleet eligibility; at most RESTART_PERMITTED; quarantined/terminal/unreconciled lanes never admit |
| restart_lineage_witness.witness | restart lineage evidence; HOLD is final; REVIEW_CANDIDATE_ONLY is necessary, never sufficient |

New, owner-pinned state: a desired-state registry
(szl.desired-state-registry/v1): per lane - group, desired state
(NEW/RUNNING/STOPPED/PARKED/DONE), actor allowlist, and the budget formula
fields (start / replay_step / next_checkpoint / finalize / reserve, micro-USD).
The registry head (last_intent_sha256) serializes proposals: only the chain
head is judged, so a replayed or stale intent cannot reorder mutations.

Ops: create (only from NEW, requires a fresh-run witness), restart (only
from STOPPED, requires the witness + fleet gates + lease + budget covering
conditional replay steps), stop (only from RUNNING), park (from RUNNING or
STOPPED). DONE and PARKED lanes are terminal for intents: the owner edits the
registry to unpark or re-open a completed lane, never an agent.

Every judgement - ADMIT or DENY - produces a hash-chained receipt
(szl.salad-admission/v1, signature_status NOT_SIGNED in v1). Admission
authorizes only the single controller replica that holds the credential; it is
not execution proof and not a post-mutation readback, both of which stay with
that caller.

## Not in v1 (explicitly, next increments)

* Intent signing with szl-evidence-core (v1 pins actor allowlists + hash
  chain, which is tamper-evidence, not authentication).
* The Salad HTTP executor that consumes ADMIT receipts and performs the
  post-mutation readback (single replica, CPU-only, sole credential holder).
* x402 conditional payment for verified milestones.
* Registry file hosting + reconciler wiring (the read-only reconciler
  already exists as tools/salad/fleet_watch_readonly.py).

## Tests

tests/test_salad_admission.py - 22 offline adversarial controls, all green
locally (2026-11-20): registry drift, intent parse/byte budgets, duplicate JSON
keys, chain-head serialization, terminal/parked/state-op table, witness HOLD
never admits (the fleet-doctor regression), fleet gate not permitted, budget
shortfall (one micro short denies, exact cover admits), receipt tamper
detection, naive clock, and ADMIT-is-not-attestation.
