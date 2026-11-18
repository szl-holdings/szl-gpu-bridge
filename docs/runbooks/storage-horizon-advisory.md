# Proof-carrying capacity horizon — additive, source-only research

**Not deployed.** This isolated Python advisory reads only independently
created v1 workstation-storage-preflight JSON receipts. It cannot launch a
trainer, set SaladCloud replicas, alter a watchdog or Flagship54, authorize a
release, delete files, configure services, access credentials or satisfy the
**verified training-dispatch owner** gate. It is not an alternative dispatcher.

## Motivation and evidence boundary

A single five-sample, ten-second storage gate catches short-term low capacity
and volatility. It does not expose capacity depletion between sweeps. A local
workstation with 44 GiB free is already **HOLD**, irrespective of successful
CI, a remote SaladCloud replica or a simulated rate of change.

The optional second lens in `tools/evidence/storage_horizon.py` requires **3–24
existing source-version-labeled receipts** in UTC time order. Each is checked
against the actual source schema and recalculated from 5–12 independently
time-separated observations, fixed policy (50 GiB, 2 GiB swing) and consistent
volume accounting. Source verdicts cannot invent a PASS. The history must span
at least 15 minutes, have no >2-hour gap, and end within 5 minutes of evaluation;
otherwise it returns HOLD/UNOBSERVED.

Let `f_i` be each receipt's minimum free bytes and `t_i` its last sample UTC.
The **observed** worst depletion slope is:

```
r_observed = max(0, max_{i<j} (f_i - f_j) / (t_j - t_i))  [bytes/sec]
H = 4 hours
f_hypothetical = f_latest - ceil(r_observed * H) - 2 GiB
```

If `f_hypothetical < 50 GiB`, the advisory adds a planning HOLD. The 2 GiB
buffer is a **policy assumption**, not a statistical variance estimate. The
slope is an observed statistic: an adversarial workload or changed traffic may
consume space much faster. **Never treat the projection as a future bound or
SLA**. Missing windows/ownership/time/identity or a currently failing capacity
window cannot be turned green with forecast math.

## Reproduction (read-only, private receipts)

```
python -m unittest tests.test_storage_horizon -v
python tools/evidence/storage_horizon.py --receipt <private-older-1.json> \
  --receipt <private-older-2.json> --receipt <private-current.json>
```

The CLI prints only aggregate numbers and status; it does not reveal receipt
paths, names, credentials or raw samples. It returns exit 78 on HOLD. Receipt
SHA-256 chaining commits to the **provided bytes** but is NOT a digital signature
or provenance guarantee. No receipt is created or changed by the advisory.

## Required next gates, in order

1. Establish the actual owner of the bridge/dispatcher without installing a
   second task. Verify durable signed leases and existing watchdog ownership.
2. Obtain independently authenticated SaladCloud project/group/instance coverage
   and live trainer steps/heartbeat/checkpoint bytes; preserve currently running
   jobs and immutable source pin. Container `running` is not trainer liveness.
3. Only when operator-approved and per-job budgets exist, separately evaluate
   each eligible pool. Massed inference and interruptible Salad training are
   **different** capacity and source/rights domains. No automatic cross-pool
   routing, paid scale, data egress or restart is implemented here.
4. Reverify capacity immediately before each separately admitted dispatch using
   the existing `laptop/storage_preflight.py`. Longitudinal math is only an
   additional advisory, never a substitute for that gate.

**CI tests validate deterministic software contract only — not GPU execution,
live owner liveness, provider capacity, quality or production readiness.**
