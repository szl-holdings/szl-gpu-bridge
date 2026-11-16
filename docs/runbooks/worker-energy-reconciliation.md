# Offline worker-energy reconciliation

## Defect and scope

A legacy workstation digest scans arbitrary numeric fields containing `energy`
or `joule`, including nested values, and accepts the first match as measured job
energy. In the worker receipt schema, `energy_start.energy_j` and
`energy_end.energy_j` are cumulative device counters. They are not interval
consumption. Their position before `energy_j.joules` lets the old walker sum
large boot counters instead of the producer's explicit delta. The old walker
also does not enforce measurement labels, receipt hashes, or interval identity.

Do not use that aggregate for billing, training-energy claims, or an estate-wide
measured-energy statement. Preserve historical artifacts; any public correction
must explicitly supersede the exact earlier artifact rather than rewriting it.

`tools/evidence/worker_energy.py` is an **offline, read-only reconciler**, not a
new provider controller or meter. It does not import the training worker, read
credentials, access the network, upload, restart training, mutate model cards,
sign a receipt, or infer training completion. It writes only a new caller-selected
local JSON observation, refusing to overwrite an existing file.

## Admission contract

The reconciler accepts only `kind=szl.salad-worker/v1`. It verifies the producer's
self-hash with the actual `json.dumps(payload, sort_keys=True)` serialization,
excluding `receipt_sha256` exactly as the producer does. Self-hash consistency is
not a signature or proof of an authentic meter.

It then requires the explicit `energy_j` object, the exact supported device-level
NVML cumulative-counter-delta method, MEASURED labels on the delta and both
counter samples, finite nonnegative numeric values (not booleans or strings),
matching nonempty GPU labels, monotonic counters, and agreement with the
producer's one-decimal rounded difference. Zero deltas are retained. Missing
pairs, counter resets, guessed units, unsupported schemas, malformed JSON,
duplicate JSON keys and inconsistent hashes remain classified exclusions.

Repeated receipts with the same verified self-hash count once, including when
whitespace differs. The report binds every examined path, its bytes hash, and
its disposition through an input-manifest hash. File size, input count, path
containment, and output creation are bounded.

## What the output does not prove

`recorded_device_delta_subtotal_joules` is an arithmetic diagnostic across the
accepted unique receipts. It is **not** a certified estate-energy total. A GPU
model name is not a unique physical device identity, and different receipts can
refer to overlapping intervals. Cache freshness and complete estate coverage are
also unproven. Consequently `total_measured_joules` remains null and
`estate_energy_state` remains UNVERIFIED. Empty accepted coverage is null, not
zero. Missing adapter attribution is counted, not invented.

This reconciler does not repair or replace the historical digest on the Hub.
It provides auditable inputs for a separately reviewed correction. A successful
CLI exit means the local reconciliation completed; it is not a production,
training-completion, model-quality, or payment-readiness verdict.

## Run against an existing receipt cache

```sh
python tools/evidence/worker_energy.py \
  --receipt-dir /path/to/existing/receipt-cache \
  --output /path/to/new-energy-reconciliation.json
```

The cache is selected explicitly. No credentials or remote repository access are
required. Preserve the original digest alongside the new observation. Do not
restart a legacy completion publisher merely because a checkpoint directory
reaches the target; terminal state, final artifacts and their run/revision binding
remain separate release requirements.

## Verification

```sh
python -m unittest discover -s tests -p 'test_worker_energy_reconciliation.py' -v
```

The root contract CI discovers these tests automatically. Cases cover cumulative
counter confusion, explicit labels and methods, missing samples, counter resets,
wrong deltas, zero energy, nonfinite/boolean/string values, unsupported schemas,
self-hash tampering, duplicate receipts, overlapping interval uncertainty, empty
coverage, input budgets, and non-overwriting local output. All fixtures are
synthetic; real cached-data replays must retain their own input manifest.

The existing producer is `tools/salad/salad_worker.py` (`receipt`,
`sample_energy_j`, and `energy_delta`); it is not changed by this repair.
