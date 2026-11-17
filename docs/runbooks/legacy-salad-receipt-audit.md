# Historical Salad receipt interpretation

This offline tool answers a narrower question than a new model evaluation: what
can the archived v1 receipt and its archived source actually support? It does not
launch compute, change cloud resources, qualify quality, or authorize promotion.
It recognizes three exact archived source fingerprints. Unknown bytes remain
UNAVAILABLE rather than acquiring a guessed interpretation.

## Preserve first

Obtain the result and source at one immutable Hub revision. Keep their original
bytes and retain the repository/revision externally in the export manifest. The
auditor hashes both inputs, rejects duplicate/non-finite JSON and inconsistent
pass flags, and creates a new output exclusively. It never edits old results.

```sh
python tools/salad/legacy_receipt_audit.py \
  --receipt /evidence/original/result.json \
  --source /evidence/original/heldout_eval.py \
  --output /evidence/new/heldout-audit.json
```

Use the corresponding archived `bench_eval.py` for the benchmark receipt, or
`proving_ground_eval.py` for the proving-ground receipt.
Exit 0 means an audit completed, NOT that a model qualified. Exit 3 means the
source is unknown. Exit 2 means invalid inputs or output creation failed.
Consumers must inspect `quality_qualification`, which is NOT_ESTABLISHED.

## What the recognized historical sources do not establish

The held-out implementation attaches a PEFT adapter to the base instance and then
calls that same base without disabling the injected adapter. Consequently the
reported arms are not established as an independent base/adapter comparison.
Its neutral metric searches the first standalone A-D token anywhere in an
uppercased, at-most-16-token generation. It retains no per-item outputs in the
summary, so a low count cannot distinguish answer errors from format or truncation
errors. A lower-bound rule with an effective threshold at or below zero cannot
reject any rate in [0,1]. Refusal, invented-ID and contamination thresholds can
reject some values; loose thresholds are not automatically mathematically vacuous.

NVML total-energy counters are millijoules. These exact historical sources stored
raw differences under `energy_j`. The audit records a derived divide-by-1000 unit
correction without rewriting the original or calling it a new measurement. The
held-out energy window also ends before its final baseline-refusal pass. Device
energy is not isolated process or whole-node energy.

The contamination task is a first-training-row verbatim-continuation probe, not
independent benchmark-leakage detection. Empty denominators remain UNAVAILABLE.
The benchmark v1 uses input-token rather than shifted-prediction denominators;
its old token/loss fields must not be relabelled as corrected v2 measurements.

The recognized proving-ground source also stored NVML millijoule differences as
`energy_j`. Its theorem/lemma/example keyword classifier never invokes Lean, so
`SKETCH_CANDIDATE` and `sorry` absence establish no kernel-checked theorem. The
audit reconciles recorded verdict counts and reports any mismatch between actual
rows and the declared unproven count. It does not verify row-chain authenticity,
qualify mathematical correctness, or change any quality threshold.

## Next evaluation boundary

The current `bench_eval.py` and `bench_launch.py` implement the separate corrected
benchmark contract. Their tests do not repair historical measurements. A new
held-out run additionally needs independently isolated arms, immutable fixture and
source pins, per-item outputs, termination/format status, and criteria agreed before
execution with failing controls. Do not tune criteria to make prior observations
pass. Do not deploy a draft source branch or bypass the controller's admission.

The fingerprints identify archived bytes, not a signature or an execution witness.
A counterparty export must contain actual source/result bytes, not hashes alone.
Private artifacts stay private unless the owner explicitly approves their release.

Validation:

```sh
python -m unittest discover -s tests -p 'test_legacy_receipt_audit.py' -v
python -m unittest discover -s tests -p 'test_salad_bench*.py' -v
```

Primary API references:
- NVIDIA NVML: https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
- PEFT model adapter-disable context: https://huggingface.co/docs/peft/en/package_reference/peft_model
