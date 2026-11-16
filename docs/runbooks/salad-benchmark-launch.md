# One-shot benchmark launch contract

This entrypoint replaces mutable-script downloads, automatic source decoding and
shell success tails. It is an integration for the existing sole controller, **not**
a new scheduler, monitor, cloud writer, release authorization or model-quality gate.

## Authority and prerequisites

The controller must admit the exact source revision and harness hash, verify the
image/dependencies and input/data-rights contract, serialize resource-generation
ownership and authorize compute separately. The launcher verifies local bytes and
arguments, not trusted-main ancestry, signature approval or the live cloud lease.
Its plan explicitly reports `source_admission_verified: false`.

Do not deploy a draft branch to bypass admission, read a key from an old script,
copy credentials from a handoff, or revive the retired receipt-presence watcher.

## Private controller contract

Replace every placeholder with an actual admitted value. The template is deliberately
not executable. No secret belongs in it; unknown fields are rejected.

```json
{
  "schema": "szl.salad-bench-launch/v1",
  "attempt_id": "<new unique 32 lowercase hex characters>",
  "source": {"repository": "szl-holdings/szl-gpu-bridge", "revision": "<reviewed 40-character commit>", "sha256": "<64-character harness hash>"},
  "inputs": {
    "base": {"repo_id": "example/base", "revision": "<40-character commit>"},
    "adapter": {"repo_id": "example/adapter", "revision": "<40-character commit>", "directory": "final"},
    "dataset": {"repo_id": "example/dataset", "revision": "<40-character commit>", "eval_file": "data/eval_repo.jsonl", "eval_sha256": "<64-character hash>", "train_file": "data/sft_train.jsonl", "train_sha256": "<64-character hash>"}
  },
  "destination": {"repo_id": "SZLHOLDINGS/frontier-54-runs", "repo_type": "dataset", "prefix": "receipts/bench"},
  "measurement": {"max_length": 2048, "train_ref_rows": 2000, "seed": 1234}
}
```

The canonical dataset destination stays explicit. Its type is not inferred from a
name; an alternative requires separate controller admission. Both split reads, the
adapter and the base use immutable revisions.

## Plan before execution

From an immutable source checkout or admitted image containing both scripts:

```sh
python tools/salad/bench_launch.py --contract /controller/launch.json --work-root /evidence/bench
```

Without `--execute`, this validates and prints the exact argument vector. It creates
no attempt directory and performs no source/weight download, model-stack import,
evaluation, upload, restart or scale operation. Invalid pins, changed source bytes,
missing v2 interface, duplicate/non-finite JSON, traversal and numeric-type errors
fail closed. There is no decoding, install step, credential fallback or provider retry.

Only the controller may add `--execute` after separate admission. Use it as the direct
foreground entrypoint, not a shell ending with `true` or an unchecked `tee` pipeline.
Execution exclusively claims `<work-root>/<attempt_id>/`, writes and flushes the
exact harness snapshot, canonical contract and launch plan, then executes the snapshot.
The benchmark records the same attempt ID, declared source revision and canonical
contract hash; its receipt also binds actual harness bytes. Linux replaces the process
so signals and exit status belong to the benchmark. Windows waits for the exact child
and returns its exit code. Real subprocess tests cover success, nonzero exit and an
uncaught exception with tiny synthetic scripts, never GPU evaluation.

An existing or partial attempt is preserved and rejected, not deleted or resumed.
The direct harness also rejects an existing output bundle before model hydration.
This is local exclusivity, not a cross-node distributed lease or an off-node archive;
the controller still owns fleet concurrency, retries, generation and archive checks.

## Recover publication, not measurements

For a completed ORIGINAL v2 bundle, use the separately admitted harness's
`--publish-only BUNDLE_DIR`. Do not supply a replacement run ID or source binding.
That mode retains the original identity and does not hydrate models or repeat
measurement. Do not reconstruct an original bundle from log summaries or relabel a
legacy v1 receipt as a corrected v2 result.

## Acceptance bounds

`status=OK` means measurements completed, not that acceptance criteria passed.
A failed `criterion_1` stays failed; `heldout_quality_measured` remains false until
separate evaluations exist. NVML scope is GPU-0 device-total, not isolated-process,
whole-node or Zeus cross-verification. Publication needs immutable-commit byte
readback; a plan, local exit, filename, HTTP 200 or RUNNING label is not that proof.
Before resource reclamation the controller must verify exact run/generation, terminal
status, accepted receipt/source bytes, required archives and drained instance counts.
This launcher has no Salad control API calls and cannot waive that gate.

Validation: `python -m unittest discover -s tests -p 'test_salad_bench*.py' -v`.

## Per-item evidence and complete comparison energy

New benchmark receipts declare `szl.salad-bench-integrity/v1`. Every offered row
has an ordered index and SHA-256 of its sorted compact JSON representation, plus
either scored token counts and loss or an explicit skip reason. The receipt does
not include row text, generated answers, or per-token logits. These hashes support
input comparison; they do not anonymize a known dataset or prove independent
execution.

The evaluator measures the base before PEFT attaches the adapter. A CPU regression
uses an attachment that mutates the base object, so evaluating a supposedly bare
base after attachment fails the regression. Base and adapter evidence must identify
the same ordered eval rows, token lengths, and skip outcomes. Publication recomputes
the row-weighted loss and coverage totals and refuses inconsistent results, even if
their file hashes were recomputed. Actual adapter-file bytes have a separate
SHA-256 in addition to the existing immutable repository revision.

The full comparison energy window starts before base evaluation and ends after the
final train-reference adapter pass. It includes time between those passes, including
adapter attachment/weight loading, and excludes initial base-model hydration.
The receipt records "includes_adapter_attachment: true" and
"includes_initial_base_hydration: false" explicitly. Its unit is joules,
derived once from raw NVML millijoule counters. Every component boundary must be
available and nondecreasing. A counter reset between otherwise valid passes makes
the total unavailable. Adding separate pass deltas is not an equivalent full window.
The scope remains GPU-0 device-total, not isolated-process or whole-node energy.
This loss benchmark performs no final baseline-refusal pass; such a pass belongs
to the separate heldout evaluator described below.

Each pass is bounded to 10,000 offered rows before base hydration. Results use
compact JSON with a 16 MiB maximum; archived source and manifest retain their
2 MiB maximum. The per-item bound prevents an unexpectedly large evaluation from
silently omitting evidence. Exceeding a bound fails closed rather than truncating
rows. Original v2 bundles without the new source declaration retain their original
contract. A new source's profile cannot be removed from its receipt during a
publication retry. This source/receipt consistency check is not admission,
signature verification, or an execution attestation.

## Historical heldout evidence remains unqualified

The archived `heldout_eval.py` for `SZLHOLDINGS/salad-frontier-v5-r1` is historical
evidence under its pinned model revision; it is not tracked executable source in
this repository. Its `all_pass` result is not a qualification for promotion.
The audit in `tools/salad/legacy_receipt_audit.py` identifies baseline contamination,
a vacuous neutral bound, missing per-item responses, incomplete energy coverage,
and missing immutable bindings. This benchmark does not repair those historical
results or satisfy heldout-quality criteria.

A separately admitted replacement must define its acceptance policy before scoring,
use independent benchmark data with immutable source/model/adapter/dataset bindings,
and either evaluate a verified bare base before attachment or use a tested
adapter-disable context for every base pass. It must preserve per-item answer,
parse/truncation outcome, and score evidence in the appropriate controlled output
destination; distinguish format failure from incorrect answers; and include all
comparison passes in the ending energy read. It must reject a threshold that
accepts every possible score, state what its contamination test actually measures,
and record a new attempt and new GPU measurements. This is a specification for
future work, not a claim that a corrected heldout run or independent reproduction
has occurred.

Historical source: [pinned integrity amendment](https://huggingface.co/SZLHOLDINGS/salad-frontier-v5-r1/blob/f84a7ecec15e31429c966e378d3e5b77efd01417/receipts/integrity-amendment-20261116T181433Z/AMENDMENT.md)
and its [input/source hash manifest](https://huggingface.co/SZLHOLDINGS/salad-frontier-v5-r1/blob/f84a7ecec15e31429c966e378d3e5b77efd01417/receipts/integrity-amendment-20261116T181433Z/manifest.json).
These private model records require the owner's existing access.
