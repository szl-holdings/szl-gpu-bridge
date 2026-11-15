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
