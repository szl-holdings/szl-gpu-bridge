# Multi-model swap pool (draft, not authorized to launch)

`tools/gpu/pool_runner.py` keeps one H100/H200-class node busy with several
reviewed runs at once. When a run finishes, plateaus, reaches its cost cap or
its time box, the pool checkpoints it, uploads the checkpoint to R2, writes a
receipt, evicts it, and loads the next admissible manifest. This file is a
design and operator note; it is **not** a launch approval. Every real launch
still needs the owner's named yes, the per-run reviewed binding and the
existing dispatch path.

## What it launches

Each run is a separate process started through the existing entrypoint only:

```text
python laptop/dispatcher.py queue/pending/<signed-envelope>.json
```

DSSE verification, the engine-key pin, the per-kind contract and the
`SZL_EXECUTION_BRIDGE_REVISION` binding still run inside that child. The pool
adds gates in front; it never replaces one. A child that exits `3` (dispatcher
refusal) is recorded as `child_gate_refused`.

## Gates (any failure means no launch)

| Gate | Refusal code examples |
| --- | --- |
| Node provider is `salad` or `massed`, never a laptop/owner host | `NODE_PROVIDER_NOT_ALLOWED`, `NODE_FORBIDDEN_HOST` |
| Node deadline has an offset and is in the future; margin > checkpoint timeout + terminate grace | `NODE_DEADLINE_PASSED`, `DEADLINE_MARGIN_BELOW_CHECKPOINT_TIMEOUT` |
| Periodic checkpoint interval at most 4 h | `CHECKPOINT_INTERVAL_ABOVE_4H` |
| Shared-GPU placement needs explicit opt-in (memory fraction is not enforced) | `SHARED_GPU_MEMORY_UNENFORCED` |
| Spend ledger exists and parses | `SPEND_LEDGER_MISSING`, `SPEND_LEDGER_INVALID` |
| Entrypoint is exactly `laptop/dispatcher.py` | `RUN_ENTRYPOINT_NOT_DISPATCHER` |
| Jobspec is a `jobspecs/*-reviewed.json`; envelope is in `queue/pending/` | `RUN_JOBSPEC_NOT_REVIEWED`, `RUN_ENVELOPE_NOT_QUEUED` |
| Per-run reviewed binding: both files hash to the pinned sha256, revision equals `SZL_EXECUTION_BRIDGE_REVISION` | `RUN_BINDING_MISSING`, `RUN_ENVELOPE_DRIFT`, `RUN_EXECUTION_REVISION_MISMATCH` |
| Worst-case projected cost (device rate x hard time box) fits every cap | `PROJECTED_COST_ABOVE_RUN_CAP`, `..._DAY_CAP`, `..._MONTH_CAP` |
| R2 credentials present before `--execute` | `R2_CREDENTIALS_MISSING` |

## Caps

No cap value is hardcoded. Caps are read from the node manifest `caps`, every
`--dispatch-caps` file (`szl.dispatch.org-caps/v1`) and an optional per-run
`caps` block. For each of `per_run_usd`, `per_day_usd` (America/New_York day)
and `per_month_usd` the **stricter** (lowest) configured value wins; a
dimension with no source refuses the launch. Day and month checks count the
spend ledger plus the worst-case reservations of runs already on the node.
While running, the pool evicts a run whose estimated spend reaches its per-run
cap and stops the whole node when day or month spend would reach its cap.

## Child contract

The pool sets, per child: `CUDA_VISIBLE_DEVICES` (one device index or one MIG
UUID), `SZL_POOL_RUN_ID`, `SZL_POOL_RUN_DIR`, `SZL_POOL_METRICS_FILE`,
`SZL_POOL_CHECKPOINT_REQUEST`, `SZL_POOL_CHECKPOINT_DIR`, and for shared
placement `SZL_POOL_CUDA_MEMORY_FRACTION`. `R2_*` and `AWS_*` variables are
removed from the child environment.

* Metrics: append `{"eval_loss": <float>, "step": <int>}` lines to the metrics
  file. Plateau = the best of the last `k` evals improved on the best before
  them by less than `threshold` (per manifest `plateau: {k, threshold,
  min_evals}`).
* Checkpoint: when the request file appears (`{"seq", "label", "dir"}`), save
  into `dir` and create `dir/COMPLETE`. If no acknowledgment arrives within
  `checkpoint_timeout_s`, the pool falls back to the newest completed or
  `checkpoint-*` directory on disk; with nothing on disk a periodic checkpoint
  failure evicts the run (`checkpoint_failed`).

## Swap log line

One JSON line per eviction, on stdout and in `<work-dir>/swaps.jsonl`:

```json
{"schema": "szl.gpu-pool.swap/v1", "event": "swap", "run_id": "...", "model": "...",
 "at_et": "2026-11-27T03:14:15-05:00", "gpu_hours": 3.91, "reason": "plateau",
 "checkpoint_uri": "r2://szl-mesh-artifacts/pool/<run>/0003-plateau.tar",
 "checkpoint_error": null, "device": "0", "usd_estimate": 7.8,
 "next_run_id": "...", "next_model": "..."}
```

`admit` and periodic `checkpoint` events use the same schema.

## Node manifest (`szl.gpu-pool.node/v1`)

```json
{"schema": "szl.gpu-pool.node/v1", "node_id": "massed-h100-01", "provider": "massed",
 "gpu_type": "H100-80GB", "node_deadline": "2026-11-28T00:00:00-05:00",
 "deadline_margin_s": 1800, "checkpoint_timeout_s": 900, "terminate_grace_s": 60,
 "checkpoint_interval_s": 14400, "vram_headroom_fraction": 0.1, "max_concurrent": 4,
 "devices": [{"id": "0", "vram_gb": 80, "isolation": "device", "usd_per_hour": 2.0}],
 "caps": {"per_run_usd": 100, "per_day_usd": 500, "per_month_usd": 5000},
 "spend_ledger": "/srv/szl/spend.jsonl",
 "r2": {"bucket": "szl-mesh-artifacts", "prefix": "pool"}}
```

The queue (`szl.gpu-pool.queue/v1`) lists runs with `run_id`, `model`,
`entrypoint`, `reviewed_jobspec`, `signed_envelope`, `reviewed_binding`
(`jobspec_sha256`, `envelope_sha256`, `execution_bridge_revision`), `vram_gb`,
`max_hours`, `plateau` and optional `output_dir` and `caps`.

```bash
python tools/gpu/pool_runner.py --node node.json --queue queue.json --work-dir /srv/pool \
  --dispatch-caps caps.json            # plan only: prints admission decisions
python -m unittest discover -s tests -p test_gpu_pool_runner.py -v
```

## Known limitations

* **Memory isolation.** Only `mig` (one run per MIG instance) and `device` (one
  run per whole GPU) are isolated by the hardware/driver. `shared` placement
  passes a memory fraction the trainer must apply itself; nothing enforces it
  today, so it is refused unless the node manifest sets
  `allow_unenforced_shared_memory: true`. VRAM fit uses the manifest estimate
  plus headroom, not live `nvidia-smi` readings.
* **Trainer support.** The existing trainers do not yet write the metrics file
  or honor the checkpoint-request file. Until they do, plateau detection has no
  input and eviction checkpoints fall back to `checkpoint-*` directories already
  on disk. That trainer work is separate from this PR.
* **Cost is an estimate.** Spend is device `usd_per_hour` x wall time, charged at
  the full device rate even for shared placement (conservative). It is not a
  provider bill and does not include storage or egress.
* **Blocking checkpoint waits.** An eviction waits up to `checkpoint_timeout_s`
  for its checkpoint; other runs are not polled during that wait.
* **Cap refusals are final for this pool session.** A run refused on a day or
  month cap is not retried later in the same session.
* **R2 helper is new.** The repo had no R2 client, so `tools/gpu/r2_checkpoint_store.py`
  adds one (boto3 S3 API, env-only credentials, size and sha256 read-back).
  boto3 is not in the training image lock yet.
