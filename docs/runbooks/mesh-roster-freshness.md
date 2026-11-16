# Mesh roster freshness and consolidation gate

The mesh brain preserves known worker inventory but must not preserve a past
successful health verdict. Previously, a worker that disappeared from both
listings or whose heartbeat became unreadable could remain alive indefinitely.
The old timestamp parser also discarded the timezone suffix and accepted future
heartbeats as fresh.

The roster now requires a successfully observed heartbeat in the current cycle,
an exact producer UTC timestamp, and age in [0, 3 * HEARTBEAT_S). It recalculates
age immediately before publishing. Future and malformed timestamps cannot mask
valid observations from the other supported heartbeat repository. Worker entries
remain available for diagnosis. Missing observations appear in
`unobserved_workers`; they do not prove the machine stopped or that its artifacts
were preserved. Existing worker `alive` and `gpu` fields remain compatible; GPU
fields on unobserved workers are last-known data, never current measurements.

The public roster is unsigned liveness data. It does not establish process
health, workload progress, authenticated node identity, resource availability,
model quality, or authority to execute a job. No scheduler, dispatcher, job
allowlist, credentials, provider configuration, worker deployment or runtime
process is changed by this repair.

## Consolidate only after preservation and workload measurements

Use immutable run identity, dataset revision and actual model destination rather
than a historical container-group name when indexing artifacts. Preserve final
weights/config/tokenizer, resumable checkpoint state, source/runtime manifests,
original logs and evaluation outputs before stopping a worker or relinquishing
storage. Keep private data and model artifacts in private destinations; public
receipt publication is a separate explicit permission and evidence gate.

A high memory allocation is not proof of useful compute, and a single low GPU
utilization sample is not proof of waste. Keep inference latency and memory
headroom separate from training throughput. Retain one writer per workload and
one controller per resource; a new observer must not revive stopped jobs or race
an existing start/retry loop. No node retirement or training restart is authorized
by a roster entry alone.

The existing mesh job allowlist contains only GPU state, disk state and node sync.
It is not a distributed training scheduler and does not combine separate GPUs'
VRAM into one memory pool. The current node-sync script uploads selected receipt
JSON; it does not itself archive all model weights or validate every artifact's
rights. Source deployment and live acceptance remain separate from this repair.

## Verification

`python -m unittest discover -s tests -p 'test_mesh*.py' -v`

Deterministic synthetic frames cover vanished workers, failed listings and reads,
retained inventory, exact expiry, recovery, independent healthy peers, future
clock skew and malformed timestamps. They perform no provider requests or jobs.
