# Training scale-down evidence preflight (source-only)

**Review-only, not a controller.** `tools/salad/stop_preflight.py` is an offline, pure policy witness for a *caller-supplied* proposed stop of a non-Flagship54 training lane. It makes no API calls, reads no keys, imports no model code, writes no files, starts no scheduler, and never authorizes or performs a stop. Flagship54's Gate-1 service remains its **only** shutdown owner. The source is not deployed into any local watcher.

## Why this exists

An operator may observe `HTTP 404` from the Hub model metadata endpoint and separately receive an HTTP 200 for a SaladCloud replica-zero PATCH. Neither is proof of missing weights, intact backups, stopped instances, or final training completion. A watchdog deadline, green CI, `RUNNING` label or a filename under `checkpoint-*`/`final/` must never turn missing evidence into shutdown authority. See the existing [post-transition witness](salad-provider-transition-witness.md), [training snapshot](training-snapshot-evidence.md) and [Flagship54 completion guard](flagship54-completion-guard.md).

## Contract

Call `assess_stop_preflight(intent=..., evidence=..., now_utc='YYYY-MM-DDTHH:MM:SSZ')` with bounded JSON-compatible dicts. An `intent` contains `schema='szl.training-stop-preflight/v1'`, a generic group ID, opaque SHA-256 owner-lease digest, positive owner generation, exact 40-hex source/base/dataset/model revisions, and positive `target_steps`. The `evidence` must match those pins and carry a same-host UTC observed time, explicit HTTP metadata 200, pinned byte-verified trainer state with exact target, declared byte-verified final adapter/config/tokenizer, independently read-back backup of the same adapter, and a fresh group-summary GET **plus a separately timestamped instance-list GET**, with no pending change. The caller must normalize every instance to `{instance_id: <unique stable ID>, state: <documented state>}` before submission, preserving the source responses in the private evidence boundary. The two GET timestamps must differ by no more than 60 seconds, and both must be fresh against a same-host UTC clock. Group status and desired replicas are not substitutes for the independent instance list. Four group counters must match the independently observed allocating/creating/running/stopping states. `downloading` cannot be silently coerced into creating because the group API has no matching summary counter; mismatches, duplicate/missing IDs, unknown states and already-drained groups return `HOLD`. Even matched counts are a **non-atomic readback**, not trainer liveness or permission to stop.

This utility **does not validate** the truth or authentication of a caller's `bytes_verified`, `independent_readback`, owner lease or receipt digest assertions. The source owner must obtain these from independently audited upstream verifiers, immutable Hub revisions, backup objects, and the current authorized provider SDK. Each assertion is an *input claim*, not certification by this function. Its best result is `READY_FOR_OWNER_REVIEW`, with `shutdown_authorized=false` and `training_complete_verified=false`. The existing owner must separately perform permission, locking, live process identity, budget, audit signatures, any quality gate and actual pre/post mutation checks. A call to this function is not a substitute for those requirements.

`HOLD` on unknown, stale, conflicting, or inaccessible evidence is mandatory. HTTP 401/403/429 additionally sets `access_stop_required=true`; no alternate credentials, endpoints, retry loops, or fallback readers are authorized. `HTTP 404` is **not** proof of deletion or grounds for scale-down. An existing ambiguous PATCH must be reconciled by current GET before any retry; check desired plus all allocating/creating/running/stopping actual instances. The producer's own clock must be considered when assessing freshness.

## Verification

Run outside BETTERWITHAGE's laptop resource boundary (or in normal source-owner CI):

```sh
python -m unittest discover -s tests -p 'test_stop_preflight.py' -v
```

These synthetic tests include HTTP 404, 401/403/429, stale clocks, owner-epoch drift, trainer shortfall, artifact and independent-backup mismatch, pending provider updates, stale or mismatched independent instance GETs, duplicate identities, downloading states, non-atomic timing, count disagreements, already-drained groups, malformed typed counts and privacy-preserving output. This is source-contract testing, **not** a live cloud observation or a verified model completion. Any future controller integration requires an explicit owner-controlled PR, failure-injection tests, receipt readback and deployment approval. Do not transplant it into an active ad-hoc watcher.
