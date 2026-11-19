# Fleet controller gates (reviewed replacement for three laptop watchers)

`tools/salad/controller_gates.py` (pure decisions, standard library only) and
`tools/salad/fleet_controller.py` (runner) replace three unreviewed watchers that ran
from `C:\SZL\edge-debug\` on BETTERWITHAGE:
`fleet_doctor_v2.py`, `hub_checkpoint_watch_20261119.py` and
`bench_receipt_watch_20261119.py`. The runner reuses the existing reviewed contracts:
[training snapshot](training-snapshot-evidence.md), `training_observer`,
[artifact qualification](artifact-qualification.md) and the
[provider transition witness](salad-provider-transition-witness.md).

**Source only.** Merging this does not stop, replace or restart any running watcher.
Cutover is a separate, owner-approved step (below).

## Defects fixed (each has a regression test)

| Retired behaviour | Replacement |
| --- | --- |
| `final/adapter_model.safetensors` in a listing = "training complete" | Completion is never claimed. `ARTIFACT_VERIFIED_AWAITING_EVALUATION` only after `artifact_qualification` verifies the bytes at one pinned revision |
| Provider status `None` treated as `stopped`, so a restart fired on an unknown status | Restart requires a reconciled witness: `stopped`, zero counts, empty instance list, replicas equal to the contract |
| Restart with no resume point or owner check | Restart requires a byte-verified `trainer_state.json` at a pinned revision **and** a verified owner lease for that group |
| One quarantined group froze every adopted group | Quarantine is per group and cleared only by the owner |
| Restart budget reset on any non-`stopped` poll | Budget resets only when the verified `global_step` advances; there is a cooldown between restarts |
| Terminal lanes could be restarted (retraining over output) | Terminal or mis-targeted lanes raise `ALERT_TERMINAL_UNQUALIFIED` and are never restarted |
| Any non-empty listing (even `.gitattributes`) = `FIRST_CHECKPOINT_LANDED` | `CHECKPOINT_STATE_VERIFIED` requires byte-verified trainer state at a pinned revision; adapter weights are explicitly *not* verified by this label |
| HTTP 404 skipped silently forever | `UNOBSERVABLE` (never "absent"), with an alert after N consecutive cycles |
| Receipt "uploaded" = published | `PUBLISHED_READBACK_VERIFIED` only if the commit oid is a 40-hex SHA, the path is listed at that commit and the bytes read back identically |
| Lexical cutoff `f >= prefix + "…T0800"` admitted `…-diag-*.json` | Strict `prefix + YYYYMMDDTHHMMSS.json` match with a parsed timestamp |
| Broken-control arm read from `broken_control`/`broken_arm`, keys the receipt never writes | Reads `broken_control_arm` and requires a non-empty `broken_control_failed_rules` plus `battery_admissible` |
| Receipt chosen by filename on a moving branch, never bound to the run | Pinned dataset revision, byte verification against LFS/blob hash, and binding on schema, adapter repo, base, base revision and start ≥ cutoff |
| Bench stopped and scaled down on any receipt, no readback | `PARK_PERMITTED` needs a verified, bound receipt, a reconciled provider and an owner lease; it is followed by a requested-zero witness readback |
| Two controllers could act at once | OS file lock per (state dir, command) |
| Plain local log | Hash-chained JSONL receipts (`verify_chain`). Tamper-evident, **UNSIGNED** |

## Contract (`szl.fleet-controller-contract/v1`)

```json
{
  "schema": "szl.fleet-controller-contract/v1",
  "state_dir": "C:/SZL/fleet-controller",
  "hf_token_file": "C:/SZL/edge-debug/.ht",
  "salad_key_file": "C:/SZL/edge-debug/.sk",
  "owner_lease_file": "C:/SZL/fleet-controller/owner-lease.json",
  "owner_lease_sha256": "<sha256 of the lease file bytes>",
  "owner_generation": 1,
  "receipts_repo": "SZLHOLDINGS/wave-logs",
  "adopted": [
    {"group": "szl-train-provenance-native-r2", "repo": "SZLHOLDINGS/salad-provenance-native-r1",
     "target_steps": 0, "base": "<base model id>", "replicas": 1,
     "max_restarts_without_progress": 3, "restart_cooldown_s": 1800}
  ],
  "checkpoint_lanes": [
    {"lane": "distill-4b-v2", "repo": "SZLHOLDINGS/salad-distill-4b-v2-r1", "target_steps": 0,
     "alert_after_unobservable_cycles": 3}
  ],
  "bench": {"group": "szl-bench-flagship54-v2-h2", "dataset": "SZLHOLDINGS/frontier-54-runs",
            "prefix": "collaboration/vgl-criteria-v2/salad-v5-heldout-v2-",
            "cutoff_utc": "2026-11-19T08:00:00Z",
            "adapter_repo": "SZLHOLDINGS/salad-flagship54-v1",
            "base": "<base model id>", "base_revision": "<40-hex>", "replicas": 1}
}
```

`target_steps: 0` is a placeholder; the observer rejects it, so the lane fails closed (no action). The owner must supply each lane's
real target, base and base revision; this PR does not guess them.

Lease file: `{"generation": 1, "groups": ["…"], "expires_utc": "YYYY-MM-DDTHH:MM:SSZ"}`.
Authority is revoked by editing it, letting it expire, removing a group or bumping the
generation. A Flagship54 **training** group must never be leased here: its Gate-1
service remains that lane's only shutdown owner
([completion guard](flagship54-completion-guard.md)).

## Run

```sh
python -m tools.salad.fleet_controller --contract C doctor              # read-only, 1 cycle
python -m tools.salad.fleet_controller --contract C checkpoints --cycles 144 --interval 600
python -m tools.salad.fleet_controller --contract C bench --apply --cycles 200
```

Without `--apply`, `RESTART_PERMITTED` and `PARK_PERMITTED` are recorded as
`DRY_RUN_WITHOUT_APPLY` and the restart budget is not consumed.

## Cutover (owner-approved, not performed by this PR)

1. Run each command once **without** `--apply` beside the old watcher and compare.
2. Write the lease, pin its digest in the contract, then stop the matching retired
   watcher **before** starting the replacement, so two controllers never act on one group.
3. Start the replacement with `--apply` only for commands whose dry run matched.

## Limits (honest)

- Provider readback is a non-atomic GET pair, not trainer liveness. `RESTART_PERMITTED`
  is eligibility, not proof the trainer will resume.
- `CHECKPOINT_STATE_VERIFIED` covers trainer state only. Weights, backups, evaluation
  quality and promotion remain separate gates.
- Receipts are hash-chained and **UNSIGNED**. Hub publication is verified by readback,
  not by signature.
- HTTP 401/403/429 halts with `access_stop_required`; there is no fallback credential.
