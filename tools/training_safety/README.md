# training_safety: fail-closed Salad controller replacements

Credential-free replacements for the four local Salad training watchers
(`watch_arm6`, `watch_arm7`, `rlvr_watch4`, `f54v2_watch`) and the held-out
eval launcher (`f54v2_heldout_fire3`). They fix four observed flaws:

1. **curl rc is not HTTP status.** The launcher logged `rc=0` next to an HTTP 400
   `pending_update_in_progress` body. Every provider write is now judged by HTTP
   status + body (`classify_write`), reusing the transition witness from #264.
2. **arm6/arm7 stopped a group on a bare `final/` filename.** A `final/` entry is
   no longer evidence of completion.
3. **rlvr_watch4 stopped when `lane.log` existed.** Existence is not completion;
   the owner's completion marker must appear in the log.
4. **f54v2_watch always reported 200.** Status is taken from the response.

## Stop gate

`replicas -> 0` is allowed only when `decide_stop()` returns `STOP_ALLOWED`:
a verified artifact (revision-pinned, hash-checked), plus an independent
backup, plus a zero-instance readback (`confirm_zero`) afterwards. Machine
churn, partial adapters, unreachable provider or Hub all end in HOLD with an
alert receipt.

## Safe defaults

* `watch_safe.py` is **observe-only** unless `--apply` is passed; `--apply` only
  ever sends `PATCH {"replicas": 0}`.
* `run_observe.py` additionally refuses `--apply`, disables PATCH and disables
  Hub downloads (use while owner config is incomplete).
* `f54v2_heldout_fire_safe.py` defaults to `--plan`; no retry/variant fallback;
  no key or HF token on any command line.
* Keys are read from the files named in the config, per request, never printed.

## Owner config (intentionally blank)

`configs/*.safe.json` leave `target_steps`, the RLVR completion marker/artifact
repo, `backup_dir` and the eval receipt path unset (`null` or `<OWNER: ...>`).
Fill them in a private copy; do not commit values. The controllers HOLD while a
field is blank.

## Tests

`tests/test_training_safety_controllers.py` (47 mocked, offline tests) and
`tests/test_training_safety_vendored_witness.py` (drift guard: the vendored
`provider_transition_witness.py` must stay byte-identical to
`tools/salad/provider_transition_witness.py`).

    python -m unittest tests.test_training_safety_controllers tests.test_training_safety_vendored_witness

## Operational note

The old watchers put the Salad key and HF token on process command lines.
Treat those credentials as exposed and rotate both.
