# training_safety: fail-closed Salad controller replacements

Replacements for the local Salad training watchers (`watch_arm6`, `watch_arm7`,
`rlvr_watch4`, `f54v2_watch`) and the held-out eval launcher
(`f54v2_heldout_fire3`). Stdlib only. They fix four flaws observed on 2026-11-19:

1. **curl rc is not HTTP status.** The launcher logged `rc=0` next to an HTTP 400
   `pending_update_in_progress` body. Writes are now judged by HTTP status + body
   (`classify_write`), reusing the transition witness from #264.
2. **arm6/arm7 stopped a group on a bare `final/` filename.** A `final/` entry is
   no longer evidence of completion.
3. **rlvr_watch4 stopped when `lane.log` existed.** Existence is not completion;
   the owner's completion marker must appear in the log tail.
4. **f54v2_watch always reported 200.** Status now comes from the response.

## Stop gate

`replicas -> 0` is sent only when `decide_stop()` returns `STOP_ALLOWED`:

* the Hub revision is pinned (40-hex) and every required file was downloaded at
  that revision and matches the Hub's size and hash;
* the step target is reached and `final/` holds only allow-listed files
  (anything unexpected under `final/` HOLDs);
* a backup copy exists under `backup_dir` and re-hashes equal. The backup must
  not be the cache dir (or inside it) or the credential dir; the volume ids are
  recorded. **That the backup is on another node is attested by the owner
  (`backup_independent`), not proved by this code.**
* repository visibility is known. A private repo is required, except that an
  `eval_receipt` may live in a public repo only when the owner sets
  `allow_public_repo: true`, and then the receipt content must also match the
  owner's `receipt_schema` and `receipt_required_fields`. A file merely existing
  at the receipt path never counts.

After a PATCH, `confirm_zero()` must read back desired replicas 0, all counts 0
and an empty instance list before the result is DONE. Machine churn is an alert,
never a stop. `STOP_ALLOWED` is not a claim that training completed or that
anything was evaluated.

## What a HOLD does (and does not do)

Any HOLD means no action. It is logged every cycle, and a `szl.controller-hold/v1`
receipt is appended when a group's HOLD code changes (churn writes its own alert
receipt). **Nothing notifies a person**: the receipt and log files are the
alert surface, so something else has to watch them. A watcher left on a
blank config will HOLD forever and the GPUs keep billing.

An exception in one group (unreadable key file after a rotation, bad config key,
log write error) is caught, logged by type only, receipted as
`HOLD_CONTROLLER_EXCEPTION_<Type>`, and the watcher continues with the other
groups and the next cycle. After `max_consecutive_failed_cycles` (default 12)
failing cycles in a row it exits with code 3.

## Credentials

Keys are read from the files named in the config at request time, never put in
argv and never logged. They are only sent over **https** to an allow-listed host:
`api.salad.com` for the Salad key, `huggingface.co` for the Hub token. Redirects
are never followed automatically; Hub downloads follow them by hand to Hugging
Face hosts only and drop the `Authorization` header on any other host. A
`salad_base` or `lane_log_url` that names another host is rejected. Only the
**tests** are credential-free; the tools read and use credentials when run.

## Safe defaults

* `watch_safe.py` is **observe-only** unless `--apply`; `--apply` only ever sends
  `PATCH {"replicas": 0}`, and is refused (exit 4) while any `<OWNER: ...>`
  placeholder remains in the config.
* `run_observe.py` additionally refuses `--apply`, disables PATCH and disables Hub
  downloads (use while owner config is incomplete).
* `f54v2_heldout_fire_safe.py` defaults to `--plan`. `--apply` is refused while
  any placeholder remains, unless the image is pinned by `@sha256:` digest, and
  if the command runs `pip install` without `--require-hashes`. No retry or
  variant fallback; credential-looking env names or values are rejected.

## Owner config

`configs/*.example.json` are templates: every owner-specific value is an
`<OWNER: ...>` placeholder and nothing in them is real. Copy one to a private
location, fill it in, and keep that copy out of git. The controllers HOLD (or
refuse) while a field is blank.

## Tests

`tests/test_training_safety_controllers.py` (offline, mocked) and
`tests/test_training_safety_vendored_witness.py` (drift guard: the vendored
`provider_transition_witness.py` must stay byte-identical to
`tools/salad/provider_transition_witness.py`).

    python -m unittest tests.test_training_safety_controllers tests.test_training_safety_vendored_witness

CI currently collects the tests but does not `py_compile` or `ruff` the files in
this directory; adding them to `ci.yml` needs a workflow change.

## Operational note

The old watchers put the Salad key and HF token on process command lines.
Treat those credentials as exposed and rotate both.
