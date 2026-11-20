# Dream learner NumPy/PyArrow import failure — source and runtime admission

**Status: source-only compatibility repair.** This runbook is not a fresh
SaladCloud fleet census, a credential rotation, an authorization to restart
a learner, an independent GPU training evaluation, or a claim of production
readiness. It does not supersede the existing Flagship54 guard and owner leases.

## Incident evidence and root cause

The user-supplied Vibe/Claude incident transcript records the initial Salad
learner as `deploying` with **desired replicas 0**, then shows a readback
after a separate owner PATCH to desired replicas 1. An authenticated owner
container log later reported, at **2026-11-19 09:30:26Z** and
**10:00:31Z**:

```text
LEARNER round_err ImportError('pyarrow requires NumPy 2.0 or newer, found 1.26.4')
```

This is an **installation-success/runtime-import-failure** class. In the
historical recipe, `numpy==1.26.4` was pinned, `datasets==3.5.1`
allowed a broad PyArrow dependency, and pip resolved 26.x. That runtime version
of PyArrow requires NumPy 2 even though the PyArrow wheel metadata did not
adequately constrain pip's runtime resolver. The independent first-party
[Apache Arrow issue #53954](https://github.com/apache/arrow/issues/53954)
reports the same 26.0.0 behavior and confirms that PyArrow
21–25 imported with NumPy 1.26.4 in its documented test. The
[Apache Arrow install guide](https://arrow.apache.org/docs/python/install.html)
now lists NumPy 2.0+ as the expected NumPy dependency. The independent
[PyTorch TorchTitan report #5365](https://github.com/pytorch/torchtitan/issues/5365)
and [Hugging Face transformers report #50221](https://github.com/huggingface/transformers/issues/50221)
confirm this can escape pip's metadata checks.

**This is not a claim that every newer/older release, Python ABI, CUDA driver,
Torch installation or proprietary model is compatible.** A failed or passed
`pip install` alone is never evidence that a learner imported, trained or
saved a valid checkpoint.

## Later evidence — do not erase successful work

Separate owner records show a v3 recipe using **NumPy 2.1.3**, with an
external no-training container import check marked successful. The owner later
recorded the v3 Salad group version 5 as `stopped`, desired replicas 0,
no instances at **2026-11-20 01:38:16Z**. This is a *historical owner
readback*, not a fresh provider measurement. The protected fleet-watch job
[44667919888](https://github.com/szl-holdings/szl-gpu-bridge/actions/runs/44667919888)
at **2026-11-20 03:28Z** got Salad **HTTP 401**, so a current fresh
provider state was **UNOBSERVED** from that channel; no alternate credentials,
endpoint bypass or automatic relaunch is authorized.

The anonymous Hugging Face public readback of
[`SZLHOLDINGS/dream-learner-v1`](https://huggingface.co/SZLHOLDINGS/dream-learner-v1)
returned exact Git revision
`58bb14d79373f284cd13a67e8c2d010a438ffe4f`, with a
139,512,976-byte adapter safetensors LFS member recorded under SHA-256
`0aeee2cdd755f8931a75012ee5ccab2e5009b86bac4a58360a7ef114be29eeb8`.
The public `wave-logs` receipt
`mesh/dreamlearner/learner-receipt-20261119T2313Z.json` identifies that
same adapter revision and self-reports 313 curriculum rows, 211 seconds of
v3 training, `train_loss=0.5732`, `round=1`, and
`self_eval=PASS`. Another v3 receipt at 23:04Z also calls itself
`round=1`; it is **not proof of a second cumulative training round**.
Public file metadata/digest listing and a learner's self-report do **not**
establish independent heldout quality, true full-resume checkpoint state,
signature authority, optimizer state restoration or deployed inference.

## Source guard implemented here

`tools/salad/numpy_pyarrow_preflight.py` is a no-network, no-import, no-key
static checker for **exact** NumPy/PyArrow pins:

- REFUSE `numpy==1.26.4` + PyArrow 26.0.0 or 26.0.1.
- Accept for **source review only** documented historical
  `numpy==1.26.4` plus PyArrow 19.0.1, 21.0.0, 22.0.0, 23.0.1,
  24.0.0 or 25.0.1; unknown older version tuples stay `HOLD`.
- Accept for **source review only** current PyArrow 26.0.x with exact
  NumPy 2.x and Python 3.11+; Python 3.10 or NumPy 3 are refused. A
  future PyArrow release or unpinned/duplicate requirement stays
  `HOLD` until separately tested.
- Refuse missing NumPy or PyArrow exact pins rather than letting a
  broad transitive `datasets` dependency choose its own arrow version.
- Every verdict reports `runtime_import_verified=false`,
  `dependency_install_verified=false`,
  `training_complete_verified=false`,
  `model_qualified=false`, and `cloud_mutation_authorized=false`.

The current `training-image/requirements.lock` declares
`numpy==2.5.5` and `pyarrow==26.0.1`; the static preflight accepts
that pair for **review**, not as a real executed test. The preflight is
wired into existing `GPU bridge contract CI` before release.
The existing GHCR build workflow now additionally checks an **actual
NumPy/PyArrow import inside the pushed immutable image digest** during
its existing CPU-only smoke step. That future workflow step will not
run a GPU trainer, and it only applies to the pinned-image lane—not the
historical dynamically installed dream-learner command.

## Required owner path (no new billable work authorized)

1. Keep the existing dream-learner group parked until the authorized
   Salad owner obtains a fresh group and **all actual instances** GET.
   Provider HTTP 401/403/429 requires an access stop, not retry with
   another token. Compare desired and all allocating/creating/running/
   stopping states, command revision, image and owner lease. Inspect
   *both* v1 and v1b to prevent duplicate paid work.
2. Source-pin `learner_v3.py` from its existing repository and require a
   reviewed, immutable recipe with full Python/Torch/NumPy/PyArrow/
   transformers/trl/peft dependencies, license and dataset rights;
   do not treat the source script in a shared mutable Hub dataset
   as a protected GitHub deployment authority.
3. In approved **CPU-only cloud CI**, verify
   `import numpy,pyarrow,datasets,torch,transformers,trl,peft` under
   the exact target image and effective constraints, plus a no-weights
   fixture. Do **not** install packages on BETTERWITHAGE or use live GPU
   allocations to debug import errors.
4. Before any new GPU round, reconcile the model repository's current
   exact source and checkpoint/adapter contents with the owner's
   job/instance identity, full optimizer/RNG/scheduler state and an
   independently read-back backup. This is distinct from a self-reported
   loss or a file merely named `adapter_model.safetensors`.
5. Pre-register heldout criteria, a nonadapted baseline and negative
   controls, require an evaluator independent of the learner, and
   preserve all attempts. A `self_eval=PASS` receipt is internal,
   **not** owner-countersigned production promotion.
6. Only after independent approval, signed canonical-image provenance
   and exact deployed-source readback may the sole authorized
   controller start a bounded paid run with explicit cost ceiling and
   an owner-controlled stop/resume path. Do not create another watcher
   or adjust scheduled tasks from this PR.

The proposed source guard is intentionally **one defensive part** of
that chain, not a replacement for provider authentication,
training/backup qualification, independent evaluation, or business
revenue settlement.
