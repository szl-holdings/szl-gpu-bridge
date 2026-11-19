# SaladCloud transition receipt safety (source-only)

`tools/salad/provider_transition_witness.py` is an offline pure witness to one caller-supplied PATCH/POST outcome and subsequent GET readbacks. It **does not** call SaladCloud, read credentials, launch training, stop instances, create a controller, retry failed operations, or authorize spend. The original Flagship54 guard remains the only Flagship54 shutdown owner.

## Incident behind this requirement

A curl process may exit `0` while a provider returns HTTP `400` with `pending_update_in_progress`. Zero transport exit only means the request completed, **not** that SaladCloud accepted a replica update. A separate unsafe failure class is treating a `final/` or `lane.log` filename as sufficient completion evidence. This module prevents the HTTP-status false admission; it does **not** operate or replace shutdown controllers. All examples are generic and contain no private run identities or local incident evidence.

## Required caller inputs

Use the existing approved provider SDK/authentication and controller ownership to collect, separately: exact transport exit code, **independently captured HTTP response status**, bounded raw body, exact group name, desired replicas, and fresh group + instance GET readbacks. Never infer HTTP status from curl exit code. An ambiguous prior PATCH outcome must be reconciled by current GET before *any* retry. Stop on 401/403/429 without alternate credential, endpoint, or account; no automatic retries.

`witness(...)` rejects missing HTTP status, non-2xx responses, error envelopes masked by 2xx, pending updates, wrong group/replicas and malformed counts. A coherent provider readback returns `PROVIDER_STATE_OBSERVED_NOT_TRAINING`, `NO_RUNNING_INSTANCE_PROVIDER_REPORTED`, or `ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION`. Its `training_launch_authorized` and `shutdown_authorized` fields **always remain false**. The two GET responses are not atomic; inspect timestamps/versions separately and never convert them into trainer-process, checkpoint, signature, quality, or backup evidence.

## Production admission remains independent

The existing dispatcher/guard must separately verify exclusive owner and generation, approved compute budget, immutable source/base/dataset/image, data rights, current resource pressure, measured process/heartbeat and checkpoint progress, resumable optimizer/RNG state, final byte integrity and backup readback, held-out evaluation and independent release approval. No new paid capacity or controller is created by this source change.

Leadership references: [NVIDIA Megatron Bridge recipes](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html) (separate full resume and pretrained initialization), [PyTorch TorchTitan](https://github.com/pytorch/torchtitan) (checkpointable training across parallelism), [Ray Train checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html) (fault recovery), and [SaladCloud group API](https://docs.salad.com/reference/saladcloud-api/container-groups/list-container-groups). These are research references; no upstream code was copied. The new checker is not a distributed trainer, and tests are synthetic.

Offline tests: `python -m unittest discover -s tests -p test_provider_transition_witness.py -v`.
