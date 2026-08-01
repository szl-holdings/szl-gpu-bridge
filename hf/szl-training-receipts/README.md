---
pretty_name: SZL Training Receipts
license: other
license_name: szl-governed-operational-evidence-no-blanket-reuse-grant
license_link: https://github.com/szl-holdings/szl-gpu-bridge/blob/main/hf/szl-training-receipts/README.md#license-and-data-handling
tags:
- governance
- receipts
- dsse
- provenance
- private-evidence
---

# SZL Training Receipts

Private, append-only evidence store for governed owner-GPU attempts. This
repository is **not a training corpus**, benchmark, model release, or proof that
a queued job ran successfully.

## Authority boundary

- Source controller:
  [`szl-holdings/szl-gpu-bridge`](https://github.com/szl-holdings/szl-gpu-bridge)
- Source revision: `__SOURCE_REVISION__`
- Target: `SZLHOLDINGS/szl-training-receipts`
- Visibility: private
- Publication authority: the trusted finalizer after local signature,
  jobspec, queue-envelope, replay-barrier, and immutable-readback validation

The source controller verifies the exact job ID, queue payload hash, enrolled
owner-laptop key ID, signature, evaluation fields, and no-publication boundary.
A missing receipt remains missing evidence; it is never converted into success.

## Expected record shape

Terminal records are stored under their exact governed job ID. The canonical
validation and refusal logic lives in:

- `cloud/nemo_v3_status.py`
- `laptop/finalize_nemo_v3_receipt.py`
- `laptop/nemo_v3_contract.py`
- `schema/nemo-v3-jobspec.v1.json`

No record count is claimed in this card. Inspect the immutable repository
revision and validate each receipt independently before relying on it.

## License and data handling

`license: other` is deliberate. Receipt envelopes may contain governed
operational metadata and do not receive a blanket data-reuse grant. Model
weights, credentials, private keys, raw training rows, and secrets must never be
uploaded here.
