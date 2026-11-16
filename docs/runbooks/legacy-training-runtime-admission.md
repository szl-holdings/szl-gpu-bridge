# Legacy training runtime admission

This repair removes mutable runtime package installation from the canonical A11oy
legacy recipe. It does not deploy an image, restart a workload, alter replicas,
rotate credentials, publish weights, or admit a training run.

## Failure being prevented

An unbounded install into an older PyTorch image can replace torch while leaving
incompatible vision/audio extensions. The observed failure pattern is a dependency
conflict, missing torchvision::nms, AutoProcessor import failure, then TRL trainer
import failure before training begins. The preflight rejects version drift, pip
conflicts, import failures and SFTConfig API mismatch before the unchanged trainer.
Do not hide this with a success tail, suppressed import, zero meter value, or
filename-only readiness test.

## Operator-owned rollout gate

Build and test an isolated prebuilt image. Record its immutable digest, complete
build lock/SBOM and a separately admitted source commit. Do not mutate a shared
running environment or select dependency versions from a card. The runtime contract
pins the relevant distributions, not every transitive package; pip check and real
imports are required but do not substitute for full image admission or GPU tests.

Set SZL_TRAINING_RUNTIME_CONTRACT to a local JSON object with exactly these keys:
schema=szl.training-runtime/v1, repository=szl-holdings/szl-gpu-bridge,
source_revision=<40-hex admitted commit>, recipe_sha256=<exact recipe bytes>,
python=<major.minor>, distributions=<exact package-version object>.
Required packages: torch, transformers, trl, datasets, peft, accelerate,
huggingface-hub, safetensors, nvidia-ml-py. Include exact torchvision and torchaudio
versions when installed. Placeholders are not executable values; no tested runtime
version set is claimed by this source change.

Run the validator with --contract and --recipe before a controller-authorized launch.
It performs pip check and an offline import/API smoke test, prints a secret-free JSON
result, and exits 78 on a hold. It never installs packages or loads model weights.
A PASS explicitly leaves source_admission_verified=false: the existing controller
must verify provenance, image digest, data rights, exclusive generation ownership,
resume-state preservation, training parameters, budget and publication authority.

The inline Python training body is unchanged. Historical cloud payloads can differ
from current source in arguments and output behavior; do not replace a live payload
merely by pasting this file or alter epochs/warmup/targets as an incidental repair.
This legacy trainer still requires separate durable-resume and publication review.
No successful training execution or runtime recovery is asserted by the preflight.

## Receipt claims

The small receipt_claims helper preserves an explicitly recorded abstention/refusal
numerator and denominator. Missing category labels stay unavailable, including an
empty subgroup. It does not infer labels from prompt wording, regrade results,
validate meters, inherit scores to current weights, or qualify production. Zero raw
meter readings remain raw readings until independently corroborated.

The Khipu card source and adjacent evidence record bind the observed original card
and unmodified receipt. Publish only after source admission using expected-parent
Hub writes, and verify the exact README bytes, unchanged siblings and visibility.

## Validation and rollback

Run the new runtime-preflight and receipt-claims unit tests. Tests use synthetic
contracts and subprocess doubles; real negative CLI cases fail before imports.
They are not GPU compatibility measurements. Normal CI and review still apply.
Rollback by a reviewed source revert; preserve observations and original receipts.
A rollback is not permission to restore unbounded runtime pip installation.
