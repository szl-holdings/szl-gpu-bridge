# Pinned training image (`ghcr.io/szl-holdings/szl-train-pinned`)

A training image where nothing is installed at boot. Built in CI by `.github/workflows/training-image-ghcr.yml`.

**Canonical worker rule:** only `tools/salad/salad_worker.py` is copied into
`/work/salad_worker.py` from the exact GitHub checkout. The earlier
`training-image/salad_worker.py` snapshot is retired and not used as an image
source. The image also includes the canonical immutable-resume dependencies
`tools/salad/resume_integrity.py` and `tools/salad/training_snapshot.py`,
with `PYTHONPATH=/work`. A stale vendor worker must not silently reintroduce
moving-main resume or lose later source fixes.

`source_runtime_contract.py` records the six fixed source-file byte hashes
at build and checks them before every worker entrypoint. The manifest is an
**image-internal integrity marker, not a signature**. For deployment, an
independent owner must authenticate the OCI digest, Sigstore signer identity,
builder/workflow revision and provenance subject, then read back that exact
digest from SaladCloud. Healthy HTTP, manifest equality, or a successful
container start is not proof that any optimizer step ran or that checkpoints
survived a preemption.

| Layer | Pin |
| --- | --- |
| Base | `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime` @ `sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56` (Ubuntu 24.04, CPython 3.12, CUDA 12.6.3) |
| torch stack | `torch 2.14.1+cu126`, `torchvision 0.29.1+cu126`, `torchaudio 2.11.0+cu126` (exactly what the base ships) |
| Training libs | `transformers==5.22.2`, `peft==0.21.2`, `trl==1.17.1` |
| Everything else | `requirements.lock`, the full closure with sha256 hashes, installed with `--require-hashes --no-deps` |

Compatibility evidence (published metadata, checked 2026-11-19):
- `torch 2.14.1+cu126` requires `cuda-toolkit[...]==12.6.3`, `nvidia-cudnn-cu12==9.10.2.21`, `nvidia-nccl-cu12==2.29.3`, `triton~=3.8.0`; the lock pins exactly those, identical to the base's installed copies.
- `torchvision 0.29.1+cu126` requires `torch>=2.14.0`. `transformers 5.22.2` needs `torch>=2.6` (extra); `trl 1.17.1` needs `torch>=2.6.0`, `transformers>=4.56.2`; `peft 0.21.2` needs `torch>=1.13`.
- The PyPI default `torch 2.14.1` is a CUDA 13 build; this image deliberately uses the CUDA 12.6 build so it runs on 12.x drivers.
- `torch 2.15.0` / `torchvision 0.19.1` (the Salad crash loop) and `torch 2.4.1` are not used.

`build_gate.py` runs at build time and fails the build on: exact versions and `torch.version.cuda == 12.6`; `torchvision.ops.nms` on CPU; a torchaudio op; a tiny Llama with a LoRA adapter forward; `trl.SFTConfig` built with the sequence-length argument it really exposes (`max_length` in TRL 1.17.1; `max_seq_length=` is rejected); `salad_worker.sft_sequence_length_kwargs`; the run-receipt uploader (`--dry-run` on a synthetic proof run, its 12 unit tests, no token present).

## Run-receipt uploader (proofrun step 4)
`/work/receipts/receipt_upload.py` (+ vendored `proofrun`) uploads a run folder to the private HF repo `SZLHOLDINGS/szl-run-receipts` as ONE commit under `runs/<run_id>/`. No token is in the image: set `HF_TOKEN` (write access to that repo only) at runtime.

```
python3 /work/receipts/receipt_upload.py <run_dir> --repo "$SZL_RUN_RECEIPT_REPO" --run-id <run_id> --large-mb 10 [--trusted-key ed25519:<fp>] [--accept-retroactive]
```
Call it once at the end of a run and once per checkpoint save, never per heartbeat, as a non-blocking step. Exit 3 means rate-limited (capped backoff, 5 s base, 300 s cap, 6 tries): skip and try at the next checkpoint. `tools/salad/salad_worker.py` is not changed by this PR (several open PRs edit it); the image's copy only adds the heartbeat wiring, and calling the uploader from the checkpoint callback is a separate change.

## Liveness heartbeat (Salad probe)
`/work/readiness_probe.py` is shipped (Rikuy's `patches/readiness_probe.py`; Saywa's held payload dir has the probe patch JSON only, no script copy). `/work/step_heartbeat.py` keeps `/tmp/szl_heartbeat.json` fresh for `python3 /work/readiness_probe.py live --file /tmp/szl_heartbeat.json --max-age-seconds 900`:
- A Trainer callback rewrites it from `on_step_end` only when `global_step` advanced and >= 60 s (`SZL_HEARTBEAT_WRITE_SECONDS`, clamped 30-180) passed; also at train begin/end. No background thread: a hung trainer goes stale and the probe fails at 900 s.
- One-shot beats at setup milestones (start, dataset ready, model loaded) so the startup probe can pass during a large download.
- Atomic (temp file in the same directory, then `os.replace`), local only, never uploaded to the Hub.
- Fields: `step` (int), `max_steps` (int or null), `run_id` (str or null; `SZL_RUN_ID` else the adapter repo), `unix` (float, `time.time()`), `pid` (int).
- Build gate checks `readiness_probe.py --help`, the callback semantics (fields, interval, no write without step progress, no temp left over) and `probe live` pass/fail.
- `tools/salad/salad_worker.py` is the sole training implementation. A guarded `step_heartbeat` hook is now in that canonical source, activated only when `SZL_HEARTBEAT_FILE` is set by the pinned image. Other workers keep their original runtime behavior.

## Maintenance
- Regenerate the lock: `./regen-lock.sh` (uv). Review the diff; the gate and CI re-verify.
- `salad_worker.py` and `readiness_probe.py` here are copies (worker = `tools/salad/salad_worker.py` at main `817ea920` plus the heartbeat wiring). Keep them in sync when those change.
- Runtime image stays token-free; only the owner's approved environment may supply scoped credentials at container creation. Launch specs must use `ghcr.io/szl-holdings/szl-train-pinned@sha256:<64hex>`, never a mutable tag.
- The existing workflow is build/sign infrastructure, **not a production deployer**. No image or replica is changed merely because this source PR passes CI. Owner must separately approve the build, verify signature/provenance and image digest, then make a reviewed deployment with independent group+instance and trainer-step/readback proof.
- `training-image/.dockerignore` is not authoritative when the workflow uses repository-root build context; the root `.dockerignore` applies. Do not remove required canonical worker/helpers from that context.
- A complete resume requires opaque optimizer/RNG/scheduler state plus immutable checkpoint file identities and a durable independent backup. Copying the files into the image does not prove any real training run resumed.
