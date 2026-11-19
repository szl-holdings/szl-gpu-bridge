# Pinned training image (`ghcr.io/szl-holdings/szl-train-pinned`)

A training image where nothing is installed at boot. Built in CI by `.github/workflows/training-image-ghcr.yml`.

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
python3 /work/receipts/receipt_upload.py <run_dir> --repo "$SZL_RECEIPT_REPO" --run-id <run_id> --large-mb 10 [--trusted-key ed25519:<fp>] [--accept-retroactive]
```
Call it once at the end of a run and once per checkpoint save, never per heartbeat, as a non-blocking step. Exit 3 means rate-limited (capped backoff, 5 s base, 300 s cap, 6 tries): skip and try at the next checkpoint. `salad_worker.py` is not changed by this PR; wiring the call into the worker's checkpoint callback is a separate change because several open PRs edit that file.

## Maintenance
- Regenerate the lock: `./regen-lock.sh` (uv). Review the diff; the gate and CI re-verify.
- `salad_worker.py` and `readiness_probe.py` here are copies (worker = `tools/salad/salad_worker.py` at main `817ea920`). Keep them in sync when those change.
- Runtime stays token-free; Salad launch specs should reference the image by `@sha256` digest.
