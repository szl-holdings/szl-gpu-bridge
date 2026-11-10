# Migration launcher qualification

The A6000 migration launchers now require literal arguments, refuse duplicate
screen sessions, and distinguish process startup from workload success.

`launch_leg.sh` works from any directory in a complete checkout. It runs
`laptop/gpu_preflight.py --require-bf16` before starting a leg. The probe checks
an 8x8 BF16 matmul and backward operation on CUDA device 0, synchronizes, and
checks known numerical results. It records the installed Torch/CUDA versions,
compute capability and VRAM. PASS establishes only this operation; it does not
establish model fit, convergence, throughput, or multi-node connectivity.
Missing Torch, unavailable CUDA, unsupported BF16, and kernel execution failures
block launch. No packages are upgraded by the probe.

Keep `tools/a6000-migration/` and `laptop/{gpu_preflight,benchmark_power}.py`
together when copying these launchers. Credentials are inherited through the
environment rather than interpolated into screen command arguments. The existing
training legs still install mutable dependencies and use mutable model/data refs;
qualify and pin those separately before a reproducible benchmark.

`serve_model.sh NAME SOURCE` requires an explicit checkpoint repository or local
GGUF file. It preserves literal source arguments, rejects an occupied port 8080,
and checks both session existence and `/health` before reporting READY. A slow
startup exits nonzero after a bounded health window but leaves the session for
inspection. Readiness does not establish correct model identity or output quality.
The existing bind address remains 0.0.0.0; use the deployment's authenticated
network boundary. GGUF/llama.cpp and HF/vLLM are separate serving routes; neither
is the BF16/LoRA training path.

The 4090/5090 benchmark legs now integrate NVML power with a monotonic clock only
during the training window. Failed, negative, or nonfinite samples produce an
INCOMPLETE report with null total energy and null derived efficiency/carbon.
Successful integration is an estimate from sampled power, not a hardware energy
counter. Tokens remain explicitly estimated; the carbon multiplier is a scenario,
not verified regional intensity. No GPU measurements were made by these unit tests.

## GPU image qualification

A provider's RUNNING status and an image name are not runtime evidence. The listed
PyTorch 2.4.1/CUDA 12.1 image predates official Blackwell wheel support; inspect the
installed runtime in case startup modifies it. PyTorch 2.7 introduced Blackwell
CUDA 12.8 wheels. Current PyTorch 2.15 uses CUDA 13.2 as its stable wheel line,
requires Python 3.11+, and changes distributed-runtime behavior. Qualify a separate
candidate image and its host driver before replacing a working A6000 environment.
Do not transfer Blackwell NVFP4 performance claims to the A6000.

Sources:
- https://pytorch.org/blog/pytorch-2-7/
- https://pytorch.org/blog/pytorch-2-15-release-blog/
- https://github.com/pytorch/pytorch/blob/main/RELEASE.md

Acceptance: one preflight per GPU type, then a separately authorized 50-step
fixed-revision BF16/LoRA run; require finite loss/gradients, no OOM, successful
checkpoint reload, measured peak VRAM, and comparable tokens/s against the current
image. Production replacement remains contingent on those results.
