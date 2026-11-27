# syntax=docker/dockerfile:1
# Pinned, hash-locked training image: NOTHING is installed at boot.
# Built in CI by .github/workflows/training-image-ghcr.yml. Derived from Rikuy's Dockerfile.pinned.template.
#
# Base: pytorch/pytorch 2.14.1 / CUDA 12.6.3 / cuDNN 9 runtime (Ubuntu 24.04, CPython 3.12), pinned by digest.
# It ships torch 2.14.1+cu126, torchvision 0.29.1+cu126, torchaudio 2.11.0+cu126; requirements.lock pins the
# same builds (and the exact nvidia-* libs) so nothing in the torch stack changes. A replica that starts either
# passes the import/GPU gate or exits non-zero before it touches data.
ARG BASE_IMAGE=pytorch/pytorch@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56
FROM ${BASE_IMAGE}

ARG GIT_COMMIT=unknown
ARG BASE_IMAGE
ARG EXPECT_TORCH=2.14.1+cu126
ARG EXPECT_TORCHVISION=0.29.1+cu126
ARG EXPECT_TORCHAUDIO=2.11.0+cu126
ARG EXPECT_CUDA=12.6
ARG EXPECT_TRANSFORMERS=5.22.2
ARG EXPECT_PEFT=0.21.2
ARG EXPECT_TRL=1.17.1

# Refuse an unpinned base (a tag can move under us).
RUN case "${BASE_IMAGE}" in *@sha256:????????????????????????????????????????????????????????????????) ;; \
      *) echo "BASE_IMAGE must be pinned by @sha256 digest, got: ${BASE_IMAGE}" >&2; exit 64;; esac

WORKDIR /work
ENV PYTHONPATH=/work
COPY training-image/requirements.lock /work/requirements.lock
# Every artifact must match a hash in the lock (--require-hashes); the lock is the full closure (--no-deps).
# The pytorch cu126 index is listed only so the "+cu126" torch builds resolve if the base ever drifts; hashes
# still gate every file. Already-identical packages from the base are left alone.
RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir --no-deps --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu126 -r /work/requirements.lock

# N-14 overlay: hash-locked mamba-ssm / causal-conv1d (Astral cu126 / torch 2.14, sm_90 in the wheels).
# Installed before PIP_NO_INDEX is set. Does not re-resolve or bump the training lock.
COPY training-image/requirements-n14.lock /work/requirements-n14.lock
RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir --no-deps --require-hashes \
      --extra-index-url https://pypi.org/simple \
      --extra-index-url https://wheels.astral.sh/simple/cu126/ \
      -r /work/requirements-n14.lock

# One canonical training worker. Do not build from the retired image-local copy:
# it predates immutable resume_integrity and may silently restart from genesis.
COPY tools/salad/salad_worker.py /work/salad_worker.py
COPY tools/salad/resume_integrity.py tools/salad/training_snapshot.py /work/tools/salad/
COPY training-image/readiness_probe.py training-image/step_heartbeat.py training-image/build_gate.py /work/
COPY training-image/source_runtime_contract.py /work/source_runtime_contract.py
# Run-receipt uploader (proofrun step 4): one commit per run / checkpoint save, never per heartbeat.
# No token is baked in: it reads HF_TOKEN from the environment at runtime.
COPY training-image/receipt_upload.py /work/receipts/receipt_upload.py
COPY training-image/vendor /work/receipts/vendor
COPY training-image/receipt_tests /work/receipts/tests

# Bind all installed worker/helper/uploader bytes after their final COPY.
RUN python3 -m py_compile /work/salad_worker.py /work/tools/salad/resume_integrity.py /work/tools/salad/training_snapshot.py \
 && python3 /work/source_runtime_contract.py create --root /work --manifest /work/source-bindings.json --source-revision "${GIT_COMMIT}" \
 && python3 /work/source_runtime_contract.py verify --root /work --manifest /work/source-bindings.json --source-revision "${GIT_COMMIT}"

# Build-time gate: the image cannot be produced if any of this fails.
RUN python3 /work/readiness_probe.py --help > /dev/null \
 && python3 -W ignore /work/build_gate.py \
 && python3 -W ignore /work/readiness_probe.py imports \
      --expect transformers=${EXPECT_TRANSFORMERS} --expect peft=${EXPECT_PEFT} --expect trl=${EXPECT_TRL} --no-gpu \
 && find /work -name __pycache__ -prune -exec rm -rf {} + \
 && rm -rf /tmp/* /root/.cache

ENV SZL_SOURCE_REVISION=${GIT_COMMIT} \
    HF_HOME=/work/hf-cache \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SZL_HEARTBEAT_FILE=/tmp/szl_heartbeat.json \
    SZL_EXPECT_TRANSFORMERS=${EXPECT_TRANSFORMERS} \
    SZL_EXPECT_PEFT=${EXPECT_PEFT} \
    SZL_EXPECT_TRL=${EXPECT_TRL} \
    SZL_RUN_RECEIPT_UPLOADER=/work/receipts/receipt_upload.py \
    SZL_RUN_RECEIPT_REPO=SZLHOLDINGS/szl-run-receipts
# PIP_NO_INDEX=1: any stray `pip install` at runtime fails instead of silently drifting.
# HF_TOKEN is deliberately NOT set here; the runtime injects it.

LABEL org.opencontainers.image.source="https://github.com/szl-holdings/szl-gpu-bridge" \
      org.opencontainers.image.description="Pinned hash-locked training image N-14 (torch 2.14.1+cu126 + mamba-ssm/causal-conv1d sm_90)" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      szl.pins="torch=${EXPECT_TORCH} torchvision=${EXPECT_TORCHVISION} torchaudio=${EXPECT_TORCHAUDIO} transformers==${EXPECT_TRANSFORMERS} peft==${EXPECT_PEFT} trl==${EXPECT_TRL}"

# Boot gate (includes the GPU bf16 smoke; the build host has no GPU) then the worker.
# Use `readiness_probe.py ready` / `live` as the Salad readiness / liveness probes.
STOPSIGNAL SIGTERM
ENTRYPOINT ["/bin/sh", "-c", "python3 /work/source_runtime_contract.py verify --root /work --manifest /work/source-bindings.json --source-revision \"$SZL_SOURCE_REVISION\" && python3 -W ignore /work/readiness_probe.py imports --from-env && exec python3 /work/salad_worker.py \"$@\"", "--"]
