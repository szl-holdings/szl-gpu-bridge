# syntax=docker/dockerfile:1
# Pinned eval image: szl-eval-pinned.
# Overlay of the latest PUBLISHED training image plus hash-locked lm-eval 0.4.13
# (Suyu suite v1, git commit ddd67220430a2470529f25fd5c05a576ca1057a0).
#
# NOTHING is installed at boot. PIP_NO_INDEX stays on from the base.
# Do not docker push, sign, or publish this image without Stephen's named yes.
#
# Base: latest published ghcr.io/szl-holdings/szl-train-pinned
#   @sha256:8c1c82f0f8551cd807178587362a44880a1f7d9c3358864df08bd3e6f7c6b1fd
#   (workflow_dispatch on main @ 5f6ff205, 2026-11-19 4:40 PM ET).
#   The earlier #283-head digest 975a572d is still on GHCR but is superseded.
#   The published digest is recorded in PIN.txt (TRAIN_IMAGE).
ARG TRAIN_IMAGE=ghcr.io/szl-holdings/szl-train-pinned@sha256:8c1c82f0f8551cd807178587362a44880a1f7d9c3358864df08bd3e6f7c6b1fd
FROM ${TRAIN_IMAGE}

ARG TRAIN_IMAGE
ARG GIT_COMMIT=unknown
ARG EXPECT_LM_EVAL=0.4.13
ARG EXPECT_LM_EVAL_COMMIT=ddd67220430a2470529f25fd5c05a576ca1057a0

# Refuse an unpinned base (a tag can move under us).
RUN case "${TRAIN_IMAGE}" in *@sha256:????????????????????????????????????????????????????????????????) ;; \
      *) echo "TRAIN_IMAGE must be pinned by @sha256 digest, got: ${TRAIN_IMAGE}" >&2; exit 64;; esac

WORKDIR /work
COPY requirements-eval.lock requirements-n14.lock constraints-from-train.txt PIN.txt build_gate.py /work/eval-image/
# Hash-locked overlay only. --no-deps --require-hashes. Already-identical
# packages from the training image are left at the same version (the lock was
# compiled against constraints-from-train.txt; the gate refuses conflicts).
# The base image sets PIP_NO_INDEX=1 so a runtime pip install cannot drift.
# Unset it only for this hashed install (same pattern as the training image
# build, which installs before PIP_NO_INDEX is set). Hashes still gate every
# file. The finished image keeps PIP_NO_INDEX=1 from the base.
RUN env -u PIP_NO_INDEX PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir --no-deps --require-hashes \
      --extra-index-url https://pypi.org/simple \
      --extra-index-url https://wheels.astral.sh/simple/cu126/ \
      -r /work/eval-image/requirements-eval.lock \
      -r /work/eval-image/requirements-n14.lock \
 && python3 /work/eval-image/build_gate.py --require-installed \
 && find /work/eval-image -name __pycache__ -prune -exec rm -rf {} + \
 && rm -rf /tmp/* /root/.cache

ENV SZL_EXPECT_LM_EVAL=${EXPECT_LM_EVAL} \
    SZL_EXPECT_LM_EVAL_COMMIT=${EXPECT_LM_EVAL_COMMIT} \
    SZL_EVAL_SUITE=v1

LABEL org.opencontainers.image.source="https://github.com/szl-holdings/szl-gpu-bridge" \
      org.opencontainers.image.description="Pinned eval image N-14: szl-train-pinned + lm-eval 0.4.13 + mamba-ssm/causal-conv1d (cu126, sm_90)" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.base.name="${TRAIN_IMAGE}" \
      szl.pins="lm-eval==${EXPECT_LM_EVAL} commit=${EXPECT_LM_EVAL_COMMIT}"

# Keep the training image entrypoint. Eval jobs should override the command
# (lm_eval / harness/run_suite.py). Do not train on a laptop; this image is
# for Massed Compute or SaladCloud only.
