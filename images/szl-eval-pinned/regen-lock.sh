#!/usr/bin/env bash
# Regenerates requirements-eval.lock (hash-locked, linux x86_64, CPython 3.12). Needs uv.
# Constrained to the training-image lock so the overlay cannot bump torch/transformers/peft/trl.
set -euo pipefail
cd "$(dirname "$0")"
uv pip compile requirements-eval.in \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --no-header --no-annotate \
  --index-url https://pypi.org/simple \
  --constraint constraints-from-train.txt \
  --output-file requirements-eval.lock
python3 build_gate.py --lock-only
