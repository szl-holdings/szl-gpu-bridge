#!/usr/bin/env bash
# Regenerates requirements.lock (hash-locked, linux x86_64, CPython 3.12). Needs uv.
set -euo pipefail
cd "$(dirname "$0")"
uv pip compile requirements.in \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --no-header --no-annotate \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  --output-file requirements.lock
