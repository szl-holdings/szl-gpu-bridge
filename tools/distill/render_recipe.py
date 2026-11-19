#!/usr/bin/env python3
"""Render the Salad container-group command for a distill lane (print only).

This module never talks to Salad, Hugging Face or any network. It prints the
``image`` and ``command`` a human launches under an exact approval. It fixes
the three recipe bugs found in the stopped lanes:

  1. BASE IMAGE: ``pytorch:2.14.1-cuda12.6`` and no pip step that touches torch
     or torchvision. The old image pinned torch 2.4.1; upgrading torch inside it
     left a stale torchvision (``torchvision::nms does not exist``).
  2. ``max_length``: handled in ``train_lane.sft_config_kwargs`` (TRL 1.17.1 has
     no ``max_seq_length``); it fails rather than dropping the argument.
  3. COMPLETION: bash never echoes LANE_COMPLETE. ``train_lane.py`` prints it
     only after final/ is uploaded and the remote file listing is verified, and
     its exit code is the script's exit code (``set -euo pipefail``).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

BASE_IMAGE = "docker.io/pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
PIP_PACKAGES = (
    "transformers==5.22.2",
    "peft==0.21.2",
    "trl==1.17.1",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
    "protobuf",
)
SHIPPED_FILES = ("distill_corpus.py", "train_lane.py")
HERE = Path(__file__).resolve().parent


def render_command() -> list[str]:
    lines = [
        "set -euo pipefail",
        "export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5",
        "mkdir -p /workspace/out /workspace/lane",
        "pip install -q " + " ".join(PIP_PACKAGES),
    ]
    for name in SHIPPED_FILES:
        blob = base64.b64encode((HERE / name).read_bytes()).decode("ascii")
        lines.append("echo %s | base64 -d > /workspace/lane/%s" % (blob, name))
    lines += [
        "cd /workspace/lane",
        "python3 -c 'import torch, torchvision, transformers, peft, trl; print(\"IMPORTS_OK\", torch.__version__, torchvision.__version__, transformers.__version__, trl.__version__)'",
        "exec python3 /workspace/lane/train_lane.py",
    ]
    return ["bash", "-lc", "\n".join(lines)]


def lint_recipe(image: str, command: list[str]) -> list[str]:
    """Return a list of problems; empty means the recipe is acceptable."""
    problems: list[str] = []
    script = command[-1] if command else ""
    if image != BASE_IMAGE:
        problems.append("image must be %s, got %s" % (BASE_IMAGE, image))
    if command[:2] != ["bash", "-lc"]:
        problems.append("command must be bash -lc <script>")
    if "set -euo pipefail" not in script:
        problems.append("script must run under set -euo pipefail")
    for line in script.splitlines():
        if line.startswith("pip install") and re.search(
            r"\b(torch|torchvision|torchaudio)\b", line
        ):
            problems.append("pip must not install/upgrade torch packages: %r" % line)
        if (
            re.search(r"\bpip install\b.*-U\b", line)
            and "pip install -q -U pip" not in line
        ):
            problems.append("no blanket -U upgrades: %r" % line)
    if "LANE_COMPLETE" in script:
        problems.append("bash must not echo LANE_COMPLETE; train_lane.py owns it")
    if "|| true" in script:
        problems.append("no '|| true' masking in the recipe")
    last = (
        [ln for ln in script.splitlines() if ln.strip()][-1] if script.strip() else ""
    )
    if "train_lane.py" not in last:
        problems.append(
            "train_lane.py must be the last command (its status is the group's status)"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--check", action="store_true", help="lint only; print problems and exit"
    )
    args = ap.parse_args(argv)
    command = render_command()
    problems = lint_recipe(BASE_IMAGE, command)
    if problems:
        print("\n".join("FAIL-CLOSED: " + p for p in problems), file=sys.stderr)
        return 2
    if not args.check:
        print(json.dumps({"image": BASE_IMAGE, "command": command}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
