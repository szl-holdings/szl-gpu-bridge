#!/usr/bin/env python3
"""Fail-closed build gate for szl-eval-pinned.

Lock-only mode (default when lm_eval is not installed, or --lock-only):
  - requirements-eval.lock pins lm-eval==0.4.13 with the PyPI wheel sha256
  - no version conflicts against constraints-from-train.txt
  - PIN file matches Suyu suite v1 (commit ddd67220 / tag v0.4.13)
  - requirements-n14.lock (if present) pins mamba-ssm and causal-conv1d
    with the Astral cu126/torch 2.14 wheel sha256s and does not bump the
    training stack

Installed mode (inside the image, or --require-installed):
  - importlib.metadata version is 0.4.13
  - `lm_eval` imports
  - training-stack packages, if present, still match the training pins
  - numpy is 2.x if pyarrow 26 is present (the #302 crash pair is refused)
  - if the N-14 lock is present: mamba-ssm and causal-conv1d versions match
  - transformers exposes NemotronHForCausalLM (CPU class import, no GPU load)

This gate never authorizes a GHCR push, a signing, a launch, or a spend.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "requirements-eval.lock"
CONSTRAINTS = HERE / "constraints-from-train.txt"
PIN = HERE / "PIN.txt"

EXPECT_VERSION = "0.4.13"
EXPECT_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"
EXPECT_TAG = "v0.4.13"
EXPECT_WHEEL_SHA256 = "5daaa1973bf874005f64f28d3834b875f6886f0d6475878e6a6c821994a5286a"
TRAIN_PINS = {
    "transformers": "5.22.2",
    "peft": "0.21.2",
    "trl": "1.17.1",
}
N14_LOCK = HERE / "requirements-n14.lock"
EXPECT_MAMBA = "2.3.2.post1+cu.12.6.torch.2.14"
EXPECT_MAMBA_WHEEL_SHA256 = "329557d964d58ebfcaf44cb17dc043356a24dd5c8ab9b68d96dc0fbf89911614"
EXPECT_CONV = "1.7.0+cu.12.6.torch.2.14"
EXPECT_CONV_WHEEL_SHA256 = "75f42d6ab5c87490d04a9a58f9e3c191c9cd88b31e8b626fc057099e01832d60"


def _die(msg: str, code: int = 1) -> None:
    print(f"BUILD_GATE FAIL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_lock(path: Path) -> dict[str, tuple[str, list[str]]]:
    text = path.read_text()
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+(?:\[[^\]]*\])?==)", text.strip())
    out: dict[str, tuple[str, list[str]]] = {}
    for b in blocks:
        m = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s\\]+)", b)
        if not m:
            continue
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", b)
        if not hashes:
            _die(f"unhashed requirement in {path.name}: {m.group(1)}=={m.group(2)}")
        out[m.group(1).lower()] = (m.group(2), hashes)
    return out


def parse_constraints(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==(\S+)", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def parse_pin(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def check_lock() -> dict[str, tuple[str, list[str]]]:
    if not LOCK.is_file():
        _die(f"missing {LOCK}")
    if not CONSTRAINTS.is_file():
        _die(f"missing {CONSTRAINTS}")
    if not PIN.is_file():
        _die(f"missing {PIN}")
    ev = parse_lock(LOCK)
    train = parse_constraints(CONSTRAINTS)
    pin = parse_pin(PIN)
    if "lm-eval" not in ev:
        _die("lm-eval is not in requirements-eval.lock")
    ver, hashes = ev["lm-eval"]
    if ver != EXPECT_VERSION:
        _die(f"lm-eval pinned at {ver}, expected {EXPECT_VERSION}")
    if EXPECT_WHEEL_SHA256 not in hashes:
        _die("requirements-eval.lock is missing the PyPI wheel sha256 for lm-eval 0.4.13")
    if pin.get("LM_EVAL_VERSION") != EXPECT_VERSION:
        _die("PIN.txt LM_EVAL_VERSION mismatch")
    if pin.get("LM_EVAL_GIT_COMMIT") != EXPECT_COMMIT:
        _die("PIN.txt LM_EVAL_GIT_COMMIT mismatch")
    if pin.get("LM_EVAL_GIT_TAG") != EXPECT_TAG:
        _die("PIN.txt LM_EVAL_GIT_TAG mismatch")
    if pin.get("LM_EVAL_WHEEL_SHA256") != EXPECT_WHEEL_SHA256:
        _die("PIN.txt LM_EVAL_WHEEL_SHA256 mismatch")
    conflicts = []
    for name, (pkg_ver, _) in ev.items():
        if name in train and train[name] != pkg_ver:
            conflicts.append(f"{name}: train {train[name]} vs eval {pkg_ver}")
    if conflicts:
        _die("eval lock would bump the training stack: " + "; ".join(conflicts))
    print(f"LOCK OK: lm-eval=={ver} wheel sha256={EXPECT_WHEEL_SHA256}")
    print(f"LOCK OK: {len(ev)} packages, 0 conflicts with training constraints ({len(train)} pins)")
    print(f"PIN OK: tag {EXPECT_TAG} commit {EXPECT_COMMIT}")
    if N14_LOCK.is_file():
        n14 = parse_lock(N14_LOCK)
        train = parse_constraints(CONSTRAINTS)
        for name, expect, wheel in (
            ("mamba-ssm", EXPECT_MAMBA, EXPECT_MAMBA_WHEEL_SHA256),
            ("causal-conv1d", EXPECT_CONV, EXPECT_CONV_WHEEL_SHA256),
        ):
            if name not in n14:
                _die(f"{name} is not in {N14_LOCK.name}")
            ver, hashes = n14[name]
            if ver != expect:
                _die(f"{name} pinned at {ver}, expected {expect}")
            if wheel not in hashes:
                _die(f"{N14_LOCK.name} is missing the expected wheel sha256 for {name}")
        n14_conflicts = []
        for name, (pkg_ver, _) in n14.items():
            if name in train and train[name] != pkg_ver:
                n14_conflicts.append(f"{name}: train {train[name]} vs n14 {pkg_ver}")
        if n14_conflicts:
            _die("N-14 lock would bump the training stack: " + "; ".join(n14_conflicts))
        if "nvidia-cutlass-dsl-libs-cu13" in n14:
            _die("N-14 lock pulled nvidia-cutlass-dsl-libs-cu13; this image is CUDA 12.6")
        print(f"LOCK OK: mamba-ssm=={EXPECT_MAMBA} wheel sha256={EXPECT_MAMBA_WHEEL_SHA256}")
        print(f"LOCK OK: causal-conv1d=={EXPECT_CONV} wheel sha256={EXPECT_CONV_WHEEL_SHA256}")
        print(f"LOCK OK: N-14 overlay {len(n14)} packages, 0 conflicts with training constraints")
    return ev


def check_installed() -> None:
    try:
        import importlib.metadata as md
    except ImportError:
        _die("importlib.metadata missing")
    try:
        got = md.version("lm-eval")
    except md.PackageNotFoundError:
        _die("lm-eval is not installed", 3)
    if got != EXPECT_VERSION:
        _die(f"installed lm-eval {got}, expected {EXPECT_VERSION}")
    try:
        import lm_eval  # noqa: F401
    except Exception as e:
        _die(f"import lm_eval failed: {e}")
    for name, expect in TRAIN_PINS.items():
        try:
            v = md.version(name)
        except md.PackageNotFoundError:
            continue
        if v != expect:
            _die(f"training pin drifted: {name} {v}, expected {expect}")
    # #302 pair: refuse pyarrow 26 + numpy 1.x if both present
    try:
        np_v = md.version("numpy")
        pa_v = md.version("pyarrow")
    except md.PackageNotFoundError:
        np_v = pa_v = None
    if np_v and pa_v and pa_v.startswith("26.") and np_v.startswith("1."):
        _die(f"refusing PyArrow {pa_v} with NumPy {np_v} (#302 crash pair)")
    print(f"INSTALLED OK: lm-eval=={got}")
    if N14_LOCK.is_file():
        for name, expect in (("mamba-ssm", EXPECT_MAMBA), ("causal-conv1d", EXPECT_CONV)):
            try:
                v = md.version(name)
            except md.PackageNotFoundError:
                _die(f"{name} is not installed", 3)
            if v != expect:
                _die(f"N-14 pin drifted: {name} {v}, expected {expect}")
        try:
            from transformers import NemotronHForCausalLM  # noqa: F401
        except Exception as e:
            _die(f"transformers NemotronHForCausalLM import failed: {e}")
        print(f"INSTALLED OK: mamba-ssm=={EXPECT_MAMBA} causal-conv1d=={EXPECT_CONV}")
        print("INSTALLED OK: transformers.NemotronHForCausalLM (CPU class import; no GPU load)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-only", action="store_true")
    ap.add_argument("--require-installed", action="store_true")
    args = ap.parse_args()
    check_lock()
    if args.lock_only:
        print("GATE PASS (lock-only; no image was built or published)")
        return 0
    installed = False
    try:
        import importlib.metadata as md
        md.version("lm-eval")
        installed = True
    except Exception:
        installed = False
    if args.require_installed and not installed:
        _die("lm-eval is not installed", 3)
    if installed:
        check_installed()
        print("GATE PASS (lock + installed)")
        return 0
    if args.require_installed:
        _die("lm-eval is not installed", 3)
    print("GATE PASS (lock-only; lm-eval is not installed in this environment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
