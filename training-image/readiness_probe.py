#!/usr/bin/env python3
"""readiness_probe.py - import gate, GPU smoke, step heartbeat, checkpoint integrity.

Rikuy, 2026-11-19. LOCAL ONLY. Standard library except where a mode says otherwise
(torch/safetensors are imported lazily, only in the modes that need them).

Modes (exit 0 = pass; non-zero = fail; every mode prints one JSON line to stdout):
  imports  --expect pkg=ver ... [--from-env] [--no-gpu]
           Imports torch/transformers/peft/trl/datasets, prints IMPORTS_OK <versions>,
           fails (exit 3) on any import error or version != expected pin. --from-env reads
           SZL_EXPECT_TRANSFORMERS / _PEFT / _TRL. Unless --no-gpu, also runs a bf16 matmul
           on the GPU (exit 4 if CUDA unavailable or smoke fails).
  beat     --file F --step N [--max-steps M] [--run-id R]
           Writes the heartbeat atomically (tmp + rename). Call from a Trainer callback on_step_end.
           Also importable: from readiness_probe import write_heartbeat.
  live     --file F [--max-age-seconds 900]
           Liveness: heartbeat file exists and is fresh. Exit 5 stale/missing.
  ready    --file F [--max-age-seconds 900]
           Readiness: as live, plus step is monotonic with the previous probe (state kept in
           F.probe). A replica that is "alive" but stuck on one step fails here (exit 6).
  ckpt     --dir D [--expect-steps N] [--write-manifest | --verify]
           Checkpoint integrity. Reads trainer_state.json global_step, computes sha256 of every file,
           writes/verifies MANIFEST.sha256. --expect-steps N: fail (exit 7) if global_step != N, so
           "25000 steps claimed" cannot pass with a checkpoint at 11250.
  final    --final D1 --ckpt D2
           Fails (exit 8) if the final adapter weights are byte-identical to the checkpoint's
           (the "final == checkpoint-200" case), unless trainer_state says that checkpoint IS the last step.
  lora     --adapter D
           LoRA no-op quarantine: needs safetensors. Fails (exit 9) if every lora_B tensor is all zero
           (adapter would not change the base model).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def out(ok: bool, code: int, **kw):
    kw.update(ok=ok, exit=code, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(json.dumps(kw, sort_keys=True))
    return 0 if ok else code


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_heartbeat(path: str, step: int, max_steps: int | None = None, run_id: str | None = None) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"step": int(step), "max_steps": max_steps, "run_id": run_id,
                               "unix": time.time(), "pid": os.getpid()}))
    os.replace(tmp, p)


def read_heartbeat(path: str):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# ---------------------------------------------------------------- imports
def cmd_imports(a):
    expect = dict(e.split("=", 1) for e in a.expect)
    if a.from_env:
        for pkg, var in (("transformers", "SZL_EXPECT_TRANSFORMERS"), ("peft", "SZL_EXPECT_PEFT"), ("trl", "SZL_EXPECT_TRL")):
            if os.environ.get(var):
                expect[pkg] = os.environ[var]
    versions, errors = {}, {}
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # ModuleNotFoundError is the failure this gate exists to catch
            errors[name] = f"{type(e).__name__}: {e}"
    if errors:
        return out(False, 3, gate="imports", errors=errors, versions=versions)
    drift = {k: {"want": v, "got": versions.get(k)} for k, v in expect.items() if versions.get(k) != v}
    if drift:
        return out(False, 3, gate="imports", drift=drift, versions=versions)
    if not a.no_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                return out(False, 4, gate="gpu", error="cuda unavailable", versions=versions)
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            if not bool(torch.isfinite(x @ x).all()):
                return out(False, 4, gate="gpu", error="non-finite bf16 matmul", versions=versions)
        except Exception as e:
            return out(False, 4, gate="gpu", error=f"{type(e).__name__}: {e}", versions=versions)
    print("IMPORTS_OK " + " ".join(f"{k}={v}" for k, v in versions.items()), file=sys.stderr)
    return out(True, 0, gate="imports", versions=versions, gpu_checked=not a.no_gpu)


# ---------------------------------------------------------------- heartbeat
def cmd_beat(a):
    write_heartbeat(a.file, a.step, a.max_steps, a.run_id)
    return out(True, 0, gate="beat", step=a.step)


def _fresh(a):
    hb = read_heartbeat(a.file)
    if not hb:
        return None, out(False, 5, gate="heartbeat", error="missing or unreadable", file=a.file)
    age = time.time() - float(hb.get("unix", 0))
    if age > a.max_age_seconds:
        return None, out(False, 5, gate="heartbeat", error="stale", age_seconds=round(age, 1), step=hb.get("step"))
    return (hb, age), None


def cmd_live(a):
    got, bad = _fresh(a)
    if bad is not None:
        return bad
    hb, age = got
    return out(True, 0, gate="live", step=hb.get("step"), age_seconds=round(age, 1))


def cmd_ready(a):
    got, bad = _fresh(a)
    if bad is not None:
        return bad
    hb, age = got
    state = Path(a.file + ".probe")
    prev = None
    try:
        prev = json.loads(state.read_text())
    except Exception:
        pass
    step = int(hb.get("step", -1))
    state.write_text(json.dumps({"step": step, "unix": time.time()}))
    done = hb.get("max_steps") is not None and step >= int(hb["max_steps"])
    if prev and not done and step <= int(prev.get("step", -1)) and time.time() - prev["unix"] > a.max_age_seconds / 3:
        return out(False, 6, gate="ready", error="step not advancing", step=step, prev_step=prev["step"])
    return out(True, 0, gate="ready", step=step, done=done)


# ---------------------------------------------------------------- checkpoints
def _files(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")


def cmd_ckpt(a):
    d = Path(a.dir)
    ts = d / "trainer_state.json"
    step = None
    if ts.is_file():
        step = json.loads(ts.read_text()).get("global_step")
    if a.expect_steps is not None and step != a.expect_steps:
        return out(False, 7, gate="ckpt", error="global_step mismatch", global_step=step, expected=a.expect_steps)
    man = d / "MANIFEST.sha256"
    if a.write_manifest:
        man.write_text("".join(f"{sha256_file(p)}  {p.relative_to(d)}\n" for p in _files(d)))
        return out(True, 0, gate="ckpt", wrote=str(man), files=len(_files(d)), global_step=step)
    if a.verify:
        if not man.is_file():
            return out(False, 7, gate="ckpt", error="no MANIFEST.sha256")
        bad = []
        for line in man.read_text().splitlines():
            h, rel = line.split("  ", 1)
            p = d / rel
            if not p.is_file() or sha256_file(p) != h:
                bad.append(rel)
        if bad:
            return out(False, 7, gate="ckpt", error="hash mismatch or missing", files=bad)
    return out(True, 0, gate="ckpt", global_step=step)


def cmd_final(a):
    f, c = Path(a.final), Path(a.ckpt)
    fw = next((p for p in (f / "adapter_model.safetensors", f / "adapter_model.bin") if p.is_file()), None)
    cw = next((p for p in (c / "adapter_model.safetensors", c / "adapter_model.bin") if p.is_file()), None)
    if not fw or not cw:
        return out(False, 8, gate="final", error="adapter weights missing", final=bool(fw), ckpt=bool(cw))
    same = sha256_file(fw) == sha256_file(cw)
    ts = c / "trainer_state.json"
    cstep = json.loads(ts.read_text()).get("global_step") if ts.is_file() else None
    max_steps = json.loads(ts.read_text()).get("max_steps") if ts.is_file() else None
    if same and not (cstep is not None and max_steps is not None and cstep == max_steps):
        return out(False, 8, gate="final", error="final adapter identical to a non-terminal checkpoint",
                   ckpt_step=cstep, max_steps=max_steps)
    return out(True, 0, gate="final", identical=same, ckpt_step=cstep, max_steps=max_steps)


def cmd_lora(a):
    try:
        from safetensors import safe_open
    except Exception as e:
        return out(False, 9, gate="lora", error=f"safetensors unavailable: {e}")
    p = Path(a.adapter) / "adapter_model.safetensors"
    if not p.is_file():
        return out(False, 9, gate="lora", error="no adapter_model.safetensors")
    nonzero = total = 0
    with safe_open(str(p), framework="pt") as f:
        for k in f.keys():
            if "lora_B" in k:
                total += 1
                if bool((f.get_tensor(k) != 0).any()):
                    nonzero += 1
    if total == 0 or nonzero == 0:
        return out(False, 9, gate="lora", error="LoRA no-op: every lora_B tensor is zero (or none found)", lora_b=total)
    return out(True, 0, gate="lora", lora_b=total, nonzero=nonzero)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("imports"); s.add_argument("--expect", action="append", default=[]); s.add_argument("--from-env", action="store_true"); s.add_argument("--no-gpu", action="store_true"); s.set_defaults(fn=cmd_imports)
    s = sp.add_parser("beat"); s.add_argument("--file", required=True); s.add_argument("--step", type=int, required=True); s.add_argument("--max-steps", type=int); s.add_argument("--run-id"); s.set_defaults(fn=cmd_beat)
    for n, fn in (("live", cmd_live), ("ready", cmd_ready)):
        s = sp.add_parser(n); s.add_argument("--file", default=os.environ.get("SZL_HEARTBEAT_FILE", "/tmp/szl_heartbeat.json")); s.add_argument("--max-age-seconds", type=float, default=900); s.set_defaults(fn=fn)
    s = sp.add_parser("ckpt"); s.add_argument("--dir", required=True); s.add_argument("--expect-steps", type=int); s.add_argument("--write-manifest", action="store_true"); s.add_argument("--verify", action="store_true"); s.set_defaults(fn=cmd_ckpt)
    s = sp.add_parser("final"); s.add_argument("--final", required=True); s.add_argument("--ckpt", required=True); s.set_defaults(fn=cmd_final)
    s = sp.add_parser("lora"); s.add_argument("--adapter", required=True); s.set_defaults(fn=cmd_lora)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
