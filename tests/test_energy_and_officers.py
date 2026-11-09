#!/usr/bin/env python3
"""Offline tests for the energy meter and officer fixes. No GPU required."""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from energy_meter import EnergyMeter, self_test  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


gardener = load_module("gardener_app", ROOT / "hf-spaces" / "estate-gardener" / "app.py")
auditor = load_module("auditor_app", ROOT / "hf-spaces" / "eternal-auditor" / "app.py")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")

# ---------- energy_meter: dead meter reports UNAVAILABLE, never zero ----------
print("== energy_meter: no GPU ==")
r = self_test()
check("self-test passes", r["self_test"] == "PASS")
check("no GPU -> UNAVAILABLE", r["energy_state"] == "UNAVAILABLE")
check("no GPU -> null joules (not zero)", r["energy_joules"] is None)
check("failure reason recorded", r["samples_err"] >= 0 and r["energy_mode"] is None)

# ---------- energy_meter: fake NVML, total-energy mode ----------
print("== energy_meter: fake total-energy NVML ==")
fake = types.ModuleType("pynvml")
calls = {"energy": [1000.0, 2000.0, 3500.0]}  # mJ: probe, start, stop
def nvmlInit(): pass
def nvmlDeviceGetHandleByIndex(i): return ("handle", i)
def nvmlDeviceGetTotalEnergyConsumption(h): return calls["energy"].pop(0) if calls["energy"] else 2500.0
def nvmlDeviceGetPowerUsage(h): raise RuntimeError("should not be called in total-energy mode")
fake.nvmlInit, fake.nvmlDeviceGetHandleByIndex = nvmlInit, nvmlDeviceGetHandleByIndex
fake.nvmlDeviceGetTotalEnergyConsumption = nvmlDeviceGetTotalEnergyConsumption
fake.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
sys.modules["pynvml"] = fake
m = EnergyMeter(); m.start(); r = m.stop()
check("mode total-energy", r["energy_mode"] == "total-energy")
check("measured 1.5 J", r["energy_joules"] == 1.5)
check("state MEASURED", r["energy_state"] == "MEASURED")
check("samples ok", r["samples_ok"] >= 2 and r["samples_err"] == 0)

# ---------- energy_meter: classic binding without total-energy -> power fallback ----------
print("== energy_meter: power-integral fallback ==")
fake2 = types.ModuleType("pynvml")
fake2.nvmlInit = nvmlInit
fake2.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
def boom(h): raise AttributeError("no such symbol")
fake2.nvmlDeviceGetTotalEnergyConsumption = boom
watts = {"w": 300.0}
def get_power(h): return watts["w"]
fake2.nvmlDeviceGetPowerUsage = get_power
sys.modules["pynvml"] = fake2
import time as _time
m2 = EnergyMeter(); m2.start()
for _ in range(5):
    _time.sleep(0.02)
    m2.tick()
r2 = m2.stop()
check("mode power-integral", r2["energy_mode"] == "power-integral")
check("measured > 0 J", r2["energy_joules"] is not None and r2["energy_joules"] > 0)
check("state MEASURED", r2["energy_state"] == "MEASURED")

# ---------- energy_meter: meter dies mid-run -> UNAVAILABLE, never zero ----------
print("== energy_meter: dead mid-run ==")
fake3 = types.ModuleType("pynvml")
fake3.nvmlInit = nvmlInit
fake3.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
fake3.nvmlDeviceGetTotalEnergyConsumption = boom
def dead(h): raise RuntimeError("NVML not initialized")
fake3.nvmlDeviceGetPowerUsage = dead
sys.modules["pynvml"] = fake3
m3 = EnergyMeter(); m3.start(); m3.tick(); r3 = m3.stop()
check("dead meter -> UNAVAILABLE", r3["energy_state"] == "UNAVAILABLE")
check("dead meter -> null joules", r3["energy_joules"] is None)
check("errors recorded", len(r3["errors"]) > 0 and r3["samples_err"] > 0)

del sys.modules["pynvml"]

# ---------- gardener: append-only, content-preserving ----------
print("== gardener: append-only ==")
with tempfile.TemporaryDirectory() as td:
    os.environ["GARDENER_STATE"] = os.path.join(td, "state.json")
    gardener = load_module("gardener_app", ROOT / "hf-spaces" / "estate-gardener" / "app.py")
    original = "# My Dataset\n\nCurated by Dr. Sara Illion, CC-BY-4.0 requested.\nReal content here."
    receipt, new = gardener.garden_readme("test/ds", original, has_license_tag=False)
    check("action APPEND", receipt["action"] == "APPEND")
    check("original preserved byte-for-byte", new.startswith(original))
    check("nothing removed (superset)", len(new) > len(original))
    check("idempotent: tagged README skipped",
          gardener.append_license_block(original, True)[1] == "SKIP_ALREADY_TAGGED")
    receipt2, _ = gardener.garden_readme("test/ds", new, has_license_tag=True)
    check("second pass skips", receipt2["action"] == "SKIP_ALREADY_TAGGED")
    empty_r, empty_a = gardener.append_license_block("", False)
    check("empty README gets block without loss", empty_a == "WRITE_EMPTY" and empty_r.strip())

# ---------- auditor: full signatures + chain persistence ----------
print("== auditor: full signatures + chain ==")
with tempfile.TemporaryDirectory() as td:
    os.environ["AUDITOR_STATE"] = os.path.join(td, "audit-state.json")
    auditor = load_module("auditor_app", ROOT / "hf-spaces" / "eternal-auditor" / "app.py")
    r1 = auditor.run_cycle([{"target": "x", "check": "y", "result": "PASS"}], secret="s3cret")
    check("signature is full 64-hex", len(r1["signature"]) == 64 and r1["signature_length"] == 64)
    check("genesis has no prev", r1["prev_receipt_sha256"] is None)
    auditor = load_module("auditor_app", ROOT / "hf-spaces" / "eternal-auditor" / "app.py")  # restart: state must persist
    r2 = auditor.run_cycle([{"target": "x", "check": "y", "result": "PASS"}], secret="s3cret")
    check("cycle continues after restart", r2["cycle"] == 2)
    check("chain links to previous", r2["prev_receipt_sha256"] == r1["body_sha256"])
    check("chain marked CONTINUOUS", r2["chain_state"] == "CONTINUOUS")
    # verify signature reproducibility
    import hmac, hashlib
    expect = hmac.new(b"s3cret", r2["body_sha256"].encode(), hashlib.sha256).hexdigest()
    check("signature verifies", expect == r2["signature"])

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
