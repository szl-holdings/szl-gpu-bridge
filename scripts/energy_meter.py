#!/usr/bin/env python3
"""energy_meter.py — hardened NVML energy sampler for the SZL GPU mesh.

Why this exists: the A6000 bench leg reported 0 joules because the producer
called an NVML function name the installed binding does not provide
(`nvmlDeviceGetPower` — not a real pynvml symbol) and converted the resulting
exception into zero. Zero is not evidence of zero energy use. Doctrine Law 6:
fail closed. Failed measurement is UNAVAILABLE (None), never zero.

Features:
  - Works with both common bindings: `pynvml` (classic) and
    `nvidia-ml-py` (the package that installs as `pynvml` on modern pip —
    both expose the same symbols, but classic pynvml lacks
    nvmlDeviceGetTotalEnergyConsumption on some versions).
  - Prefers the total-energy counter (mJ -> J); rejects observed counter resets.
  - Falls back to power sampling (nvmlDeviceGetPowerUsage, mW) integrated
    over monotonic elapsed time. Integration is a sampled estimate, not calibration.
  - Records sample success/error counts and the distinct errors seen, so a
    receipt can never silently claim MEASURED from a dead meter.
  - A failed or incomplete measurement window returns None, never a partial total.
  - A meter instance records one window; repeated stop() returns the same snapshot.
  - CLI self-test: `python3 energy_meter.py --self-test` prints a receipt
    (works without a GPU — reports UNAVAILABLE honestly).
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time


class EnergyMeter:
    """Samples GPU energy. Never fabricates: no successful sample -> None."""

    def __init__(self, index: int = 0):
        self.index = index
        self.samples_ok = 0
        self.samples_err = 0
        self.errors: list[str] = []
        self._nvml = None
        self._handle = None
        self._mode = None  # "total-energy" | "power-integral" | None
        self._j0 = None
        self._t0 = None
        self._integral_j = 0.0
        self._last_t = None
        self._last_w = None
        self._intervals = 0
        self._started = False
        self._receipt = None
        self._init()

    # -- setup -----------------------------------------------------------
    def _record_err(self, exc: Exception) -> None:
        self.samples_err += 1
        msg = f"{type(exc).__name__}: {exc}"
        if msg not in self.errors:
            self.errors.append(msg)
        if len(self.errors) > 16:
            self.errors = self.errors[:16]

    def _init(self) -> None:
        try:
            import pynvml  # classic binding or nvidia-ml-py shim
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as exc:  # no GPU, no driver, no binding
            self._record_err(exc)
            return
        # Prefer the monotonic total-energy counter (millijoules). Probe by
        # calling once (harmless on a monotonic counter); fall back to power
        # integration if the symbol is missing or broken. Probe errors are
        # capability facts, not sample failures — they don't count.
        try:
            self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self._mode = "total-energy"
        except Exception:
            try:
                self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self._mode = "power-integral"
            except Exception as exc:
                self._record_err(exc)

    # -- sampling --------------------------------------------------------
    def start(self) -> None:
        if self._started:
            raise RuntimeError("energy meter already started; use a new instance")
        self._started = True
        self._t0 = time.monotonic()
        self._last_t = self._t0
        if self._mode == "total-energy":
            self._j0 = self._read_raw()
        elif self._mode == "power-integral":
            self._last_w = self._read_raw()

    def _read_raw(self):
        """One guarded, validated NVML read; successful samples are finite and nonnegative."""
        if self._nvml is None or self._handle is None or self._mode is None:
            return None
        try:
            if self._mode == "total-energy":
                raw = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            else:
                raw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
            # bool is an int subclass but cannot be a physical measurement.
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("NVML sample is not numeric")
            if not math.isfinite(raw) or raw < 0:
                raise ValueError("NVML sample is not finite and nonnegative")
            value = raw / 1000.0  # mJ -> J or mW -> W
            self.samples_ok += 1
            return value
        except Exception as exc:
            self._record_err(exc)
            return None

    def tick(self) -> None:
        """Accumulate sampled power; a missing interval invalidates the whole window."""
        if not self._started or self._receipt is not None:
            raise RuntimeError("energy meter has no active window")
        if self._mode != "power-integral":
            return
        now = time.monotonic()
        w = self._read_raw()
        if w is None or self._last_w is None or self._last_t is None:
            self._last_t, self._last_w = now, w
            return
        dt = now - self._last_t
        if not math.isfinite(dt) or dt < 0:
            self._record_err(ValueError("invalid monotonic sampling interval"))
        elif dt > 0:
            increment = (0.5 * self._last_w + 0.5 * w) * dt
            total = self._integral_j + increment
            if not math.isfinite(total):
                self._record_err(ValueError("non-finite integrated energy"))
            else:
                self._integral_j = total
                self._intervals += 1
        self._last_t, self._last_w = now, w

    def stop(self) -> dict:
        """Finish exactly one window. Missing samples cannot certify a partial total."""
        if not self._started:
            raise RuntimeError("energy meter must be started before stopping")
        if self._receipt is not None:
            return copy.deepcopy(self._receipt)
        energy = None
        if self._mode == "total-energy":
            j1 = self._read_raw()
            if self._j0 is not None and j1 is not None:
                if j1 < self._j0:
                    self._record_err(ValueError("energy counter decreased during window"))
                elif self.samples_err == 0:
                    energy = round(j1 - self._j0, 3)
        elif self._mode == "power-integral":
            self.tick()
            if self.samples_err == 0 and self._intervals > 0:
                # A genuine zero from valid samples is not an unavailable reading.
                energy = round(self._integral_j, 3)
        elapsed = time.monotonic() - self._t0
        if not math.isfinite(elapsed) or elapsed < 0:
            self._record_err(ValueError("invalid monotonic measurement window"))
            energy, elapsed = None, None
        self._receipt = {
            "energy_joules": energy,
            "energy_state": "MEASURED" if energy is not None else "UNAVAILABLE",
            "energy_mode": self._mode,
            "samples_ok": self.samples_ok,
            "samples_err": self.samples_err,
            "errors": list(self.errors),
            "wall_seconds": round(elapsed, 3) if elapsed is not None else None,
        }
        return copy.deepcopy(self._receipt)


def self_test() -> dict:
    """Honest self-test without requiring a GPU. A dead meter must report
    UNAVAILABLE with the failure reason, never zero joules."""
    meter = EnergyMeter()
    meter.start()
    meter.tick()
    receipt = meter.stop()
    ok = receipt["energy_state"] in ("MEASURED", "UNAVAILABLE")
    ok = ok and (receipt["energy_joules"] is None or (
        math.isfinite(receipt["energy_joules"]) and receipt["energy_joules"] >= 0))
    ok = ok and not (receipt["samples_err"] > 0 and receipt["energy_state"] == "MEASURED")
    return {"self_test": "PASS" if ok else "FAIL", **receipt}


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = self_test()
        print(json.dumps(result, indent=2, allow_nan=False))
        sys.exit(0 if result["self_test"] == "PASS" else 1)
    print(__doc__)
