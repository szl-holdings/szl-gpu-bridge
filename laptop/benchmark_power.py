"""Integrate sampled power without turning missing readings into zero energy."""

import math
import threading
import time


class PowerSampler:
    def __init__(self, read_watts, interval=1.0, clock=time.monotonic):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.read_watts = read_watts
        self.interval = interval
        self.clock = clock
        self.samples = 0
        self.errors = 0
        self.joules = 0.0
        self.covered_seconds = 0.0
        self.last = None
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None
        self.ended = None

    def sample(self):
        now = self.clock()
        try:
            watts = float(self.read_watts())
            if not math.isfinite(watts) or watts < 0:
                raise ValueError("invalid power reading")
        except Exception:
            self.errors += 1
            self.last = None  # never integrate across an unknown interval
            return
        if self.last is not None:
            then, previous_watts = self.last
            duration = now - then
            if duration < 0:
                self.errors += 1
                self.last = None
                return
            self.joules += (previous_watts + watts) * 0.5 * duration
            self.covered_seconds += duration
        self.last = (now, watts)
        self.samples += 1

    def start(self):
        if self.started is not None:
            raise RuntimeError("sampler cannot be restarted")
        self.started = self.clock()
        self.sample()

        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.started is None:
            raise RuntimeError("sampler was not started")
        if self.ended is not None:
            return
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, 2 * self.interval))
        if self.thread.is_alive():
            self.errors += 1
        else:
            self.sample()
        self.ended = self.clock()

    def report(self, tokens_estimated):
        if self.ended is None:
            raise RuntimeError("stop the sampler before reporting")
        complete = self.errors == 0 and self.samples >= 2 and self.covered_seconds > 0
        joules = self.joules if complete else None
        return {
            "status": "MEASURED_POWER_INTEGRATED" if complete else "INCOMPLETE",
            "measurement_kind": "estimate_from_sampled_power_not_energy_counter",
            "joules_measured": joules,
            "method": "NVML power.draw; monotonic trapezoidal integration during training",
            "sampling_interval_seconds": self.interval,
            "samples": self.samples,
            "sampling_errors": self.errors,
            "covered_seconds": self.covered_seconds,
            "wall_seconds": self.ended - self.started,
            "avg_watts": joules / self.covered_seconds if complete else None,
            "kwh": joules / 3.6e6 if complete else None,
            "tokens_estimated": tokens_estimated,
            "joules_per_token": (
                joules / tokens_estimated if complete and tokens_estimated > 0 else None
            ),
            "co2_kg_estimated_us_average": joules / 3.6e6 * 0.385 if complete else None,
            "carbon_note": "Scenario using 0.385 kgCO2/kWh; not verified node-region carbon intensity",
        }
