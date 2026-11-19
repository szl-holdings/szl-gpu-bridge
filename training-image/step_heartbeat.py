"""step_heartbeat.py - local liveness heartbeat for the Salad liveness probe.

The probe is `python3 /work/readiness_probe.py live --file /tmp/szl_heartbeat.json --max-age-seconds 900`.
This module keeps that file fresh ONLY while training makes progress:

  * StepHeartbeatCallback.on_step_end rewrites it when the trainer's global_step has advanced AND at least
    SZL_HEARTBEAT_WRITE_SECONDS (default 60, clamped to 30..180) have passed since the last write.
    There is no background thread: a hung trainer stops writing and the file goes stale (probe fails at 900 s).
  * beat() is also called at a few setup milestones (tokenizer, dataset, model load, before train) so the
    startup probe can pass while a large model downloads. Milestone calls are one-shot, not periodic.
  * The file is local only: written to a temp file in the same directory, then os.replace()d.
    It is never uploaded to the Hub and never committed anywhere.

Heartbeat JSON fields (same as readiness_probe.write_heartbeat):
  step (int), max_steps (int|null), run_id (str|null), unix (float, time.time()), pid (int)

Failures to write are swallowed after one stderr line: the heartbeat must never stop training.
"""
from __future__ import annotations

import json
import os
import sys
import time

DEFAULT_PATH = "/tmp/szl_heartbeat.json"


def heartbeat_path() -> str:
    return os.environ.get("SZL_HEARTBEAT_FILE", DEFAULT_PATH)


def write_seconds() -> float:
    try:
        v = float(os.environ.get("SZL_HEARTBEAT_WRITE_SECONDS", "60"))
    except ValueError:
        v = 60.0
    return min(max(v, 30.0), 180.0)


def beat(step, max_steps=None, run_id=None, path=None) -> bool:
    """Atomically rewrite the heartbeat file. Returns True if written."""
    path = path or heartbeat_path()
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"step": int(step), "max_steps": None if max_steps is None else int(max_steps),
                       "run_id": run_id, "unix": time.time(), "pid": os.getpid()}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001 - never stop training over a heartbeat
        print(f"heartbeat write failed ({type(exc).__name__}); training continues", file=sys.stderr, flush=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def make_callback(max_steps=None, run_id=None, path=None, clock=time.monotonic):
    from transformers import TrainerCallback

    class StepHeartbeatCallback(TrainerCallback):
        def __init__(self):
            self.last_write = None
            self.last_step = None

        def _maybe(self, state, force=False):
            step = getattr(state, "global_step", None)
            if type(step) is not int or step < 0:
                return
            now = clock()
            advanced = self.last_step is None or step > self.last_step
            due = self.last_write is None or now - self.last_write >= write_seconds()
            if (force and advanced) or (advanced and due):
                if beat(step, max_steps if max_steps is not None else getattr(state, "max_steps", None), run_id, path):
                    self.last_write, self.last_step = now, step

        def on_train_begin(self, args, state, control, **kw):
            # resumed runs report their resumed step; a fresh run reports 0
            step = getattr(state, "global_step", 0)
            if type(step) is int and step >= 0 and beat(step, max_steps, run_id, path):
                self.last_write, self.last_step = clock(), step

        def on_step_end(self, args, state, control, **kw):
            self._maybe(state)
            return control

        def on_train_end(self, args, state, control, **kw):
            self._maybe(state, force=True)

    return StepHeartbeatCallback()
