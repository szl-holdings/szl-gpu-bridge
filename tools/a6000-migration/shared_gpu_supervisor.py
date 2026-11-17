#!/usr/bin/env python3
"""Bound one training process group while preserving a local serving workload.

Linux only. The lock coordinates cooperating launchers; the memory monitor also
detects pressure from other workloads. No process other than this launch's child
group is signalled. State deliberately excludes commands and environment values.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def health_responds(url):
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect
        )
        with opener.open(url, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def free_mib(gpu_index):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_index),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    value = int(result.stdout.strip())
    if value < 0:
        raise ValueError("negative GPU free memory")
    return value


def write_state(path, state, **updates):
    state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_child(child, grace_seconds):
    """The child is a session leader created by this supervisor, never a lookup."""
    if child.poll() is not None:
        return child.returncode, False
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The owned child can exit between poll() and signal delivery; reap it below.
        pass
    try:
        return child.wait(timeout=grace_seconds), False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            # It completed during the grace-period race; wait() still reaps it.
            pass
        return child.wait(), True


def supervise(args):
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.lock_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = args.state_dir / "state.json"
    state = {
        "schema": "szl.shared-gpu-supervisor/v1",
        "supervisor_pid": os.getpid(),
        "gpu_index": args.gpu_index,
        "child_pid": None,
        "child_returncode": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    lock = os.open(args.lock_file, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    child = None
    previous_handlers = {}
    requested_signal = []
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A duplicate launch must not overwrite the admitted run's state.
            write_state(
                path.with_name(f"refused-{os.getpid()}.json"),
                state,
                status="refused",
                reason="gpu_lock_busy",
            )
            return 75
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(
                sig, lambda number, frame: requested_signal.append(number)
            )
        try:
            available = free_mib(args.gpu_index)
        except (OSError, ValueError, subprocess.SubprocessError):
            write_state(
                path, state, status="refused", reason="gpu_telemetry_unavailable"
            )
            return 75
        ready = health_responds(args.health_url)
        reason = (
            "serving_unhealthy"
            if not ready
            else "insufficient_gpu_headroom"
            if available < args.min_free_mib
            else None
        )
        if reason or requested_signal:
            write_state(
                path,
                state,
                status="refused",
                reason=reason or "supervisor_interrupted",
                free_mib=available,
                serving_healthy=ready,
            )
            return 75
        started = time.monotonic()
        child = subprocess.Popen(
            args.command,
            shell=False,
            start_new_session=True,
            close_fds=True,
            stdin=subprocess.DEVNULL,
        )
        bad_health = low_memory = 0
        write_state(path, state, status="running", reason=None, child_pid=child.pid)
        while True:
            returncode = child.poll()
            if returncode is not None:
                write_state(
                    path,
                    state,
                    status="exited",
                    reason="child_exited",
                    child_returncode=returncode,
                )
                return returncode if returncode >= 0 else 128 - returncode
            reason = None
            # Native signal callbacks can mutate this list between any two bytecodes.
            if requested_signal:
                reason = "supervisor_interrupted"
            elif time.monotonic() - started >= args.max_seconds:
                reason = "deadline_exceeded"
            else:
                ready = health_responds(args.health_url)
                bad_health = 0 if ready else bad_health + 1
                try:
                    available = free_mib(args.gpu_index)
                    low_memory = (
                        low_memory + 1 if available < args.abort_free_mib else 0
                    )
                except (OSError, ValueError, subprocess.SubprocessError):
                    available = None
                    reason = "gpu_telemetry_lost"
                if bad_health >= args.health_failures:
                    reason = "serving_health_lost"
                elif low_memory >= 2:
                    reason = "gpu_memory_pressure"
                write_state(
                    path,
                    state,
                    free_mib=available,
                    serving_healthy=ready,
                    consecutive_health_failures=bad_health,
                    consecutive_low_memory_samples=low_memory,
                )
            if reason:
                write_state(path, state, status="stopping", reason=reason)
                returncode, forced = stop_child(child, args.grace_seconds)
                write_state(
                    path,
                    state,
                    status="stopped",
                    child_returncode=returncode,
                    forced_kill=forced,
                )
                return 124 if reason == "deadline_exceeded" else 75
            time.sleep(args.poll_seconds)
    except Exception as exc:
        returncode, forced = (
            stop_child(child, args.grace_seconds)
            if child is not None
            else (None, False)
        )
        write_state(
            path,
            state,
            status="error",
            reason="supervisor_error",
            error_type=type(exc).__name__,
            child_returncode=returncode,
            forced_kill=forced,
        )
        return 70
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        os.close(lock)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=32768)
    parser.add_argument("--abort-free-mib", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--health-failures", type=int, default=3)
    parser.add_argument("--grace-seconds", type=float, default=90)
    parser.add_argument("--max-seconds", type=float, default=21600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    url = urllib.parse.urlsplit(args.health_url)
    if (
        url.scheme not in ("http", "https")
        or url.hostname not in ("localhost", "127.0.0.1", "::1")
        or url.username
        or url.password
    ):
        parser.error("health URL must address a loopback server without credentials")
    if not args.command or args.gpu_index < 0:
        parser.error("provide a command and a nonnegative GPU index")
    for name in (
        "min_free_mib",
        "abort_free_mib",
        "poll_seconds",
        "health_failures",
        "grace_seconds",
        "max_seconds",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.abort_free_mib >= args.min_free_mib:
        parser.error("abort-free-mib must be below min-free-mib")
    return args


if __name__ == "__main__":
    raise SystemExit(supervise(parse_args()))
