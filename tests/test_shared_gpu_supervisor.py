"""GPU-free supervision tests, including ownership against real Linux processes."""

from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/a6000-migration/shared_gpu_supervisor.py"
)
g = None
if sys.platform == "linux":
    SPEC = importlib.util.spec_from_file_location("shared_gpu_supervisor", PATH)
    g = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(g)


class FakeChild:
    pid = 987654

    def __init__(self, stubborn=False):
        self.returncode = None
        self.stubborn = stubborn
        self.signals = []

    def poll(self):
        return self.returncode

    def killpg(self, pid, sig):
        if pid != self.pid:
            raise AssertionError("attempted to signal an unowned group")
        self.signals.append(sig)
        if sig == signal.SIGKILL or not self.stubborn:
            self.returncode = -sig

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("owned test child", timeout)
        return self.returncode


@unittest.skipUnless(sys.platform == "linux", "Linux flock and process groups")
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            lock_file=self.root / "gpu0.lock",
            state_dir=self.root / "run",
            health_url="http://127.0.0.1:8081/health",
            gpu_index=0,
            min_free_mib=32768,
            abort_free_mib=4096,
            poll_seconds=1,
            health_failures=3,
            grace_seconds=1,
            max_seconds=10,
            command=["python", "worker.py"],
        )
        self.child = FakeChild()

    def state(self):
        return json.loads((self.args.state_dir / "state.json").read_text())

    def run_mocked(self, *, free=40000, health=True, finish_after=None):
        clock = [0]

        def sleep(seconds):
            clock[0] += seconds
            if finish_after is not None and clock[0] >= finish_after:
                self.child.returncode = 0

        with ExitStack() as stack:
            factory = stack.enter_context(
                patch.object(g.subprocess, "Popen", return_value=self.child)
            )
            stack.enter_context(
                patch.object(
                    g,
                    "free_mib",
                    side_effect=free if isinstance(free, list) else None,
                    return_value=free,
                )
            )
            stack.enter_context(
                patch.object(
                    g,
                    "health_responds",
                    side_effect=health if isinstance(health, list) else None,
                    return_value=health,
                )
            )
            stack.enter_context(patch.object(g.time, "sleep", side_effect=sleep))
            stack.enter_context(
                patch.object(g.time, "monotonic", side_effect=lambda: clock[0])
            )
            stack.enter_context(
                patch.object(g.os, "killpg", side_effect=self.child.killpg)
            )
            result = g.supervise(self.args)
        return result, factory

    def test_preflight_low_memory_never_starts_child(self):
        rc, factory = self.run_mocked(free=32767)
        self.assertEqual(rc, 75)
        factory.assert_not_called()
        self.assertEqual(self.state()["reason"], "insufficient_gpu_headroom")

    def test_preflight_unhealthy_never_starts_child(self):
        rc, factory = self.run_mocked(health=False)
        self.assertEqual(rc, 75)
        factory.assert_not_called()
        self.assertEqual(self.state()["reason"], "serving_unhealthy")

    def test_preflight_telemetry_failure_never_starts_child(self):
        rc, factory = self.run_mocked(free=[OSError("GPU query unavailable")])
        self.assertEqual(rc, 75)
        factory.assert_not_called()
        self.assertEqual(self.state()["reason"], "gpu_telemetry_unavailable")

    def test_lock_conflict_does_not_replace_running_state(self):
        self.args.state_dir.mkdir()
        (self.args.state_dir / "state.json").write_text('{"status":"running"}')
        with self.args.lock_file.open("w") as lock:
            g.fcntl.flock(lock, g.fcntl.LOCK_EX | g.fcntl.LOCK_NB)
            rc, factory = self.run_mocked()
        self.assertEqual(rc, 75)
        factory.assert_not_called()
        self.assertEqual(self.state(), {"status": "running"})
        refusal = list(self.args.state_dir.glob("refused-*.json"))
        self.assertEqual(json.loads(refusal[0].read_text())["reason"], "gpu_lock_busy")

    def test_recovered_health_and_memory_do_not_accumulate(self):
        rc, factory = self.run_mocked(
            free=[40000, 3000, 40000, 3000, 40000],
            health=[True, False, False, True, False],
            finish_after=4,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(self.child.signals, [])
        self.assertEqual(self.state()["child_returncode"], 0)
        call = factory.call_args
        self.assertFalse(call.kwargs["shell"])
        self.assertTrue(call.kwargs["start_new_session"])
        self.assertTrue(call.kwargs["close_fds"])
        self.assertNotIn("env", call.kwargs)

    def test_three_consecutive_health_failures_stop_owned_group(self):
        rc, _ = self.run_mocked(health=[True, False, False, False])
        self.assertEqual(rc, 75)
        self.assertEqual(self.state()["reason"], "serving_health_lost")
        self.assertEqual(self.child.signals, [signal.SIGTERM])

    def test_two_low_memory_samples_stop_owned_group(self):
        rc, _ = self.run_mocked(free=[40000, 4095, 4095])
        self.assertEqual(rc, 75)
        self.assertEqual(self.state()["reason"], "gpu_memory_pressure")
        self.assertEqual(self.state()["child_returncode"], -signal.SIGTERM)

    def test_telemetry_failure_after_launch_stops_child(self):
        rc, _ = self.run_mocked(
            free=[40000, subprocess.TimeoutExpired("nvidia-smi", 5)]
        )
        self.assertEqual(rc, 75)
        self.assertEqual(self.state()["reason"], "gpu_telemetry_lost")

    def test_deadline_escalates_only_owned_group_after_grace(self):
        self.child = FakeChild(stubborn=True)
        rc, _ = self.run_mocked()
        self.assertEqual(rc, 124)
        self.assertEqual(self.child.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(self.state()["reason"], "deadline_exceeded")
        self.assertTrue(self.state()["forced_kill"])

    def test_no_command_or_secret_environment_in_state(self):
        self.args.command.append("sentinel-command-secret")
        with patch.dict(os.environ, HF_TOKEN="sentinel-environment-secret"):
            self.run_mocked(finish_after=1)
        text = (self.args.state_dir / "state.json").read_text()
        self.assertNotIn("sentinel", text)
        self.assertNotIn("command", text)
        self.assertEqual(
            (self.args.state_dir / "state.json").stat().st_mode & 0o777, 0o600
        )

    @unittest.skipUnless(sys.platform == "linux", "Linux process groups")
    def test_real_termination_preserves_unrelated_process(self):
        self.args.command = [sys.executable, "-c", "import time; time.sleep(30)"]
        self.args.poll_seconds = 0.01
        self.args.max_seconds = 1
        unrelated = subprocess.Popen(self.args.command, start_new_session=True)
        try:
            with (
                patch.object(g, "free_mib", return_value=40000),
                patch.object(
                    g, "health_responds", side_effect=[True, False, False, False]
                ),
            ):
                self.assertEqual(g.supervise(self.args), 75)
            self.assertIsNone(unrelated.poll())
            state = self.state()
            self.assertEqual(state["child_returncode"], -signal.SIGTERM)
            with self.assertRaises(ProcessLookupError):
                os.kill(state["child_pid"], 0)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)


@unittest.skipUnless(sys.platform == "linux", "Linux supervisor module")
class ParsingTests(unittest.TestCase):
    def arguments(self, *extra):
        return [
            "--lock-file",
            "/tmp/gpu0.lock",
            "--state-dir",
            "/tmp/run",
            "--health-url",
            "http://127.0.0.1:8081/health",
            *extra,
            "--",
            "python",
            "worker.py",
            "--max-steps",
            "200",
        ]

    def test_separator_preserves_worker_arguments(self):
        args = g.parse_args(self.arguments())
        self.assertEqual(args.command, ["python", "worker.py", "--max-steps", "200"])

    def test_nan_and_external_health_are_rejected(self):
        for extra in [
            ["--poll-seconds", "nan"],
            ["--health-url", "https://example.com/health"],
            ["--health-url", "http://secret@localhost/health"],
        ]:
            with (
                self.subTest(extra=extra),
                patch("sys.stderr", Mock()),
                self.assertRaises(SystemExit),
            ):
                g.parse_args(self.arguments(*extra))

    def test_malformed_gpu_telemetry_fails_closed(self):
        with patch.object(
            g.subprocess, "run", return_value=SimpleNamespace(stdout="N/A\n")
        ):
            with self.assertRaises(ValueError):
                g.free_mib(0)


if __name__ == "__main__":
    unittest.main()
