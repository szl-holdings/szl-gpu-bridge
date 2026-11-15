#!/usr/bin/env python3
"""Regression tests for vllm_smoke_test.sh against the loopback stub.

No GPU, model or live server is used: nvidia-smi is a stub on PATH, HOME and
the output directory are temporary, and requests go only to the stub on an
OS-assigned 127.0.0.1 port (never 8000).
Run: python3 -m unittest discover -s tools/evals/tests -v
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = pathlib.Path(os.environ.get("SMOKE_SCRIPT", HERE.parent / "vllm_smoke_test.sh"))
STUB = HERE / "stub_chat_server.py"
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Stub:
    def __init__(self, workdir, mode):
        port_file = pathlib.Path(workdir) / f"stub-{mode}.port"
        self.proc = subprocess.Popen([sys.executable, str(STUB), "--mode", mode,
                                      "--delay", "0.05", "--port-file", str(port_file)])
        deadline = time.monotonic() + 10
        while not port_file.exists():
            if time.monotonic() > deadline or self.proc.poll() is not None:
                self.proc.kill()
                raise RuntimeError("stub did not start")
            time.sleep(0.05)
        self.port = int(port_file.read_text())
        if self.port == 8000:
            self.close()
            raise RuntimeError("stub must never use port 8000")

    def stats(self):
        with NO_PROXY.open(f"http://127.0.0.1:{self.port}/stats", timeout=5) as r:
            return json.load(r)

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)


def run_script(port, home, cwd, extra_env=None):
    bin_dir = pathlib.Path(home) / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'stub-gpu, 0.0, 0 MiB'\n")
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "all_proxy", "szl_eval_out")}
    env["HOME"] = str(home)
    env["PATH"] = os.pathsep.join([str(bin_dir), str(pathlib.Path(sys.executable).parent),
                                   env.get("PATH", "")])
    env.update(extra_env or {})
    return subprocess.run(["bash", str(SCRIPT), str(port)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


class Run:
    """One script run against a fresh stub."""

    def __init__(self, mode, seed_stale=False, out_override=False):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.home, self.cwd = root / "home", root / "cwd"
        self.home.mkdir()
        self.cwd.mkdir()
        self.out = root / "custom-out" if out_override else self.home / "szl-evals" / "vllm-smoke"
        if seed_stale:
            self.out.mkdir(parents=True)
            (self.out / "results.json").write_text(json.dumps({"verdict": "STALE"}))
        stub = Stub(root, mode)
        try:
            self.port = stub.port
            extra = {"SZL_EVAL_OUT": str(self.out)} if out_override else None
            self.proc = run_script(stub.port, self.home, self.cwd, extra)
            self.stats = stub.stats()
        finally:
            stub.close()
        path = self.out / "results.json"
        self.result = json.loads(path.read_text()) if path.exists() else {}


class WrongAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = Run("wrong", seed_stale=True)

    @classmethod
    def tearDownClass(cls):
        cls.smoke.tmp.cleanup()

    def test_exit_status(self):
        self.assertEqual(self.smoke.proc.returncode, 0, self.smoke.proc.stderr[-2000:])

    def test_every_request_reaches_the_given_port(self):
        self.assertEqual(self.smoke.stats["requests"], 100)
        self.assertTrue(self.smoke.result["endpoint"].endswith(
            f":{self.smoke.port}/v1/chat/completions"))
        boot = json.loads((self.smoke.out / "boot.json").read_text())
        self.assertEqual(boot["port"], self.smoke.port)

    def test_lanes_run_at_their_stated_concurrency(self):
        arrivals = self.smoke.stats["arrivals"]
        self.assertEqual(max(arrivals[:50]), 1)   # concurrency_1 is sequential
        self.assertEqual(max(arrivals[50:]), 4)   # concurrency_4 keeps four in flight
        self.assertEqual(self.smoke.result["concurrency_1"]["max_in_flight"], 1)
        self.assertEqual(self.smoke.result["concurrency_4"]["max_in_flight"], 4)

    def test_wrong_answers_are_recorded_as_incorrect(self):
        c = self.smoke.result["correctness"]
        self.assertEqual((c["checked"], c["correct"], c["incorrect_count"]), (100, 0, 100))
        self.assertEqual(self.smoke.result["unique_prompts"], 50)

    def test_results_go_to_the_reported_path_not_cwd(self):
        self.assertFalse((self.smoke.cwd / "results.json").exists())
        lines = self.smoke.proc.stdout.splitlines()
        self.assertIn(f"=== Smoke test complete. Results: {self.smoke.out}/results.json ===", lines)
        self.assertEqual(self.smoke.result.get("total_calls"), 100)  # not the seeded stale file
        self.assertIn(f"Verdict: {self.smoke.result['verdict']}", lines)
        self.assertNotIn("Verdict: STALE", lines)

    def test_wrong_answers_do_not_pass(self):
        self.assertEqual(self.smoke.result["verdict"], "FAIL")
        self.assertEqual(self.smoke.result["state"], "FAILED")


class RightAnswers(unittest.TestCase):
    def test_right_answers_are_recorded_as_correct(self):
        run = Run("right")
        try:
            c = run.result["correctness"]
            self.assertEqual((c["checked"], c["correct"], c["incorrect_count"]), (100, 100, 0))
            self.assertEqual(run.result["verdict"], "PASS")
            self.assertEqual(run.result["state"], "MEASURED")
        finally:
            run.tmp.cleanup()


class OutputOverride(unittest.TestCase):
    def test_szl_eval_out_holds_all_outputs(self):
        run = Run("wrong", out_override=True)
        try:
            self.assertTrue((run.out / "results.json").exists())
            self.assertTrue((run.out / "boot.json").exists())
            self.assertFalse((run.home / "szl-evals").exists())
        finally:
            run.tmp.cleanup()


class PortValidation(unittest.TestCase):
    def test_non_integer_port_is_refused_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script("8001'); print('injected", tmp, tmp)
            self.assertEqual(proc.returncode, 2, proc.stderr[-500:])
            self.assertFalse((pathlib.Path(tmp) / "szl-evals").exists())


if __name__ == "__main__":
    unittest.main()
