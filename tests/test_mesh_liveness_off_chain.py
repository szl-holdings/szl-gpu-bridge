from __future__ import annotations

import importlib.util
import io
import itertools
import json
import os
import pathlib
import tempfile
import time
import types
import unittest
import urllib.error
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON = ROOT / "tools" / "mesh" / "mesh_daemon.py"


def load(env=None):
    """A fresh copy of the daemon module, read under a controlled environment."""
    clean = {k: v for k, v in os.environ.items() if k not in ("MESH_LIVENESS_DS", "MESH_HOME")}
    clean.update(env or {})
    with mock.patch.dict(os.environ, clean, clear=True):
        spec = importlib.util.spec_from_file_location("mesh_daemon_under_test", DAEMON)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeApi:
    def __init__(self, fail_repo=None):
        self.writes, self.fail_repo = [], fail_repo

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        if repo_id == self.fail_repo:
            raise OSError("repo unreachable")
        self.writes.append((repo_id, path_in_repo))


class StopLoop(Exception):
    pass


def stop_after(cycles):
    """A stand-in for the daemon's `time` whose sleep ends the loop after N cycles."""
    ticks = itertools.count(1)

    def sleep(_):
        if next(ticks) >= cycles:
            raise StopLoop

    return types.SimpleNamespace(time=time.time, sleep=sleep, strftime=time.strftime,
                                 gmtime=time.gmtime, strptime=time.strptime)


class Base(unittest.TestCase):
    def setUp(self):
        self.mesh = load()
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.mesh.MESH_DIR = pathlib.Path(home.name)
        self.mesh.gpu_state = lambda: {"state": "UNAVAILABLE"}
        lock = open(os.devnull, "w")
        self.addCleanup(lock.close)
        self.mesh.claim_node = lambda path: lock

    def run_loop(self, fn, cycles):
        self.mesh.time = stop_after(cycles)
        with self.assertRaises(StopLoop), mock.patch("sys.stdout", io.StringIO()):
            fn()


class WorkerTests(Base):
    def heartbeat(self):
        return (self.mesh.LIVENESS_DS, f"mesh/workers/{self.mesh.NODE_ID}/heartbeat.json")

    def register(self):
        return (self.mesh.DS, f"mesh/workers/{self.mesh.NODE_ID}/register.json")

    def test_heartbeats_never_commit_to_the_receipts_chain(self):
        api, polls = FakeApi(), []
        self.mesh.poll = lambda *a, **k: polls.append(1)
        self.run_loop(lambda: self.mesh.worker_loop(api, "brain:0", b""), cycles=3)
        self.assertEqual(api.writes, [self.register()] + [self.heartbeat()] * 3)
        self.assertEqual(len(polls), 3)

    def test_failed_heartbeat_does_not_stop_polling(self):
        api, polls = FakeApi(fail_repo=self.mesh.LIVENESS_DS), []
        self.mesh.poll = lambda *a, **k: polls.append(1)
        self.run_loop(lambda: self.mesh.worker_loop(api, "brain:0", b""), cycles=3)
        self.assertEqual(api.writes, [self.register()])
        self.assertEqual(len(polls), 3)

    def test_job_results_stay_on_the_chain(self):
        api = FakeApi()
        job = {"id": "j1", "kind": "gpu-state", "node": self.mesh.NODE_ID}
        self.mesh.execute_job = lambda job, key, done: {"state": "DONE"}
        with mock.patch("sys.stdout", io.StringIO()):
            self.mesh.poll(api, b"", None, None, {"done": 0, "rejected": 0}, read=lambda path: job)
        self.assertEqual(api.writes, [(self.mesh.DS, f"mesh/results/{self.mesh.NODE_ID}/j1.json")])


class BrainTests(Base):
    def serve(self, heartbeats):
        """heartbeats: {repo: {node: age_seconds}}. Anything else answers 404."""
        hf = self.mesh.HF

        def urlopen(req, timeout=None):
            url = req.full_url
            for repo, nodes in heartbeats.items():
                if url == f"{hf}/api/datasets/{repo}/tree/main/mesh/workers?recursive=true":
                    body = [{"type": "file", "path": f"mesh/workers/{n}/heartbeat.json"} for n in nodes]
                    return io.BytesIO(json.dumps(body).encode())
                for node, age in nodes.items():
                    if url == f"{hf}/datasets/{repo}/resolve/main/mesh/workers/{node}/heartbeat.json":
                        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age))
                        return io.BytesIO(json.dumps({"ts": ts, "gpu": {"state": "MEASURED"}}).encode())
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        patcher = mock.patch.object(self.mesh.urllib.request, "urlopen", urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def roster(self, heartbeats, api=None, cycles=1):
        api = api or FakeApi()
        states = []
        real = self.mesh.receipt
        self.mesh.receipt = lambda kind, data: states.append(data) or real(kind, data)
        self.serve(heartbeats)
        self.run_loop(lambda: self.mesh.brain_loop(api), cycles)
        return api, states

    def test_roster_never_commits_to_the_receipts_chain(self):
        api, _ = self.roster({self.mesh.LIVENESS_DS: {"new": 5}}, cycles=2)
        self.assertEqual(api.writes, [(self.mesh.LIVENESS_DS, "mesh/brain/state.json")] * 2)

    def test_mixed_rollout_sees_workers_on_both_sides(self):
        _, states = self.roster({self.mesh.LIVENESS_DS: {"new": 5}, self.mesh.DS: {"old": 5}})
        alive = {"alive": True, "gpu": {"state": "MEASURED"}}
        self.assertEqual(states[0]["workers"], {"new": alive, "old": alive})

    def test_upgraded_worker_is_not_judged_by_its_frozen_chain_heartbeat(self):
        _, states = self.roster({self.mesh.LIVENESS_DS: {"n": 5}, self.mesh.DS: {"n": 86400}})
        self.assertTrue(states[0]["workers"]["n"]["alive"])

    def test_stale_worker_is_reported_dead_not_dropped(self):
        _, states = self.roster({self.mesh.DS: {"gone": 86400}})
        self.assertEqual(states[0]["workers"]["gone"]["alive"], False)
        self.assertEqual((states[0]["alive_count"], states[0]["total_known"]), (0, 1))

    def test_empty_liveness_repo_still_reports_chain_workers(self):
        _, states = self.roster({self.mesh.DS: {"old": 5}})
        self.assertEqual(states[0]["alive_count"], 1)

    def test_failed_roster_write_does_not_stop_the_brain(self):
        api, states = self.roster({self.mesh.LIVENESS_DS: {"n": 5}},
                                  api=FakeApi(fail_repo=self.mesh.LIVENESS_DS), cycles=3)
        self.assertEqual(api.writes, [])
        self.assertEqual(len(states), 3)


class ConfigTests(unittest.TestCase):
    def test_default_liveness_repo_is_not_the_chain(self):
        mesh = load()
        self.assertEqual(mesh.LIVENESS_DS, "SZLHOLDINGS/mesh-liveness")
        self.assertNotEqual(mesh.LIVENESS_DS, mesh.DS)
        self.assertIsNone(mesh.liveness_problem())

    def test_empty_override_falls_back_to_the_default(self):
        self.assertEqual(load({"MESH_LIVENESS_DS": ""}).LIVENESS_DS, "SZLHOLDINGS/mesh-liveness")

    def test_pointing_liveness_at_the_chain_is_refused(self):
        mesh = load({"MESH_LIVENESS_DS": "SZLHOLDINGS/frontier-54-runs"})
        self.assertIn("receipts chain", mesh.liveness_problem())
        api = FakeApi()
        with self.assertRaises(RuntimeError):
            mesh.liveness_write(api, "mesh/brain/state.json", {})
        self.assertEqual(api.writes, [])

    def test_malformed_liveness_repo_is_refused(self):
        for bad in ("nodataset", "a/b/c", "has space/x", "/x"):
            self.assertIn("owner/name", load({"MESH_LIVENESS_DS": bad}).liveness_problem())

    def test_main_fails_closed_before_any_write(self):
        mesh = load({"MESH_LIVENESS_DS": "SZLHOLDINGS/frontier-54-runs"})
        with mock.patch.dict(os.environ, {"HF_TOKEN": "x"}), \
                mock.patch("sys.argv", ["mesh_daemon.py"]), \
                mock.patch("sys.stderr", io.StringIO()) as err:
            self.assertEqual(mesh.main(), 2)
        self.assertIn("FAIL-CLOSED", err.getvalue())


if __name__ == "__main__":
    unittest.main()
