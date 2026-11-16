"""Deterministic roster freshness; synthetic reads, no provider or job calls."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
import urllib.error
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "tools/mesh/mesh_daemon.py"
T0 = 1794860000


class EndCycles(Exception):
    pass


def stamp(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


class RosterFreshnessTests(unittest.TestCase):
    def run_roster(self, frames, step=60, initial_offset=0.0):
        spec = importlib.util.spec_from_file_location("mesh_freshness_subject", SOURCE)
        mesh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mesh)
        state = {"cycle": 0, "now": float(T0) + initial_offset}
        outputs = []

        def sleep(_):
            state["cycle"] += 1
            state["now"] += step
            if state["cycle"] >= len(frames):
                raise EndCycles

        def tree(req, timeout=None):
            if f"/api/datasets/{mesh.LIVENESS_DS}/" not in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "fixture", None, None)
            frame = frames[state["cycle"]]
            if isinstance(frame, Exception):
                raise frame
            return io.BytesIO(
                json.dumps(
                    [{"path": f"mesh/workers/{node}/heartbeat.json"} for node in frame]
                ).encode()
            )

        def heartbeat(path, **kwargs):
            node = path.split("/")[2]
            value = frames[state["cycle"]][node]
            if isinstance(value, Exception):
                raise value
            return {"ts": value, "gpu": {"state": "MEASURED"}}

        clock = types.SimpleNamespace(
            time=lambda: state["now"],
            sleep=sleep,
            strftime=time.strftime,
            gmtime=time.gmtime,
            strptime=time.strptime,
        )
        with tempfile.TemporaryDirectory() as folder:
            mesh.MESH_DIR = Path(folder)
            with (
                mock.patch.object(mesh, "time", clock),
                mock.patch.object(mesh.urllib.request, "urlopen", tree),
                mock.patch.object(mesh, "chain_read", heartbeat),
                mock.patch.object(
                    mesh, "liveness_write", lambda api, path, data: outputs.append(data)
                ),
                mock.patch("sys.stdout", io.StringIO()),
                mock.patch.object(
                    mesh, "execute_job", side_effect=AssertionError("no job dispatch")
                ),
            ):
                with self.assertRaises(EndCycles):
                    mesh.brain_loop(None)
        return outputs

    def test_removed_worker_cannot_keep_old_alive_verdict(self):
        states = self.run_roster([{"worker": stamp(T0)}, {}])
        self.assertTrue(states[0]["workers"]["worker"]["alive"])
        self.assertFalse(states[1]["workers"]["worker"]["alive"])
        self.assertEqual(states[1]["total_known"], 1)

    def test_failed_listing_retains_inventory_but_not_health(self):
        states = self.run_roster([{"worker": stamp(T0)}, OSError("fixture outage")])
        self.assertEqual(states[-1]["total_known"], 1)
        self.assertEqual(states[-1]["alive_count"], 0)

    def test_failed_heartbeat_read_is_not_cached_health(self):
        states = self.run_roster(
            [{"worker": stamp(T0)}, {"worker": OSError("fixture read failure")}]
        )
        self.assertFalse(states[-1]["workers"]["worker"]["alive"])

    def test_retained_last_seen_expires_across_multiple_empty_cycles(self):
        states = self.run_roster([{"worker": stamp(T0)}, {}, {}, {}])
        self.assertEqual(states[-1]["alive_count"], 0)
        self.assertEqual(states[-1]["total_known"], 1)

    def test_future_heartbeat_cannot_prove_liveness(self):
        states = self.run_roster([{"worker": stamp(T0 + 60)}])
        self.assertEqual(states[-1]["alive_count"], 0)

    def test_subsecond_future_timestamp_is_not_rounded_to_fresh(self):
        states = self.run_roster(
            [{"worker": stamp(T0 + 1)}], step=0.5, initial_offset=0.5
        )
        self.assertEqual(states[-1]["alive_count"], 0)

    def test_wrong_timezone_or_trailing_data_cannot_prove_liveness(self):
        for suffix in ("+05:00", "garbage", "Zjunk", ""):
            with self.subTest(suffix=suffix):
                states = self.run_roster([{"worker": stamp(T0)[:-1] + suffix}])
                self.assertEqual(states[-1]["alive_count"], 0)

    def test_exact_expiry_boundary_is_not_fresh(self):
        states = self.run_roster([{"worker": stamp(T0 - 180)}])
        self.assertEqual(states[-1]["alive_count"], 0)

    def test_current_fresh_heartbeat_still_proves_observed_liveness(self):
        states = self.run_roster([{"worker": stamp(T0 - 179)}])
        self.assertEqual(states[-1]["alive_count"], 1)

    def test_recovered_worker_can_become_fresh_again(self):
        states = self.run_roster(
            [{"worker": stamp(T0)}, {}, {"worker": stamp(T0 + 120)}]
        )
        self.assertFalse(states[1]["workers"]["worker"]["alive"])
        self.assertTrue(states[2]["workers"]["worker"]["alive"])
        self.assertEqual(states[2]["total_known"], 1)

    def test_failed_peer_does_not_hide_other_fresh_worker(self):
        states = self.run_roster(
            [
                {"a": stamp(T0), "b": stamp(T0)},
                {"a": stamp(T0 + 60), "b": OSError("fixture")},
            ]
        )
        self.assertEqual(states[-1]["alive_count"], 1)
        self.assertTrue(states[-1]["workers"]["a"]["alive"])
        self.assertFalse(states[-1]["workers"]["b"]["alive"])


if __name__ == "__main__":
    unittest.main()
