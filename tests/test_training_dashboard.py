"""Behavioral contracts for the observer -> local HTTP -> dashboard boundary."""

import copy
import datetime as dt
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from tools.dashboard import training_server as view

NOW = dt.datetime(2026, 11, 16, 23, 30, tzinfo=dt.timezone.utc)


def observation():
    return {
        "schema": view.INPUT_SCHEMA,
        "cycle": 3,
        "started_at_utc": (NOW - dt.timedelta(seconds=20)).isoformat(),
        "observed_at_utc": (NOW - dt.timedelta(seconds=10)).isoformat(),
        "stale_after_seconds": 660,
        "training_complete_verified": False,
        "promotion_authorized": False,
        "candidates": [
            {
                "repo": "SZLHOLDINGS/example",
                "target_steps": 200,
                "state": "TERMINAL_METADATA_PRESENT",
                "observed_at_utc": (NOW - dt.timedelta(seconds=15)).isoformat(),
                "training_complete_verified": False,
                "promotion_authorized": False,
                "trainer_process_liveness_verified": False,
                "snapshot": {
                    "id": "SZLHOLDINGS/example",
                    "status": "OBSERVED",
                    "revision": "a" * 40,
                    "latest_checkpoint": 200,
                    "private": True,
                    "final_adapter": True,
                    "final_config": True,
                    "final_tokenizer": True,
                    "training_complete": "UNVERIFIED",
                },
                "trainer_state": {
                    "global_step": 200,
                    "max_steps": 200,
                    "integrity": "PINNED_OBJECT_BYTES_VERIFIED",
                    "trainer_state_sha256": "c" * 64,
                },
                "provider": {
                    "state": "OBSERVED",
                    "instance_count": 1,
                    "instance_states": ["running"],
                    "ready_count": 1,
                    "trainer_process_liveness_verified": False,
                },
            }
        ],
    }


def encoded(value):
    return json.dumps(value).encode()


class ProjectionTests(unittest.TestCase):
    def project(self, data=None, **kwargs):
        return view.project_observation(
            encoded(observation() if data is None else data), now=NOW, **kwargs
        )

    def test_terminal_metadata_and_running_provider_never_authorize(self):
        result = self.project()
        self.assertEqual(result["status"], "CURRENT")
        row = result["candidates"][0]
        self.assertEqual(row["state"], "TERMINAL_METADATA_PRESENT")
        self.assertEqual(row["provider"]["instance_states"], ["running"])
        for target in [result, row, row["provider"]]:
            for name in view.FLAGS:
                if name in target:
                    self.assertIs(target[name], False)
        self.assertEqual(result["scope"], "SELECTED_CANDIDATES")

    def test_identity_binds_exact_input_bytes_without_echoing_extra_fields(self):
        data = observation()
        data["secret"] = "never-echo-this"
        data["candidates"][0]["snapshot"]["token"] = "never-echo-this"
        data["candidates"][0]["provider"]["reason"] = "never-echo-this"
        raw = encoded(data)
        result = view.project_observation(raw, now=NOW)
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertNotIn("never-echo-this", json.dumps(result))
        self.assertNotIn("private", result["candidates"][0])

    def test_sha256_revision_supported_by_producer_is_preserved(self):
        data = observation()
        data["candidates"][0]["snapshot"]["revision"] = "b" * 64
        result = self.project(data)
        self.assertEqual(result["status"], "CURRENT")
        self.assertEqual(result["candidates"][0]["revision"], "b" * 64)

    def test_contradictory_source_metadata_is_invalid(self):
        mutations = [
            lambda row: row["snapshot"].update(id="SZLHOLDINGS/other"),
            lambda row: row["snapshot"].update(status="UNAVAILABLE"),
            lambda row: row["snapshot"].update(final_adapter=False),
            lambda row: row["snapshot"].update(final_config=False),
            lambda row: row["snapshot"].update(final_tokenizer=False),
            lambda row: row.update(state="NO_CHECKPOINT_OBSERVED"),
            lambda row: row["trainer_state"].pop("trainer_state_sha256"),
            lambda row: row["trainer_state"].update(trainer_state_sha256="main"),
        ]
        for mutate in mutations:
            data = observation()
            mutate(data["candidates"][0])
            self.assertEqual(self.project(data)["status"], "INVALID")

    def test_old_candidate_is_stale_even_when_report_was_just_written(self):
        data = observation()
        data["started_at_utc"] = (NOW - dt.timedelta(seconds=1000)).isoformat()
        data["candidates"][0]["observed_at_utc"] = (
            NOW - dt.timedelta(seconds=700)
        ).isoformat()
        result = self.project(data)
        self.assertEqual(result["status"], "STALE")
        row = result["candidates"][0]
        self.assertEqual(row["state"], "STALE")
        self.assertEqual(row["observed_state"], "TERMINAL_METADATA_PRESENT")
        self.assertEqual(row["provider"]["state"], "STALE")
        self.assertEqual(row["provider"]["observed_state"], "OBSERVED")

    def test_expired_source_does_not_become_current_on_read(self):
        raw = encoded(observation())
        result = view.project_observation(raw, now=NOW + dt.timedelta(seconds=700))
        self.assertEqual(result["status"], "STALE")
        self.assertEqual(result["reason"], "OBSERVATION_EXPIRED")
        self.assertEqual(result["candidates"][0]["freshness"], "STALE")

    def test_producer_cannot_extend_operator_freshness_cap(self):
        data = observation()
        data["stale_after_seconds"] = 7260
        raw = encoded(data)
        result = view.project_observation(raw, now=NOW + dt.timedelta(seconds=950))
        self.assertEqual(result["source"]["stale_after_seconds"], 900)
        self.assertEqual(result["status"], "STALE")

    def test_unknown_evidence_and_zero_instances_are_distinct(self):
        data = observation()
        row = data["candidates"][0]
        row.update(state="UNAVAILABLE")
        row.pop("snapshot")
        row.pop("trainer_state")
        row["provider"].update(instance_count=0, instance_states=[], ready_count=0)
        result = self.project(data)
        self.assertEqual(result["status"], "CURRENT")
        projected = result["candidates"][0]
        self.assertEqual(projected["state"], "UNAVAILABLE")
        self.assertIsNone(projected["global_step"])
        self.assertEqual(projected["provider"]["instance_count"], 0)
        row["provider"] = {
            "state": "UNAVAILABLE",
            "trainer_process_liveness_verified": False,
        }
        self.assertIsNone(
            self.project(data)["candidates"][0]["provider"]["instance_count"]
        )

    def test_unrecognized_provider_state_is_not_reflected(self):
        data = observation()
        data["candidates"][0]["provider"]["instance_states"] = [
            "<script>secret</script>"
        ]
        result = self.project(data)
        self.assertEqual(
            result["candidates"][0]["provider"]["instance_states"], ["unknown"]
        )
        self.assertNotIn("script", json.dumps(result))

    def test_invalid_evidence_cannot_claim_current_or_reuse_candidates(self):
        mutations = [
            lambda x: x.update(schema="another-schema"),
            lambda x: x.update(training_complete_verified=True),
            lambda x: x.update(promotion_authorized=0),
            lambda x: x.update(stale_after_seconds=True),
            lambda x: x.update(stale_after_seconds=99999999),
            lambda x: x.update(
                observed_at_utc=(NOW + dt.timedelta(seconds=1)).isoformat()
            ),
            lambda x: x.update(observed_at_utc="2026-11-16T23:30:00"),
            lambda x: x.update(candidates=[]),
            lambda x: x["candidates"].append(copy.deepcopy(x["candidates"][0])),
            lambda x: x["candidates"][0].update(promotion_authorized=True),
            lambda x: x["candidates"][0].update(target_steps=True),
            lambda x: x["candidates"][0].update(state="TRAINING_COMPLETE"),
            lambda x: x["candidates"][0]["snapshot"].update(revision="main"),
            lambda x: x["candidates"][0]["trainer_state"].update(global_step=199),
            lambda x: x["candidates"][0]["trainer_state"].update(max_steps=201),
            lambda x: x["candidates"][0]["provider"].update(instance_count=True),
            lambda x: x["candidates"][0]["provider"].update(ready_count=2),
            lambda x: x["candidates"][0]["provider"].update(instance_states=[]),
            lambda x: x["candidates"][0]["provider"].update(
                trainer_process_liveness_verified=True
            ),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                data = observation()
                mutate(data)
                result = self.project(data)
                self.assertEqual(result["status"], "INVALID")
                self.assertEqual(result["candidates"], [])
                self.assertIsNone(result["source"])

    def test_strict_json_and_read_budget(self):
        for raw in [
            b"{}",
            b"[]",
            b"not-json",
            b'{"x":1,"x":2}',
            b'{"x":NaN}',
            b'{"x":1e999}',
            b" " * (view.MAX_BYTES + 1),
        ]:
            with self.subTest(length=len(raw)):
                self.assertEqual(
                    view.project_observation(raw, now=NOW)["status"], "INVALID"
                )

    def test_missing_file_is_unavailable_and_path_is_not_disclosed(self):
        result = view.read_projection("/missing/private-secret.json", now=NOW)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["reason"], "SOURCE_NOT_FOUND")
        self.assertNotIn("private-secret", json.dumps(result))

    def test_regular_file_restriction_and_bounded_read(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(view.read_projection(temp, now=NOW)["status"], "INVALID")
            path = Path(temp) / "status.json"
            path.write_bytes(b" " * (view.MAX_BYTES + 1))
            self.assertEqual(view.read_projection(path, now=NOW)["status"], "INVALID")


class HttpBoundaryTests(unittest.TestCase):
    def setUp(self):
        clock = mock.patch.object(view, "utc_now", return_value=NOW)
        clock.start()
        self.addCleanup(clock.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "status.json"
        self.path.write_bytes(encoded(observation()))
        self.server = view.make_server(self.path, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, *, method="GET", headers=None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=3
        )
        try:
            conn.request(method, path, headers=headers or {})
            response = conn.getresponse()
            body = response.read()
            return response.status, dict(response.getheaders()), body
        finally:
            conn.close()

    def test_backend_reload_drives_frontend_without_provider_calls(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        code, headers, body = self.request("/api/training/status")
        self.assertEqual(code, 200)
        self.assertEqual(
            json.loads(body)["source"]["sha256"],
            hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.path.write_bytes(b'{"invalid":true}')
        code, _, body = self.request("/api/training/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["status"], "INVALID")
        self.assertEqual(json.loads(body)["candidates"], [])
        self.path.unlink()
        self.assertEqual(
            json.loads(self.request("/api/training/status")[2])["status"], "UNAVAILABLE"
        )

    def test_host_and_cross_origin_requests_are_rejected(self):
        for headers in [
            {"Host": "attacker.example"},
            {"Origin": "https://attacker.example"},
            {"Origin": "null"},
        ]:
            with self.subTest(headers=headers):
                self.assertEqual(
                    self.request("/api/training/status", headers=headers)[0], 403
                )

    def test_no_paths_or_writes_are_exposed(self):
        for path in [
            "/../status.json",
            "/status.json",
            "/api/training/status?path=other",
            "/.git/config",
            "/api/training/start",
        ]:
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)
        for method in ["POST", "PUT", "PATCH", "DELETE"]:
            self.assertEqual(
                self.request("/api/training/status", method=method)[0], 405
            )

    def test_health_is_only_component_health_and_html_is_same_origin(self):
        code, _, body = self.request("/healthz")
        self.assertEqual(code, 200)
        health = json.loads(body)
        self.assertTrue(health["ok"])
        self.assertFalse(health["training_complete_verified"])
        code, headers, body = self.request("/training")
        self.assertEqual(code, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"Training", body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])


if __name__ == "__main__":
    unittest.main()
