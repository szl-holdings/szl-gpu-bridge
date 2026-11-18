"""Source-scoped Salad fleet evidence contracts; no network or provider effects."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from tools.salad import fleet_watch_readonly as watch

NOW = datetime(2026, 11, 17, 23, 30, tzinfo=timezone.utc)


def group(i):
    status = "running" if i < 74 else "deploying" if i < 136 else "stopped"
    return {
        "name": f"szl-fixture-{i:03d}",
        "replicas": 1,
        "priority": "lowest",
        "container": {
            "image": "container.example/worker:mutable",
            "environment_variables": {"HF_TOKEN": "DO_NOT_LEAK_SENTINEL"},
            "logging": {"http": {"password": "DO_NOT_LEAK_SENTINEL"}},
        },
        "current_state": {
            "status": status,
            "instance_status_counts": {
                "running_count": 1 if status == "running" else 0,
            },
        },
    }


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Client:
    def __init__(self, body):
        self.body = body
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "has_key_header": bool(request.get_header("Salad-api-key")),
                "timeout": timeout,
            }
        )
        return Response(json.dumps(self.body).encode())


class FleetContractTests(unittest.TestCase):
    def test_139_groups_prove_100_item_coverage_assumption_is_unsafe(self):
        result = watch.summarize_fleet({"items": [group(i) for i in range(139)]})
        self.assertEqual(result["groups_observed"], 139)
        self.assertEqual(
            result["group_state_counts"], {"deploying": 62, "running": 74, "stopped": 3}
        )
        self.assertEqual(result["configured_replicas_known_subtotal"], 139)
        self.assertEqual(result["running_instances_provider_reported_subtotal"], 74)
        self.assertEqual(result["groups_without_immutable_image_reference"], 139)
        self.assertEqual(
            result["provider_collection_scope"],
            "ONE_BOUNDED_RESPONSE_NOT_PROVEN_EXHAUSTIVE",
        )
        self.assertFalse(result["training_liveness_verified"])
        self.assertFalse(result["ready_instances_independently_verified"])
        self.assertFalse(result["billable_runtime_state_verified"])
        self.assertNotIn("DO_NOT_LEAK_SENTINEL", json.dumps(result))

    def test_metadata_digest_is_canonical_across_provider_order(self):
        left = watch.summarize_fleet({"items": [group(2), group(1), group(0)]})
        right = watch.summarize_fleet({"items": [group(0), group(2), group(1)]})
        self.assertEqual(left["group_metadata_sha256"], right["group_metadata_sha256"])

    def test_empty_collection_triggers_investigation_not_zero_capacity_claim(self):
        result, drift = watch.collect(
            "synthetic",
            reader=lambda url, *, token=None: {"items": []} if token else [],
            now=NOW,
        )
        self.assertEqual(result["fleet"]["groups_observed"], 0)
        self.assertEqual(result["release_admission"], "HOLD")
        self.assertIn("SALAD_EMPTY_COLLECTION_NOT_PROVEN_COMPLETE", drift)

    def test_duplicate_or_unknown_input_is_unavailable_not_zero(self):
        for payload in (
            {},
            {"items": None},
            {"items": [42]},
            {"items": [group(1), group(1)]},
            {"items": [dict(group(1), name="../unsafe")]},
            {"items": [dict(group(1), current_state=None)]},
            {"items": [group(i) for i in range(301)]},
        ):
            with self.subTest(case=str(type(payload))):
                with self.assertRaises(watch.Unavailable):
                    watch.summarize_fleet(payload)

    def test_ambiguous_running_state_cannot_qualify_training(self):
        g = group(1)
        g["current_state"]["instance_status_counts"]["running_count"] = 0
        result = watch.summarize_fleet({"items": [g]})
        self.assertEqual(result["anomalies_total"], 1)
        self.assertFalse(result["trainer_steps_observed"])
        self.assertFalse(result["training_liveness_verified"])
        self.assertFalse(result["source_deployment_parity_verified"])

    def test_untrusted_collection_metadata_never_crashes_or_leaks_values(self):
        g = group(0)
        g["current_state"]["status"] = ["provider-unexpected-shape"]
        g["priority"] = {"secret": "DO_NOT_LEAK_SENTINEL"}
        result = watch.summarize_fleet({"items": [g]})
        self.assertEqual(result["group_state_counts"], {"unknown": 1})
        self.assertEqual(result["priority_counts"], {"unknown": 1})
        self.assertEqual(result["anomalies_total"], 1)
        self.assertNotIn("DO_NOT_LEAK_SENTINEL", json.dumps(result))

    def test_workflow_uses_pinned_source_and_safe_data_only_issue_writer(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / ".github/workflows/fleet-watch.yml").read_text()
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("python3 -I -B tools/salad/fleet_watch_readonly.py", text)
        self.assertIn("SALAD_API_KEY: ${{ secrets.SALAD_API_KEY }}", text)
        self.assertIn("JSON.parse(process.env.FLEET_DRIFT_JSON", text)
        self.assertIn("JSON.parse(process.env.FLEET_REPORT_JSON", text)
        self.assertNotIn("const drift = ${{", text)
        self.assertNotIn("const report = ${{", text)

    def test_unknown_replica_and_instance_counts_never_become_zero(self):
        g = group(0)
        g["replicas"] = None
        g["current_state"].pop("instance_status_counts")
        result = watch.summarize_fleet({"items": [g]})
        self.assertEqual(result["configured_replicas_unknown_groups"], 1)
        self.assertEqual(result["running_instance_count_unknown_groups"], 1)
        self.assertEqual(result["anomalies_total"], 0)

    def test_exact_scoped_api_uses_one_get_and_cannot_forward_key(self):
        client = Client({"items": [group(0)]})
        obj = watch.read_json(watch.PROVIDER_URL, token="NOT_A_REAL_KEY", opener=client)
        self.assertEqual(obj["items"][0]["name"], "szl-fixture-000")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["method"], "GET")
        self.assertTrue(client.requests[0]["has_key_header"])
        self.assertLessEqual(client.requests[0]["timeout"], 12)
        with self.assertRaisesRegex(watch.Unavailable, "CREDENTIAL_SCOPE_REFUSED"):
            watch.read_json(watch.CHAIN_URL, token="NOT_A_REAL_KEY", opener=client)
        self.assertEqual(len(client.requests), 1)
        self.assertIsNone(
            watch.NoRedirect().redirect_request(
                None, None, 301, "", {}, "https://other.example/"
            )
        )

    def test_public_chain_commit_recency_is_global_not_per_lane(self):
        chain = [
            {"date": "2026-11-17T23:29:00Z", "title": "boot"},
            {"date": "2026-11-17T23:28:15.123Z", "title": "fail-closed"},
            {"date": "2026-11-16T20:00:00+00:00", "title": "stale"},
            {"date": "not-a-date"},
        ]
        result = watch.summarize_chain(chain, now=NOW)
        self.assertEqual(result["recent_commit_count_in_sample"], 2)
        self.assertFalse(result["per_lane_heartbeat_verified"])
        self.assertFalse(result["training_completion_verified"])
        self.assertEqual(result["scope"], "GLOBAL_CHAIN_RECENCY_NOT_LANE_LIVENESS")

    def test_401_stops_probing_and_retains_explicit_hold(self):
        calls = []

        def reader(url, *, token=None):
            calls.append((url, token is not None))
            if token is not None:
                raise urllib.error.HTTPError(url, 401, "secret detail", None, None)
            return []

        record, drift = watch.collect("DO_NOT_LEAK_SENTINEL", reader=reader, now=NOW)
        self.assertEqual(calls, [(watch.PROVIDER_URL, True), (watch.CHAIN_URL, False)])
        self.assertEqual(record["fleet"]["reason"], "SALAD_HTTP_401")
        self.assertIn("SALAD_HTTP_401", drift)
        self.assertEqual(record["release_admission"], "HOLD")
        self.assertNotIn("DO_NOT_LEAK_SENTINEL", json.dumps(record))

    def test_missing_key_cannot_read_protected_provider(self):
        def reader(url, *, token=None):
            if token is not None:
                self.fail("provider lookup with missing key")
            return []

        record, drift = watch.collect(None, reader=reader, now=NOW)
        self.assertEqual(record["fleet"]["provider_observation"], "UNAVAILABLE")
        self.assertEqual(drift, ["SALAD_KEY_NOT_CONFIGURED"])

    def test_output_is_one_line_json_without_token_or_runner_injection(self):
        report, drift = watch.collect(
            "DO_NOT_LEAK_SENTINEL",
            reader=lambda url, *, token=None: {"items": [group(0)]} if token else [],
            now=NOW,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output"
            with patch("builtins.print") as output:
                watch.emit(report, drift, output_path=str(path))
            output_text = path.read_text(encoding="utf-8")
            self.assertEqual(len(output_text.splitlines()), 3)
            fields = dict(line.split("=", 1) for line in output_text.splitlines())
            self.assertEqual(fields["has_drift"], "false")
            self.assertEqual(json.loads(fields["drift"]), [])
            self.assertEqual(json.loads(fields["report"])["release_admission"], "HOLD")
            self.assertNotIn("DO_NOT_LEAK_SENTINEL", output_text)
            self.assertNotIn("DO_NOT_LEAK_SENTINEL", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
