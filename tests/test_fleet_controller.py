"""Runner wiring for the reviewed fleet controller, with fake provider and Hub (no network)."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from test_artifact_qualification import FakeApi, fixture
from tools.salad import controller_gates as gates
from tools.salad import fleet_controller as fc

PREFIX = "collaboration/vgl-criteria-v2/salad-v5-heldout-v2-"


class FakeSalad:
    def __init__(self, status="stopped", replicas=1, running=0):
        self.calls = []
        self.status, self.replicas, self.running = status, replicas, running

    def call(self, method, path, body=None, content_type="application/json"):
        self.calls.append((method, path, body))
        if method == "PATCH":
            self.replicas = body["replicas"]
            self.status, self.running = "stopped", 0
        if method == "POST" and path.endswith("/stop"):
            self.status, self.running = "stopped", 0
        return 0, 202, b""

    def get_json(self, path):
        self.calls.append(("GET", path, None))
        name = path.strip("/").split("/")[0]
        if path.endswith("/instances"):
            return {
                "instances": [
                    {"instance_id": str(i), "state": "running"}
                    for i in range(self.running)
                ]
            }, 200
        counts = {k: 0 for k in gates.COUNTS}
        counts["running_count"] = self.running
        return {
            "name": name,
            "replicas": self.replicas,
            "pending_change": False,
            "current_state": {"status": self.status, "instance_status_counts": counts},
        }, 200

    def mutations(self):
        return [c for c in self.calls if c[0] != "GET"]


class FakeHub:
    def __init__(self, model_files=None, dataset_files=None):
        self.model = FakeApi(model_files or fixture(step=100))
        self.api = self.model
        self.dataset_files = dataset_files or {}
        self.published = []

    def fetch(
        self, *, repo_id, filename, revision, local_dir, etag_timeout, repo_type="model"
    ):
        if repo_type == "dataset":
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.served.get(filename, self.dataset_files[filename]))
            return str(path)
        return self.model.fetch(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
            etag_timeout=etag_timeout,
        )

    served = {}

    def dataset_snapshot(self, repo_id):
        entries = {
            name: NS(
                rfilename=name,
                size=len(raw),
                lfs=None,
                blob_id=hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(),
            )
            for name, raw in self.dataset_files.items()
        }
        return "d" * 40, entries

    def publish(self, repo_id, path, payload):
        self.published.append(path)
        return gates.publication_result(
            payload=payload, commit_oid="e" * 40, readback=payload, path_listed=True
        )


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        lease = json.dumps(
            {
                "generation": 1,
                "groups": ["szl-train-example", "szl-bench-example"],
                "expires_utc": "2099-01-01T00:00:00Z",
            }
        ).encode()
        (root / "lease.json").write_bytes(lease)
        self.contract = {
            "schema": fc.CONTRACT_SCHEMA,
            "state_dir": str(root / "state"),
            "owner_lease_file": str(root / "lease.json"),
            "owner_lease_sha256": hashlib.sha256(lease).hexdigest(),
            "owner_generation": 1,
            "receipts_repo": "SZLHOLDINGS/wave-logs",
            "adopted": [
                {
                    "group": "szl-train-example",
                    "repo": "SZLHOLDINGS/example",
                    "target_steps": 200,
                    "base": "org/base",
                    "replicas": 1,
                }
            ],
            "bench": {
                "group": "szl-bench-example",
                "dataset": "SZLHOLDINGS/frontier-54-runs",
                "prefix": PREFIX,
                "cutoff_utc": "2026-11-19T08:00:00Z",
                "adapter_repo": "SZLHOLDINGS/salad-flagship54-v1",
                "base": "SZLHOLDINGS/base",
                "base_revision": "2" * 40,
                "replicas": 1,
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    def receipts(self, command):
        ledger = fc.Ledger(self.contract["state_dir"], command)
        lines = ledger.receipts.read_text(encoding="utf-8").splitlines()
        self.assertTrue(gates.verify_chain(lines))
        return [json.loads(x) for x in lines]


class DoctorTests(Base):
    def run_doctor(self, salad, *, apply):
        fc.doctor_cycle(
            self.contract,
            salad,
            FakeHub(),
            fc.Ledger(self.contract["state_dir"], "doctor"),
            apply=apply,
        )
        return self.receipts("doctor")

    def test_dry_run_never_mutates_and_keeps_budget(self):
        salad = FakeSalad()
        recs = self.run_doctor(salad, apply=False)
        self.assertEqual(recs[0]["decision"], "RESTART_PERMITTED")
        self.assertEqual(recs[0]["action"]["reason"], "DRY_RUN_WITHOUT_APPLY")
        self.assertEqual(salad.mutations(), [])
        state = fc.Ledger(self.contract["state_dir"], "doctor").load()
        self.assertEqual(state["lanes"]["szl-train-example"]["attempts"], 0)

    def test_apply_restarts_then_reads_back(self):
        salad = FakeSalad()
        recs = self.run_doctor(salad, apply=True)
        self.assertEqual(
            salad.mutations(), [("POST", "/szl-train-example/start", None)]
        )
        readback = recs[0]["action"]["post_readback"]
        self.assertTrue(readback["provider_write_acknowledged"])
        self.assertFalse(readback["trainer_process_verified"])

    def test_apply_without_lease_does_not_mutate(self):
        Path(self.contract["owner_lease_file"]).write_text("{}")
        salad = FakeSalad()
        recs = self.run_doctor(salad, apply=True)
        self.assertEqual(recs[0]["decision"], "ALERT_STOPPED_NO_OWNER_LEASE")
        self.assertEqual(recs[0]["lease"], "OWNER_LEASE_DIGEST_MISMATCH")
        self.assertEqual(salad.mutations(), [])

    def test_running_lane_is_observed_only(self):
        salad = FakeSalad(status="running", running=1)
        self.assertEqual(self.run_doctor(salad, apply=True)[0]["decision"], "OBSERVE")
        self.assertEqual(salad.mutations(), [])


def receipt_bytes(**over):
    rec = {
        "schema": "szl.salad-heldout/v2",
        "kind": "MEASURED",
        "started_utc": "2026-11-19T09:00:00Z",
        "base": "SZLHOLDINGS/base",
        "base_revision": "2" * 40,
        "adapter_repo": "SZLHOLDINGS/salad-flagship54-v1",
        "base_arm_isolation": {
            "lora_layers_total": 4,
            "lora_layers_disabled_during_base_eval": 4,
        },
        "base_arm": {"refusal_rate": 0.4, "neutral_acc": 0.5, "neutral_correct": 10},
        "adapter_arm": {"refusal_rate": 0.6, "neutral_acc": 0.6, "neutral_correct": 12},
        "broken_control_arm": {},
        "broken_control_failed_rules": ["bc_refusal_band"],
        "battery_admissible": True,
    }
    rec.update(over)
    return json.dumps(rec).encode()


class BenchRunnerTests(Base):
    def run_bench(self, files, salad, *, apply, served=None):
        hub = FakeHub(dataset_files=files)
        hub.served = served or {}
        fc.bench_cycle(
            self.contract,
            salad,
            hub,
            fc.Ledger(self.contract["state_dir"], "bench"),
            apply=apply,
        )
        ledger = fc.Ledger(self.contract["state_dir"], "bench")
        return self.receipts("bench") if ledger.receipts.exists() else []

    def test_regression_diag_file_never_parks(self):
        salad = FakeSalad(status="running", running=1)
        recs = self.run_bench(
            {PREFIX + "diag-20261119T083831Z-final.json": receipt_bytes()},
            salad,
            apply=True,
        )
        self.assertEqual(recs, [])
        self.assertEqual(salad.mutations(), [])

    def test_verified_bound_receipt_parks_with_zero_readback(self):
        salad = FakeSalad(status="running", running=1)
        recs = self.run_bench(
            {PREFIX + "20261119T090000.json": receipt_bytes()}, salad, apply=True
        )
        self.assertEqual(
            recs[0]["decision"], "PARK_PROVIDER_ZERO_REPORTED_NOT_COMPLETION"
        )
        self.assertEqual([m[0] for m in salad.mutations()], ["POST", "PATCH"])
        self.assertEqual(
            recs[0]["action"]["post_readback"]["state"],
            "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION",
        )
        again = self.run_bench(
            {PREFIX + "20261119T090000.json": receipt_bytes()}, salad, apply=True
        )
        self.assertEqual(len(again), len(recs))

    def test_tampered_bytes_hold(self):
        name = PREFIX + "20261119T090000.json"
        salad = FakeSalad(status="running", running=1)
        recs = self.run_bench(
            {name: receipt_bytes()},
            salad,
            apply=True,
            served={name: receipt_bytes(kind="SAMPLE")},
        )
        self.assertEqual(recs[0]["decision"], "HOLD_RECEIPT_UNVERIFIED")
        self.assertEqual(salad.mutations(), [])

    def test_foreign_adapter_receipt_does_not_park(self):
        salad = FakeSalad(status="running", running=1)
        recs = self.run_bench(
            {
                PREFIX + "20261119T090000.json": receipt_bytes(
                    adapter_repo="SZLHOLDINGS/x"
                )
            },
            salad,
            apply=True,
        )
        self.assertEqual(recs[0]["decision"], "HOLD_RECEIPT_NOT_BOUND")
        self.assertEqual(salad.mutations(), [])

    def test_dry_run_does_not_stop_or_patch(self):
        salad = FakeSalad(status="running", running=1)
        recs = self.run_bench(
            {PREFIX + "20261119T090000.json": receipt_bytes()},
            salad,
            apply=False,
        )
        self.assertEqual(recs[0]["decision"], "PARK_PERMITTED")
        self.assertEqual(salad.mutations(), [])

    def test_rejected_stop_statuses_never_patch_or_replay(self):
        for status in (400, 401, 403, 429, 503):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as temp:
                    self.contract["state_dir"] = str(Path(temp) / "state")

                    class RejectedStop(FakeSalad):
                        def call(self, method, path, body=None, content_type=None):
                            if method == "POST" and path.endswith("/stop"):
                                self.calls.append((method, path, body))
                                return 0, status, b""
                            return super().call(method, path, body, content_type)

                    salad = RejectedStop(status="running", running=1)
                    files = {PREFIX + "20261119T090000.json": receipt_bytes()}
                    recs = self.run_bench(files, salad, apply=True)
                    self.assertEqual(
                        recs[0]["decision"], "HOLD_STOP_REQUEST_UNVERIFIED"
                    )
                    self.assertEqual(
                        salad.mutations(),
                        [("POST", "/szl-bench-example/stop", None)],
                    )
                    ledger = fc.Ledger(self.contract["state_dir"], "bench")
                    self.assertIn("szl-bench-example", ledger.load()["stop_intents"])
                    self.run_bench(files, salad, apply=True)
                    self.assertEqual(len(salad.mutations()), 1)

    def test_async_202_stop_cannot_cascade_into_patch(self):
        class SlowStop(FakeSalad):
            def call(self, method, path, body=None, content_type="application/json"):
                if method == "POST" and path.endswith("/stop"):
                    self.calls.append((method, path, body))
                    self.status = "stopping"
                    return 0, 202, b""
                return super().call(method, path, body, content_type)

        salad = SlowStop(status="running", running=1)
        files = {PREFIX + "20261119T090000.json": receipt_bytes()}
        recs = self.run_bench(files, salad, apply=True)
        self.assertEqual(recs[0]["decision"], "HOLD_STOP_SETTLEMENT_UNVERIFIED")
        self.assertEqual([x[0] for x in salad.mutations()], ["POST"])
        self.assertIn("stop_readback", recs[0]["action"])
        self.run_bench(files, salad, apply=True)
        self.assertEqual([x[0] for x in salad.mutations()], ["POST"])

    def test_owner_lease_changed_after_stop_forbids_patch(self):
        salad = FakeSalad(status="running", running=1)
        with patch.object(
            fc,
            "lease_check",
            side_effect=[
                (True, "OWNER_LEASE_VERIFIED"),
                (False, "OWNER_LEASE_DIGEST_MISMATCH"),
            ],
        ):
            recs = self.run_bench(
                {PREFIX + "20261119T090000.json": receipt_bytes()},
                salad,
                apply=True,
            )
        self.assertEqual(recs[0]["decision"], "HOLD_OWNER_LEASE_CHANGED")
        self.assertEqual([x[0] for x in salad.mutations()], ["POST"])

    def test_rejected_patch_retains_manual_hold_not_done(self):
        class BadPatch(FakeSalad):
            def call(self, method, path, body=None, content_type="application/json"):
                if method == "PATCH":
                    self.calls.append((method, path, body))
                    return 0, 429, b""
                return super().call(method, path, body, content_type)

        salad = BadPatch(status="running", running=1)
        files = {PREFIX + "20261119T090000.json": receipt_bytes()}
        recs = self.run_bench(files, salad, apply=True)
        self.assertEqual(recs[0]["decision"], "HOLD_PARK_REQUEST_UNVERIFIED")
        self.assertEqual([x[0] for x in salad.mutations()], ["POST", "PATCH"])
        ledger = fc.Ledger(self.contract["state_dir"], "bench")
        self.assertIn("szl-bench-example", ledger.load()["stop_intents"])
        self.run_bench(files, salad, apply=True)
        self.assertEqual([x[0] for x in salad.mutations()], ["POST", "PATCH"])

    def test_stop_intent_is_persisted_before_first_provider_effect(self):
        parent = self

        class CheckIntent(FakeSalad):
            def call(self, method, path, body=None, content_type="application/json"):
                if method == "POST" and path.endswith("/stop"):
                    ledger = fc.Ledger(parent.contract["state_dir"], "bench")
                    intent = ledger.load()["stop_intents"]["szl-bench-example"]
                    parent.assertEqual(intent["phase"], "POST_STOP_OUTCOME_UNRESOLVED")
                return super().call(method, path, body, content_type)

        salad = CheckIntent(status="running", running=1)
        recs = self.run_bench(
            {PREFIX + "20261119T090000.json": receipt_bytes()},
            salad,
            apply=True,
        )
        self.assertEqual(
            recs[0]["decision"], "PARK_PROVIDER_ZERO_REPORTED_NOT_COMPLETION"
        )
        ledger = fc.Ledger(self.contract["state_dir"], "bench")
        self.assertNotIn("szl-bench-example", ledger.load()["stop_intents"])


class BenchLeaseLaterTests(Base):
    run_bench = BenchRunnerTests.run_bench

    def test_lease_granted_later_still_parks_once(self):
        files = {PREFIX + "20261119T090000.json": receipt_bytes()}
        lease_path = Path(self.contract["owner_lease_file"])
        good = lease_path.read_bytes()
        lease_path.write_text("{}")
        salad = FakeSalad(status="running", running=1)
        first = self.run_bench(files, salad, apply=True)
        self.assertEqual(first[0]["decision"], "ALERT_PARK_NEEDS_OWNER_LEASE")
        quiet = self.run_bench(files, salad, apply=True)
        self.assertEqual(len(quiet), len(first))
        self.assertEqual(salad.mutations(), [])
        lease_path.write_bytes(good)
        parked = self.run_bench(files, salad, apply=True)
        self.assertEqual(
            parked[-2]["decision"], "PARK_PROVIDER_ZERO_REPORTED_NOT_COMPLETION"
        )
        self.assertEqual([m[0] for m in salad.mutations()], ["POST", "PATCH"])


class LockTests(Base):
    def test_second_controller_refused(self):
        first = fc.Ledger(self.contract["state_dir"], "doctor")
        first.lock()
        with self.assertRaises(SystemExit):
            fc.Ledger(self.contract["state_dir"], "doctor").lock()
        first._lock_handle.close()


if __name__ == "__main__":
    unittest.main()
