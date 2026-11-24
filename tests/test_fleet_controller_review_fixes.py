"""Regression tests for the PR #282 review findings left open after merge.

Access-stop must halt; omitted bench pins must not act as wildcards; a HOLD
qualification must not be cached forever; an unverified receipt download must be
re-assessed; doctor and bench must share one provider-action lock; one group must
not sit in two action modes. No network, no keys.
"""

import datetime as dt
import json
from pathlib import Path
import unittest
from unittest import mock

from test_artifact_qualification import fixture
from test_controller_gates import RECEIPT, SPEC
from test_fleet_controller import (
    PREFIX,
    Base,
    FakeHub,
    FakeSalad,
    receipt_bytes,
)
from tools.salad import controller_gates as gates
from tools.salad import fleet_controller as fc


class DeniedSalad(FakeSalad):
    def get_json(self, path):
        self.calls.append(("GET", path, None))
        return None, 403


class AccessStopHaltTests(Base):
    def write_contract(self):
        root = Path(self.tmp.name)
        for name in ("hf", "sk"):
            (root / name).write_text("not-a-real-secret")
        self.contract.update(
            hf_token_file=str(root / "hf"), salad_key_file=str(root / "sk")
        )
        self.contract["adopted"].append(
            dict(self.contract["adopted"][0], group="szl-train-second")
        )
        path = root / "contract.json"
        path.write_text(json.dumps(self.contract))
        return str(path)

    def test_denial_halts_run_and_blocks_restart_until_owner_clears(self):
        contract = self.write_contract()
        salad = DeniedSalad()
        with (
            mock.patch.object(fc, "Hub", lambda token: FakeHub()),
            mock.patch.object(fc, "Salad", lambda key: salad),
            mock.patch.object(fc.time, "sleep") as sleep,
        ):
            code = fc.main(["--contract", contract, "doctor", "--cycles", "5"])
            self.assertEqual(code, 3)
            sleep.assert_not_called()
            # The second adopted lane is never queried with the denied key.
            self.assertEqual([c[1] for c in salad.calls], ["/szl-train-example"])
            self.assertEqual(salad.mutations(), [])
            marker = Path(self.contract["state_dir"]) / fc.ACCESS_STOP_MARKER
            self.assertTrue(marker.exists())
            with self.assertRaises(SystemExit) as ctx:
                fc.main(["--contract", contract, "doctor"])
            self.assertIn("ACCESS_STOP_MARKER_PRESENT", str(ctx.exception))
            self.assertEqual(len(salad.calls), 1)
        self.assertEqual(self.receipts("doctor")[-1]["decision"], "HALT_ACCESS_STOP")

    def test_bench_denial_keeps_receipt_pending_and_raises(self):
        files = {PREFIX + "20261119T090000.json": receipt_bytes()}
        hub = FakeHub(dataset_files=files)
        ledger = fc.Ledger(self.contract["state_dir"], "bench")
        with self.assertRaises(fc.AccessStop):
            fc.bench_cycle(self.contract, DeniedSalad(), hub, ledger, apply=True)
        state = ledger.load()
        self.assertEqual(state["handled_receipts"], [])
        self.assertIn(PREFIX + "20261119T090000.json", state["pending_receipts"])


class BenchPinTests(unittest.TestCase):
    def test_omitted_pins_never_bind(self):
        for field in ("adapter_repo", "base", "base_revision", "cutoff_utc"):
            with self.subTest(field=field):
                spec = {k: v for k, v in SPEC.items() if k != field}
                verdict, gate = gates.assess_heldout_receipt(RECEIPT, spec)
                self.assertEqual(verdict, "GATE_FAIL_CONTRACT_PINS_MISSING")
                self.assertFalse(gate["bound"])
                self.assertEqual(
                    gates.decide_bench_park(
                        receipt_verified=True,
                        verdict=verdict,
                        lease_ok=True,
                        provider={"group_state_reconciled": True},
                    ),
                    "HOLD_RECEIPT_NOT_BOUND",
                )

    def test_malformed_pins_never_bind(self):
        for field, value in (
            ("base_revision", "main"),
            ("adapter_repo", "x/y"),
            ("base", " "),
        ):
            with self.subTest(field=field):
                verdict, _ = gates.assess_heldout_receipt(
                    RECEIPT, dict(SPEC, **{field: value})
                )
                self.assertEqual(verdict, "GATE_FAIL_CONTRACT_PINS_MISSING")


class QualificationRetryTests(Base):
    def test_hold_is_retried_after_backoff_and_success_is_kept(self):
        results = [
            {"status": "HOLD", "hold_reasons": ["TENSOR_INSPECTION_INCOMPLETE"]},
            {"status": "ARTIFACT_VERIFIED", "ready_for_evaluation": True},
        ]
        calls = []

        def qualify(api, fetch, candidate):
            calls.append(candidate["repo"])
            return results[len(calls) - 1]

        start = dt.datetime(2026, 11, 24, 12, 0, tzinfo=dt.timezone.utc)
        hub = FakeHub(model_files=fixture(step=200))
        ledger = fc.Ledger(self.contract["state_dir"], "doctor")
        decisions = []
        with mock.patch.object(fc, "observe_candidate", qualify):
            for minutes in (0, 30, 61, 120):
                with mock.patch.object(
                    fc, "now_utc", lambda m=minutes: start + dt.timedelta(minutes=m)
                ):
                    fc.doctor_cycle(
                        self.contract, FakeSalad(), hub, ledger, apply=False
                    )
                decisions.append(
                    [r for r in self.receipts("doctor") if r["kind"] == "doctor"][-1][
                        "decision"
                    ]
                )
        self.assertEqual(len(calls), 2)
        self.assertEqual(decisions[:2], ["ALERT_TERMINAL_UNQUALIFIED"] * 2)
        self.assertEqual(decisions[2:], ["ARTIFACT_VERIFIED_AWAITING_EVALUATION"] * 2)


class ReceiptRetryTests(Base):
    def test_unverified_download_is_reassessed(self):
        name = PREFIX + "20261119T090000.json"
        files = {name: receipt_bytes()}
        salad = FakeSalad(status="running", running=1)
        ledger = fc.Ledger(self.contract["state_dir"], "bench")
        hub = FakeHub(dataset_files=files)
        hub.served = {name: receipt_bytes(kind="SAMPLE")}
        fc.bench_cycle(self.contract, salad, hub, ledger, apply=False)
        hub.served = {}
        fc.bench_cycle(self.contract, salad, hub, ledger, apply=False)
        decisions = [
            r["decision"] for r in self.receipts("bench") if r["kind"] == "bench"
        ]
        self.assertEqual(decisions, ["HOLD_RECEIPT_UNVERIFIED", "PARK_PERMITTED"])
        self.assertEqual(salad.mutations(), [])


class SharedLockAndContractTests(Base):
    def test_doctor_and_bench_share_one_provider_action_lock(self):
        doctor = fc.Ledger(self.contract["state_dir"], "doctor")
        doctor.lock()
        try:
            with self.assertRaises(SystemExit):
                fc.Ledger(self.contract["state_dir"], "bench").lock()
            reader = fc.Ledger(self.contract["state_dir"], "checkpoints")
            reader.lock()
            reader._lock_handle.close()
        finally:
            doctor._lock_handle.close()

    def write(self, contract):
        path = Path(self.tmp.name) / "c.json"
        path.write_text(json.dumps(contract))
        return str(path)

    def test_group_in_two_action_modes_is_rejected(self):
        contract = dict(self.contract)
        contract["bench"] = dict(contract["bench"], group="szl-train-example")
        with self.assertRaises(SystemExit) as ctx:
            fc.load_contract(self.write(contract))
        self.assertIn("CONTRACT_GROUP_IN_TWO_ACTION_MODES", str(ctx.exception))

    def test_duplicate_adopted_group_is_rejected(self):
        contract = dict(self.contract)
        contract["adopted"] = contract["adopted"] * 2
        with self.assertRaises(SystemExit) as ctx:
            fc.load_contract(self.write(contract))
        self.assertIn("CONTRACT_DUPLICATE_ADOPTED_GROUP", str(ctx.exception))

    def test_lease_label_is_local_integrity_only(self):
        ok, label = fc.lease_check(
            self.contract, "szl-train-example", dt.datetime.now(dt.timezone.utc)
        )
        self.assertTrue(ok)
        self.assertEqual(label, "LOCAL_LEASE_INTEGRITY_OK")


if __name__ == "__main__":
    unittest.main()
