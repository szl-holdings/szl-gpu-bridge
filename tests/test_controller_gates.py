"""Negative controls for the reviewed fleet controller gates (no network, no keys).

Each ``test_regression_*`` reproduces a defect in the retired ad-hoc watchers
(fleet_doctor_v2, hub_checkpoint_watch_20261119, bench_receipt_watch_20261119).
"""

import copy
import datetime as dt
import hashlib
import json
import unittest

from test_artifact_qualification import FakeApi, fixture
from tools.salad import controller_gates as g
from tools.salad.provider_transition_witness import witness
from tools.salad.training_observer import observe_training

NOW = dt.datetime(2026, 11, 19, 16, 0, tzinfo=dt.timezone.utc)
LANE = {
    "group": "szl-train-example",
    "repo": "SZLHOLDINGS/example",
    "target_steps": 200,
    "base": "org/base",
    "replicas": 1,
}
LEASE = json.dumps(
    {
        "generation": 2,
        "groups": ["szl-train-example", "szl-bench-example"],
        "expires_utc": "2026-11-20T00:00:00Z",
    }
).encode()
CONTRACT = {
    "owner_lease_sha256": hashlib.sha256(LEASE).hexdigest(),
    "owner_generation": 2,
}


def provider(status, *, replicas=1, instances=(), counts=None, requested=1):
    counts = counts or {k: 0 for k in g.COUNTS}
    group = {
        "name": "szl-train-example",
        "replicas": replicas,
        "pending_change": False,
        "current_state": {"status": status, "instance_status_counts": counts},
    }
    return witness(
        transport_exit=0,
        http_status=200,
        body=b"",
        group=group,
        instances={"instances": list(instances)},
        name="szl-train-example",
        requested_replicas=requested,
    )


def training(step=100, target=200, files=None):
    api = FakeApi(files if files is not None else fixture(step=step, target=target))
    return observe_training(
        api, api.fetch, {"repo": "SZLHOLDINGS/example", "target_steps": target}
    )


def decide(prov, train, *, lease_ok=True, qual=None, ledger=None, now=NOW, lane=LANE):
    return g.decide_adopted_lane(
        contract=lane,
        lease_ok=lease_ok,
        provider=prov,
        training=train,
        qualification=qual,
        ledger=ledger,
        now=now,
    )


class LeaseTests(unittest.TestCase):
    def test_verified(self):
        self.assertEqual(
            g.verify_owner_lease(CONTRACT, LEASE, "szl-train-example", NOW),
            (True, "OWNER_LEASE_VERIFIED"),
        )

    def test_any_drift_revokes(self):
        cases = {
            "OWNER_LEASE_DIGEST_MISMATCH": (
                CONTRACT,
                LEASE + b" ",
                "szl-train-example",
                NOW,
            ),
            "OWNER_GENERATION_MISMATCH": (
                dict(CONTRACT, owner_generation=3),
                LEASE,
                "szl-train-example",
                NOW,
            ),
            "GROUP_NOT_LEASED": (CONTRACT, LEASE, "szl-train-other", NOW),
            "OWNER_LEASE_EXPIRED": (
                CONTRACT,
                LEASE,
                "szl-train-example",
                NOW + dt.timedelta(days=1),
            ),
            "OWNER_LEASE_NOT_PINNED": ({}, LEASE, "szl-train-example", NOW),
        }
        for reason, args in cases.items():
            with self.subTest(reason):
                self.assertEqual(g.verify_owner_lease(*args), (False, reason))


class AdoptedLaneTests(unittest.TestCase):
    def test_restart_needs_stopped_lease_and_pinned_resume_point(self):
        out = decide(provider("stopped"), training())
        self.assertEqual(out["decision"], "RESTART_PERMITTED")
        self.assertEqual(out["resume_step"], 100)
        self.assertEqual(out["resume_revision"], "a" * 40)
        self.assertEqual(out["ledger"]["attempts"], 1)
        self.assertFalse(out["training_complete_verified"])

    def test_regression_final_filename_is_not_completion(self):
        out = decide(provider("stopped"), training(step=200), qual={"status": "HOLD"})
        self.assertEqual(out["decision"], "ALERT_TERMINAL_UNQUALIFIED")
        self.assertFalse(out["restart_permitted"])
        self.assertFalse(out["artifact_verified"])

    def test_qualified_artifact_is_evaluation_eligible_only(self):
        out = decide(
            provider("stopped"),
            training(step=200),
            qual={"status": "ARTIFACT_VERIFIED", "ready_for_evaluation": True},
        )
        self.assertEqual(out["decision"], "ARTIFACT_VERIFIED_AWAITING_EVALUATION")
        self.assertFalse(out["training_complete_verified"])
        self.assertFalse(out["promotion_authorized"])

    def test_regression_unknown_status_is_not_stopped(self):
        out = decide(provider(None), training())
        self.assertEqual(out["decision"], "HOLD_PROVIDER_UNRECONCILED")
        self.assertFalse(out["restart_permitted"])

    def test_stopped_with_live_instances_holds(self):
        counts = dict({k: 0 for k in g.COUNTS}, running_count=1)
        prov = provider("stopped", counts=counts, instances=[{"state": "running"}])
        self.assertEqual(
            decide(prov, training())["decision"], "HOLD_PROVIDER_UNRECONCILED"
        )

    def test_replica_drift_by_another_owner_holds(self):
        self.assertEqual(
            decide(provider("stopped", replicas=0), training())["decision"],
            "HOLD_PROVIDER_UNRECONCILED",
        )

    def test_no_lease_alerts_without_restart(self):
        out = decide(provider("stopped"), training(), lease_ok=False)
        self.assertEqual(out["decision"], "ALERT_STOPPED_NO_OWNER_LEASE")

    def test_no_checkpoint_is_not_resumable(self):
        out = decide(provider("stopped"), training(files={".gitattributes": b"*"}))
        self.assertEqual(out["decision"], "ALERT_STOPPED_NO_RESUME_POINT")

    def test_failed_provider_is_alert_not_restart(self):
        self.assertEqual(
            decide(provider("failed"), training())["decision"], "ALERT_PROVIDER_FAILED"
        )

    def test_access_stop_halts(self):
        prov = witness(
            transport_exit=0,
            http_status=403,
            body=b"",
            group=None,
            instances=None,
            name="szl-train-example",
            requested_replicas=1,
        )
        self.assertEqual(decide(prov, training())["decision"], "HALT_ACCESS_STOP")

    def test_cooldown(self):
        first = decide(provider("stopped"), training())
        again = decide(
            provider("stopped"),
            training(),
            ledger=first["ledger"],
            now=NOW + dt.timedelta(minutes=5),
        )
        self.assertEqual(again["decision"], "HOLD_RESTART_COOLDOWN")

    def test_regression_crash_loop_quarantines_without_verified_progress(self):
        ledger, now = None, NOW
        for _ in range(3):
            out = decide(provider("stopped"), training(), ledger=ledger, now=now)
            self.assertEqual(out["decision"], "RESTART_PERMITTED")
            # v2 reset attempts on any non-stopped poll; a deploying blip must not.
            blip = decide(
                provider("deploying"), training(), ledger=out["ledger"], now=now
            )
            self.assertEqual(blip["ledger"]["attempts"], out["ledger"]["attempts"])
            ledger, now = blip["ledger"], now + dt.timedelta(hours=1)
        out = decide(provider("stopped"), training(), ledger=ledger, now=now)
        self.assertEqual(out["decision"], "QUARANTINE_CRASH_LOOP")
        held = decide(
            provider("stopped"), training(step=150), ledger=out["ledger"], now=now
        )
        self.assertEqual(held["decision"], "HOLD_QUARANTINED")

    def test_verified_progress_resets_budget(self):
        out = decide(
            provider("stopped"),
            training(step=100),
            ledger={"attempts": 2, "last_step": 50},
        )
        self.assertEqual(out["ledger"]["attempts"], 1)
        self.assertEqual(out["ledger"]["last_step"], 100)

    def test_regression_quarantine_is_per_group(self):
        quarantined = {"attempts": 3, "last_step": 100, "quarantined": True}
        self.assertEqual(
            decide(provider("stopped"), training(), ledger=quarantined)["decision"],
            "HOLD_QUARANTINED",
        )
        self.assertEqual(
            decide(provider("stopped"), training(), ledger=None)["decision"],
            "RESTART_PERMITTED",
        )


class CheckpointTests(unittest.TestCase):
    def test_regression_gitattributes_only_is_not_a_checkpoint(self):
        label = g.classify_checkpoint(training(files={".gitattributes": b"*"}))
        self.assertEqual(label["label"], "NO_CHECKPOINT")

    def test_regression_404_is_unobservable_not_skipped(self):
        class Missing(FakeApi):
            def model_info(self, repo, **kwargs):
                raise LookupError("404")

        api = Missing({})
        result = observe_training(
            api, api.fetch, {"repo": "SZLHOLDINGS/example", "target_steps": 1}
        )
        self.assertEqual(g.classify_checkpoint(result)["label"], "UNOBSERVABLE")

    def test_verified_state_does_not_claim_weights(self):
        label = g.classify_checkpoint(training(step=100))
        self.assertEqual(label["label"], "CHECKPOINT_STATE_VERIFIED")
        self.assertEqual(label["step"], 100)
        self.assertEqual(label["revision"], "a" * 40)
        self.assertFalse(label["adapter_weights_verified"])
        self.assertRegex(label["trainer_state_sha256"], r"^[0-9a-f]{64}$")

    def test_tampered_trainer_state_is_hold(self):
        files = fixture(step=100)
        api = FakeApi(files)
        api.blobs["checkpoint-100/trainer_state.json"] = (
            b'{"global_step": 100, "max_steps": 2000}'
        )
        result = observe_training(
            api, api.fetch, {"repo": "SZLHOLDINGS/example", "target_steps": 200}
        )
        self.assertNotEqual(
            g.classify_checkpoint(result)["label"], "CHECKPOINT_STATE_VERIFIED"
        )


PREFIX = "collaboration/vgl-criteria-v2/salad-v5-heldout-v2-"
CUTOFF = dt.datetime(2026, 11, 19, 8, 0, tzinfo=dt.timezone.utc)
SPEC = {
    "adapter_repo": "SZLHOLDINGS/salad-flagship54-v1",
    "base": "SZLHOLDINGS/base",
    "base_revision": "2" * 40,
    "cutoff_utc": "2026-11-19T08:00:00Z",
}
RECEIPT = {
    "schema": "szl.salad-heldout/v2",
    "kind": "MEASURED",
    "started_utc": "2026-11-19T09:00:00.000Z",
    "base": "SZLHOLDINGS/base",
    "base_revision": "2" * 40,
    "adapter_repo": "SZLHOLDINGS/salad-flagship54-v1",
    "base_arm_isolation": {
        "lora_layers_total": 132,
        "lora_layers_disabled_during_base_eval": 132,
    },
    "base_arm": {
        "refusal_rate": 0.45,
        "neutral_acc": 0.5,
        "neutral_correct": 10,
        "parse_fail": 0,
    },
    "adapter_arm": {
        "refusal_rate": 0.7,
        "neutral_acc": 0.75,
        "neutral_correct": 15,
        "parse_fail": 0,
    },
    "broken_control_arm": {"refusal_rate": 0.05},
    "broken_control_failed_rules": ["bc_refusal_band"],
    "battery_admissible": True,
}


class BenchTests(unittest.TestCase):
    def test_regression_lexical_cutoff_admitted_diag_names(self):
        names = [
            PREFIX + "20261117T102738.json",
            PREFIX + "diag-20261119T083831Z-final.json",
            PREFIX + "20261119T091500.json.tmp",
        ]
        # The retired filter: lexical >= admitted the diag file.
        self.assertGreaterEqual(names[1], PREFIX + "20261119T0800")
        self.assertIsNone(
            g.select_heldout_receipt(names, prefix=PREFIX, cutoff_utc=CUTOFF)
        )
        names.append(PREFIX + "20261119T091500.json")
        self.assertEqual(
            g.select_heldout_receipt(names, prefix=PREFIX, cutoff_utc=CUTOFF), names[-1]
        )

    def test_admissible_receipt_is_owner_review_only(self):
        verdict, gate = g.assess_heldout_receipt(RECEIPT, SPEC)
        self.assertEqual(verdict, "MEASURE_ADMISSIBLE_OWNER_REVIEW_REQUIRED")
        self.assertTrue(gate["broken_control_present"])

    def test_regression_broken_control_key_is_enforced(self):
        rec = {k: v for k, v in RECEIPT.items() if k != "broken_control_arm"}
        rec["broken_control"] = {"legacy": True}
        self.assertEqual(
            g.assess_heldout_receipt(rec, SPEC)[0],
            "GATE_FAIL_BROKEN_CONTROL_NOT_DISCRIMINATING",
        )
        rec = dict(RECEIPT, broken_control_failed_rules=[])
        self.assertEqual(
            g.assess_heldout_receipt(rec, SPEC)[0],
            "GATE_FAIL_BROKEN_CONTROL_NOT_DISCRIMINATING",
        )

    def test_foreign_or_stale_receipt_is_not_bound(self):
        for field, value in (
            ("adapter_repo", "SZLHOLDINGS/other"),
            ("base_revision", "3" * 40),
            ("started_utc", "2026-11-19T07:59:59Z"),
            ("schema", "x"),
        ):
            with self.subTest(field):
                verdict, _ = g.assess_heldout_receipt(
                    dict(RECEIPT, **{field: value}), SPEC
                )
                self.assertEqual(verdict, "GATE_FAIL_RECEIPT_NOT_BOUND_TO_THIS_RUN")

    def test_noop_and_isolation_gates(self):
        rec = copy.deepcopy(RECEIPT)
        rec["adapter_arm"].update(
            refusal_rate=0.45, neutral_acc=0.5, neutral_correct=10
        )
        self.assertEqual(
            g.assess_heldout_receipt(rec, SPEC)[0], "QUARANTINE_NOOP_SUSPECT"
        )
        rec = copy.deepcopy(RECEIPT)
        rec["base_arm_isolation"]["lora_layers_disabled_during_base_eval"] = 0
        self.assertEqual(
            g.assess_heldout_receipt(rec, SPEC)[0], "GATE_FAIL_BASE_ARM_NOT_ISOLATED"
        )

    def test_park_requires_every_gate(self):
        running = provider(
            "running",
            counts=dict({k: 0 for k in g.COUNTS}, running_count=1),
            instances=[{"state": "running"}],
        )
        ok = dict(
            receipt_verified=True,
            verdict="GATE_FAIL_NOT_MEASURED",
            lease_ok=True,
            provider=running,
        )
        self.assertEqual(g.decide_bench_park(**ok), "PARK_PERMITTED")
        self.assertEqual(
            g.decide_bench_park(**dict(ok, receipt_verified=False)),
            "HOLD_RECEIPT_UNVERIFIED",
        )
        self.assertEqual(
            g.decide_bench_park(**dict(ok, lease_ok=False)),
            "ALERT_PARK_NEEDS_OWNER_LEASE",
        )
        self.assertEqual(
            g.decide_bench_park(
                **dict(ok, verdict="GATE_FAIL_RECEIPT_NOT_BOUND_TO_THIS_RUN")
            ),
            "HOLD_RECEIPT_NOT_BOUND",
        )
        self.assertEqual(
            g.decide_bench_park(**dict(ok, provider=provider(None))),
            "HOLD_PROVIDER_UNRECONCILED",
        )

    def test_entry_bytes(self):
        raw = b'{"a":1}'
        blob = hashlib.sha1(b"blob 7\0" + raw).hexdigest()
        self.assertTrue(
            g.entry_bytes_verified(raw, size=7, lfs_sha256=None, blob_id=blob)
        )
        self.assertFalse(
            g.entry_bytes_verified(raw + b" ", size=8, lfs_sha256=None, blob_id=blob)
        )
        self.assertFalse(
            g.entry_bytes_verified(raw, size=7, lfs_sha256="0" * 64, blob_id=blob)
        )


class PublicationAndChainTests(unittest.TestCase):
    def test_publication_needs_pinned_identical_readback(self):
        good = g.publication_result(
            payload=b"x", commit_oid="c" * 40, readback=b"x", path_listed=True
        )
        self.assertEqual(good["state"], "PUBLISHED_READBACK_VERIFIED")
        for kwargs in (
            {"commit_oid": None},
            {"readback": b"y"},
            {"path_listed": False},
            {"commit_oid": "main"},
        ):
            args = {
                "payload": b"x",
                "commit_oid": "c" * 40,
                "readback": b"x",
                "path_listed": True,
            }
            args.update(kwargs)
            self.assertEqual(
                g.publication_result(**args)["state"], "PUBLICATION_UNVERIFIED"
            )

    def test_chain_detects_edit_and_reorder(self):
        a = g.chain_record({"n": 1}, None)
        b = g.chain_record({"n": 2}, a["sha256"])
        lines = [json.dumps(a), json.dumps(b)]
        self.assertTrue(g.verify_chain(lines))
        self.assertFalse(g.verify_chain(lines[::-1]))
        self.assertFalse(
            g.verify_chain([lines[0], lines[1].replace('"n": 2', '"n": 3')])
        )


if __name__ == "__main__":
    unittest.main()
