"""Adversarial GPUOS-inspired proof-carrying lease protocol tests; CPU ONLY."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import random
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

PATH = Path(__file__).resolve().parents[1] / "tools/gpu/lease_protocol.py"
SPEC = importlib.util.spec_from_file_location("szl_gpu_lease_protocol", PATH)
lease = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lease
SPEC.loader.exec_module(lease)

SOURCE_SHA = "5d8f6034a84fe37a4116ff9d7cb4a35f7dcbe667"
UPSTREAM_SHA = "9eb66b9c68a3e449fdb3b2bd6edf3ddf6bd6d2ca"
COMPLETE_DIGEST = "c" * 64


def key(task="alpha", generation=1, epoch=1):
    return lease.LeaseToken(task=task, generation=generation, epoch=epoch)


def sample_spec():
    token = {"task": "alpha", "generation": 1, "epoch": 1}
    return {
        "schema": lease.SCHEMA,
        "source": {
            "repository": "szl-holdings/szl-gpu-bridge",
            "revision": SOURCE_SHA,
            "upstream_revision": UPSTREAM_SHA,
            "evidence_kind": "SYNTHETIC",
        },
        "memory_limit_mib": 49152,
        "checkpoint": {
            "slice_overhead_us": 2,
            "overhead_budget_ppm": 20000,
            "pause_target_us": 500,
            "publication_us": 10,
            "drain_us": 20,
            "ack_us": 5,
        },
        "tail": {
            "samples": [{"baseline_us": 100, "shared_us": 104} for _ in range(300)],
            "slowdown_budget_ppm": 60000,
            "risk_budget_ppm": 10000,
            "confidence_ppm": 950000,
        },
        "events": [
            {
                "op": "prepare",
                **token,
                "backend": "finite_sliced",
                "trusted": True,
                "memory_mib": 1024,
            },
            {"op": "grant", **token, "start": 0, "extent": 10},
            {
                "op": "complete",
                **token,
                "grant_id": 1,
                "committed": 10,
                "completion_digest": COMPLETE_DIGEST,
            },
            {"op": "revoke", **token},
            {"op": "ack", **token},
            {"op": "retire", **token},
            {"op": "publish_epoch", "next_epoch": 2},
        ],
    }


class LeaseLedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = lease.LeaseLedger(memory_limit_mib=8192)
        self.token = key()
        self.ledger.prepare(
            self.token, backend="finite_sliced", trusted=True, memory_mib=1024
        )

    def test_complete_protocol_and_explicit_resource_return_mask(self):
        seq = self.ledger.grant(self.token, start=0, extent=64)
        self.assertEqual(seq, 1)
        self.assertEqual(self.ledger.revoke(self.token), 1)
        with self.assertRaisesRegex(lease.Hold, "GRANTS_CLOSED"):
            self.ledger.grant(self.token, start=64, extent=1)
        with self.assertRaisesRegex(lease.Hold, "INFLIGHT_NOT_DRAINED"):
            self.ledger.ack(self.token)
        self.assertEqual(
            self.ledger.complete(
                self.token,
                grant_id=seq,
                committed=64,
                completion_digest=COMPLETE_DIGEST,
            ),
            "COMPLETION_REPORTED_NOT_GPU_VERIFIED",
        )
        self.assertEqual(
            self.ledger.ack(self.token),
            {
                "future_issue_closed": True,
                "cta_return_reported": True,
                "memory_return_reported": False,
                "code_unload_safe": False,
            },
        )
        self.assertEqual(self.ledger.retire(self.token), "RETIRED_LOGICAL_ONLY")
        self.assertEqual(self.ledger.used_mib, 0)
        self.assertEqual(self.ledger.publish_epoch(next_epoch=2), 2)
        with self.assertRaisesRegex(lease.Hold, "STALE_OR_ABSENT_LEASE"):
            self.ledger.ack(self.token)

    def test_cooperative_resident_pause_does_not_claim_cta_release(self):
        t = key("cooperative")
        self.ledger.prepare(
            t, backend="cooperative_resident", memory_mib=100, trusted=True
        )
        self.ledger.revoke(t)
        ack = self.ledger.ack(t)
        self.assertFalse(ack["cta_return_reported"])
        self.assertFalse(ack["memory_return_reported"])
        self.assertFalse(ack["code_unload_safe"])

    def test_submitted_cursor_never_becomes_committed_early(self):
        seq = self.ledger.grant(self.token, start=0, extent=25)
        state = self.ledger.snapshot()["leases"][0]
        self.assertEqual(state["submitted"], 25)
        self.assertEqual(state["committed"], 0)
        for bad_cursor in (0, 24, 26):
            with self.subTest(committed=bad_cursor):
                with self.assertRaisesRegex(
                    lease.Hold, "COMMIT_BEFORE_DEVICE_COMPLETION_UNVERIFIED"
                ):
                    self.ledger.complete(
                        self.token,
                        grant_id=seq,
                        committed=bad_cursor,
                        completion_digest=COMPLETE_DIGEST,
                    )
        self.assertEqual(self.ledger.snapshot()["leases"][0]["committed"], 0)

    def test_reject_wrong_or_replayed_completion_event(self):
        seq = self.ledger.grant(self.token, start=0, extent=1)
        with self.assertRaisesRegex(lease.Hold, "COMPLETION_EVENT_DIGEST_INVALID"):
            self.ledger.complete(
                self.token, grant_id=seq, committed=1, completion_digest="x"
            )
        with self.assertRaisesRegex(lease.Hold, "STALE_OR_ABSENT_GRANT"):
            self.ledger.complete(
                self.token,
                grant_id=seq + 1,
                committed=1,
                completion_digest=COMPLETE_DIGEST,
            )
        self.ledger.complete(
            self.token,
            grant_id=seq,
            committed=1,
            completion_digest=COMPLETE_DIGEST,
        )
        with self.assertRaisesRegex(lease.Hold, "STALE_OR_ABSENT_GRANT"):
            self.ledger.complete(
                self.token,
                grant_id=seq,
                committed=1,
                completion_digest=COMPLETE_DIGEST,
            )

    def test_code_epoch_cannot_hot_swap_while_caller_live(self):
        with self.assertRaisesRegex(lease.Hold, "LIVE_CODE_CALLERS"):
            self.ledger.publish_epoch(next_epoch=2)
        self.ledger.revoke(self.token)
        self.ledger.ack(self.token)
        self.ledger.retire(self.token)
        self.ledger.publish_epoch(next_epoch=2)
        with self.assertRaisesRegex(lease.Hold, "CODE_EPOCH_STALE"):
            self.ledger.prepare(
                key("beta", 1, 1),
                backend="finite_sliced",
                memory_mib=1,
                trusted=True,
            )
        with self.assertRaisesRegex(lease.Hold, "CODE_EPOCH_STALE"):
            self.ledger.prepare(
                self.token,
                backend="finite_sliced",
                memory_mib=1,
                trusted=True,
            )
        self.ledger.prepare(
            key("alpha", 2, 2),
            backend="finite_sliced",
            memory_mib=1,
            trusted=True,
        )
        with self.assertRaisesRegex(lease.Hold, "GENERATION_REPLAY"):
            self.ledger.prepare(
                key("alpha", 2, 2),
                backend="finite_sliced",
                memory_mib=1,
                trusted=True,
            )

    def test_memory_is_reserved_even_for_idle_lease(self):
        extra = key("oversubscribed")
        with self.assertRaisesRegex(lease.Hold, "RESIDENT_MEMORY_EXCEEDED"):
            self.ledger.prepare(
                extra, memory_mib=8192, backend="finite_sliced", trusted=True
            )
        self.assertEqual(self.ledger.used_mib, 1024)

    def test_reader_prevents_reuse_until_last_consumer(self):
        self.ledger.reader_add(self.token)
        self.ledger.revoke(self.token)
        self.ledger.ack(self.token)
        with self.assertRaisesRegex(lease.Hold, "CONSUMER_STILL_LIVE"):
            self.ledger.retire(self.token)
        self.ledger.reader_done(self.token)
        self.ledger.retire(self.token)

    def test_new_readers_are_refused_after_revoke(self):
        self.ledger.revoke(self.token)
        with self.assertRaisesRegex(lease.Hold, "GRANTS_CLOSED"):
            self.ledger.reader_add(self.token)
        self.ledger.ack(self.token)
        self.ledger.retire(self.token)

    def test_untrusted_shared_gpu_is_not_tenant_isolation(self):
        for backend in lease.BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(lease.Hold, "TRUST_DOMAIN_UNVERIFIED"):
                    self.ledger.prepare(
                        key("untrusted"),
                        backend=backend,
                        memory_mib=100,
                        trusted=False,
                    )
        self.assertEqual(self.ledger.used_mib, 1024)

    def test_stale_slot_generation_never_reuses_old_lease(self):
        self.ledger.revoke(self.token)
        self.ledger.ack(self.token)
        self.ledger.retire(self.token)
        new = key("alpha", generation=2)
        self.ledger.prepare(new, backend="finite_sliced", memory_mib=10, trusted=True)
        with self.assertRaisesRegex(lease.Hold, "STALE_OR_ABSENT_LEASE"):
            self.ledger.grant(self.token, start=0, extent=1)

    def test_duplicate_generation_and_double_revoke_rejected(self):
        with self.assertRaisesRegex(lease.Hold, "GENERATION_REPLAY"):
            self.ledger.prepare(
                self.token,
                backend="finite_sliced",
                memory_mib=1,
                trusted=True,
            )
        self.ledger.revoke(self.token)
        with self.assertRaisesRegex(lease.Hold, "ALREADY_REVOKED"):
            self.ledger.revoke(self.token)

    def test_deterministic_adversarial_state_machine_stress(self):
        # 250 isolated sequences (5,000+ potential operations): no device use.
        actions = (
            "grant",
            "complete",
            "reader_add",
            "reader_done",
            "revoke",
            "ack",
            "retire",
        )
        for seed in range(250):
            rng = random.Random(seed)
            ledger = lease.LeaseLedger(memory_limit_mib=128)
            token = key(f"case{seed}")
            ledger.prepare(token, backend="finite_sliced", trusted=True, memory_mib=64)
            last_frontier = None
            for _ in range(24):
                act = rng.choice(actions)
                state = ledger.snapshot()["leases"][0]
                try:
                    if act == "grant":
                        ledger.grant(token, start=state["submitted"], extent=1)
                    elif act == "complete":
                        ledger.complete(
                            token,
                            grant_id=state["inflight"] or 1,
                            committed=state["submitted"],
                            completion_digest=COMPLETE_DIGEST,
                        )
                    elif act == "reader_add":
                        ledger.reader_add(token)
                    elif act == "reader_done":
                        ledger.reader_done(token)
                    elif act == "revoke":
                        ledger.revoke(token)
                    elif act == "ack":
                        ledger.ack(token)
                    else:
                        ledger.retire(token)
                except lease.Hold:
                    pass
                view = ledger.snapshot()
                record = view["leases"][0]
                self.assertGreaterEqual(record["submitted"], record["committed"])
                self.assertGreaterEqual(record["committed"], 0)
                self.assertIn(view["memory_reserved_mib"], (0, 64))
                self.assertFalse(view["gpu_state_observed"])
                if record["revoked"] is not None:
                    if last_frontier is None:
                        last_frontier = record["grant_frontier"]
                    self.assertEqual(record["grant_frontier"], last_frontier)
                if record["retired"]:
                    self.assertEqual(view["memory_reserved_mib"], 0)
                    break

    def test_unknown_backend_cannot_claim_hardware_preemption(self):
        with self.assertRaisesRegex(lease.Hold, "BACKEND_UNSUPPORTED"):
            self.ledger.prepare(
                key("fake"),
                backend="magic_hardware_preemption",
                memory_mib=1,
                trusted=True,
            )

    def test_atomic_revoke_and_grant_race_never_grants_after_closure(self):
        for index in range(60):
            ledger = lease.LeaseLedger(memory_limit_mib=100)
            token = key(f"task{index}")
            ledger.prepare(token, backend="finite_sliced", trusted=True, memory_mib=1)
            barrier = threading.Barrier(2)
            outcome: list[tuple[str, int | str]] = []

            def grant():
                barrier.wait()
                try:
                    outcome.append(("grant", ledger.grant(token, start=0, extent=1)))
                except lease.Hold as exc:
                    outcome.append(("denied", str(exc)))

            def revoke():
                barrier.wait()
                outcome.append(("revoked", ledger.revoke(token)))

            with ThreadPoolExecutor(max_workers=2) as pool:
                work = (pool.submit(grant), pool.submit(revoke))
                for future in work:
                    future.result(timeout=5)
            self.assertEqual(len(outcome), 2)
            self.assertEqual(ledger.leases[token].revoke_frontier in (0, 1), True)
            if ledger.leases[token].revoke_frontier == 0:
                self.assertIn(("denied", "GRANTS_CLOSED"), outcome)
            else:
                self.assertIn(("grant", 1), outcome)
            with self.assertRaisesRegex(lease.Hold, "GRANTS_CLOSED"):
                ledger.grant(token, start=0, extent=1)


class MathTests(unittest.TestCase):
    def test_overhead_and_pause_interval_uses_correct_units(self):
        profile = sample_spec()["checkpoint"]
        result = lease.checkpoint_interval(profile)
        self.assertEqual(result["minimum_useful_slice_us"], 98)
        self.assertEqual(result["maximum_useful_slice_us"], 465)
        self.assertTrue(result["nonempty_interval"])
        self.assertFalse(result["actual_preemption_measured"])

    def test_empty_checkpoint_interval_refuses_simulated_admission(self):
        profile = sample_spec()["checkpoint"]
        profile["pause_target_us"] = 5
        self.assertFalse(lease.checkpoint_interval(profile)["nonempty_interval"])

    def test_zero_violations_require_at_least_300_for_one_percent_tail(self):
        examples = sample_spec()["tail"]["samples"]
        result = lease.tail_budget(examples)
        self.assertTrue(result["statistically_eligible_if_iid_and_authentic"])
        self.assertLessEqual(result["upper_tail_violation_fraction_conditional"], 0.01)
        self.assertFalse(result["iid_independence_verified"])
        self.assertFalse(result["candidate_is_production_admission"])
        with self.assertRaisesRegex(lease.Hold, "TAIL_SAMPLE_COVERAGE_INSUFFICIENT"):
            lease.tail_budget(examples[:299])

    def test_one_excess_breaks_conservative_tail_gate(self):
        examples = sample_spec()["tail"]["samples"]
        examples[0]["shared_us"] = 200
        result = lease.tail_budget(examples)
        self.assertFalse(result["statistically_eligible_if_iid_and_authentic"])
        self.assertEqual(result["violations_observed"], 1)

    def test_nonfinite_and_boolean_calibrations_fail_closed(self):
        examples = sample_spec()["tail"]["samples"]
        for value in (float("nan"), float("inf"), True, -1):
            with self.subTest(value=str(value)):
                examples[0]["shared_us"] = value
                with self.assertRaises(lease.Hold):
                    lease.tail_budget(examples)
        sample = sample_spec()["tail"]["samples"]
        sample[0]["extra_secret"] = "not_disclosed"
        with self.assertRaisesRegex(lease.Hold, "TAIL_SAMPLE_FIELDS_INVALID"):
            lease.tail_budget(sample)


class ReplayTests(unittest.TestCase):
    def test_successful_simulation_never_grants_real_authority(self):
        result = lease.replay(sample_spec())
        self.assertEqual(result["failed_at"], None)
        self.assertEqual(result["state"], "SIMULATED_PLAN_FEASIBLE_NOT_ADMITTED")
        self.assertEqual(result["ledger"]["memory_reserved_mib"], 0)
        self.assertEqual(result["ledger"]["code_epoch"], 2)
        self.assertEqual(result["event_count"], 7)
        self.assertEqual(
            result["event_chain_signature"], "UNSIGNED_STRUCTURAL_HASH_ONLY"
        )
        self.assertFalse(result["authority"]["production_admission"])
        self.assertFalse(result["authority"]["gpu_kernels_launched"])
        self.assertFalse(result["authority"]["tenant_isolation_verified"])
        self.assertFalse(result["source"]["input_authenticated"])
        self.assertFalse(result["source"]["upstream_code_copied"])

    def test_empty_lease_activity_is_not_mathematically_admitted(self):
        data = sample_spec()
        data["events"] = [{"op": "publish_epoch", "next_epoch": 2}]
        result = lease.replay(data)
        self.assertEqual(result["state"], "SIMULATED_PLAN_HOLD")
        self.assertIsNone(result["failed_at"])

    def test_infeasible_pause_returns_nonzero_even_when_events_valid(self):
        data = sample_spec()
        data["checkpoint"]["pause_target_us"] = 5
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(lease.main(["--input", str(path)]), 2)
            self.assertEqual(
                json.loads(output.getvalue())["state"], "SIMULATED_PLAN_HOLD"
            )

    def test_change_in_causal_sequence_changes_digest(self):
        first = lease.replay(sample_spec())
        copy_ = sample_spec()
        copy_["events"].insert(
            3,
            {
                "op": "reader_add",
                "task": "alpha",
                "generation": 1,
                "epoch": 1,
            },
        )
        copy_["events"].insert(
            4,
            {
                "op": "reader_done",
                "task": "alpha",
                "generation": 1,
                "epoch": 1,
            },
        )
        second = lease.replay(copy_)
        self.assertNotEqual(first["event_chain_sha256"], second["event_chain_sha256"])
        self.assertEqual(second["failed_at"], None)

    def test_attempt_to_ack_before_draining_is_hold(self):
        data = sample_spec()
        data["events"].pop(2)
        response = lease.replay(data)
        self.assertEqual(response["failed_at"], 3)
        self.assertEqual(response["events"][-1]["hold_code"], "INFLIGHT_NOT_DRAINED")
        self.assertEqual(response["authority"]["provider_writes"], 0)

    def test_unknown_fields_and_secrets_are_not_echoed(self):
        data = sample_spec()
        data["events"][0]["command"] = "PRIVATE_SENSITIVE_TOKEN"
        result = lease.replay(data)
        self.assertEqual(result["failed_at"], 0)
        self.assertEqual(result["events"][0]["hold_code"], "EVENT_FIELDS_INVALID")
        self.assertNotIn("PRIVATE_SENSITIVE_TOKEN", json.dumps(result))

    def test_non_dict_event_never_crashes_or_leaks(self):
        data = sample_spec()
        data["events"][0] = "PRIVATE_SENSITIVE_TOKEN"
        result = lease.replay(data)
        self.assertEqual(result["failed_at"], 0)
        self.assertEqual(result["events"][0]["operation"], "invalid")
        self.assertNotIn("PRIVATE_SENSITIVE_TOKEN", json.dumps(result))

    def test_unhashable_operator_string_cannot_crash(self):
        data = sample_spec()
        data["events"][0]["op"] = ["PRIVATE_OP_TOKEN"]
        result = lease.replay(data)
        self.assertEqual(result["failed_at"], 0)
        self.assertEqual(result["events"][0]["operation"], "invalid")
        self.assertNotIn("PRIVATE_OP_TOKEN", json.dumps(result))

    def test_source_identity_or_backend_drift_not_admitted(self):
        data = sample_spec()
        data["source"]["upstream_revision"] = "a" * 40
        with self.assertRaisesRegex(lease.Hold, "UPSTREAM_REFERENCE_DRIFT"):
            lease.replay(data)
        data["source"]["upstream_revision"] = UPSTREAM_SHA
        data["source"]["evidence_kind"] = "PROVIDER_CERTIFIED"
        with self.assertRaisesRegex(lease.Hold, "UNAUTHENTICATED_EVIDENCE"):
            lease.replay(data)

    def test_html_is_static_escaped_and_never_claims_production(self):
        result = lease.replay(sample_spec())
        page = lease.render_html(result)
        self.assertIn("Content-Security-Policy", page)
        self.assertIn("connect-src 'none'", page)
        self.assertIn("RESEARCH HOLD", page)
        self.assertNotIn("<script", page)
        self.assertNotIn("fetch(", page)
        result["authority"]["production_admission"] = True
        with self.assertRaisesRegex(lease.Hold, "HTML_RENDER_UNTRUSTED"):
            lease.render_html(result)

    def test_cli_roundtrip_bounded_json_file_without_network(self):
        data = sample_spec()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            decoded, digest = lease.read_input(str(source))
            self.assertEqual(decoded, data)
            self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_duplicate_json_keys_and_unknown_nonfinite_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
            with self.assertRaisesRegex(lease.Hold, "DUPLICATE_JSON_KEY"):
                lease.read_input(str(path))
            path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(lease.Hold, "NONFINITE_JSON"):
                lease.read_input(str(path))

    def test_input_symlinks_and_large_files_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            large = root / "large.json"
            large.write_bytes(b"x" * (lease.MAX_INPUT_BYTES + 1))
            with self.assertRaisesRegex(lease.Hold, "SOURCE_BYTE_LIMIT"):
                lease.read_input(str(large))
            symlink = root / "alias.json"
            try:
                symlink.symlink_to(large)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable on this workstation")
            with self.assertRaisesRegex(lease.Hold, "SOURCE_UNAVAILABLE"):
                lease.read_input(str(symlink))

    def test_deterministic_replay_never_uses_wallclock_network_or_gpu(self):
        data = sample_spec()
        source = PATH.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import subprocess", "import socket"):
            self.assertNotIn(forbidden, source)
        result = lease.replay(data)
        self.assertFalse(result["authority"]["gpu_kernels_launched"])
        self.assertFalse(result["authority"]["provider_writes"])
        self.assertEqual(
            result["event_chain_sha256"], lease.replay(data)["event_chain_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
