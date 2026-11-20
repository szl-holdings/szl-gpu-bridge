"""Negative controls for the single-writer Salad admission gate (offline, stdlib).

Every ADMIT here is eligibility plus pinned authority for the single controller
replica only; the gate itself performs no I/O and never touches Salad. The
regression comments cite the retired ungated controllers (fleet_doctor_v2,
chaperone loops, 2026-11-20 halt receipt) and the task-21 dream session.
"""

import datetime as dt
import json
import unittest

from tools.salad import admission as sac

NOW = dt.datetime(2026, 11, 20, 7, 30, tzinfo=dt.timezone.utc)
BUDGET = {
    "remaining": 10_000_000,
    "start": 1_000_000,
    "replay_step": 10_000,
    "next_checkpoint": 500_000,
    "finalize": 250_000,
    "reserve": 1_000_000,
}


def registry(
    desired="STOPPED",
    budget=None,
    last=None,
    actors=None,
    group="szl-train-example",
):
    return {
        "schema": sac.REGISTRY_SCHEMA,
        "last_intent_sha256": last,
        "lanes": {
            "khipu-4b-v2": {
                "group": group,
                "desired_state": desired,
                "actors": actors if actors is not None else ["vibe-cto", "kimi-ctl"],
                "budget": budget if budget is not None else dict(BUDGET),
            }
        },
    }


def intent(op="restart", actor="vibe-cto", lane="khipu-4b-v2", prev=None, **over):
    body = {
        "schema": sac.INTENT_SCHEMA,
        "intent_id": "intent-0001",
        "actor": actor,
        "op": op,
        "lane_id": lane,
        "group": "szl-train-example",
        "prev_intent_sha256": prev,
    }
    body.update(over)
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def witness(
    state="REVIEW_CANDIDATE_ONLY",
    reason="FULL_STATE_DECLARED_UNATTESTED",
    replay=40,
):
    return {
        "schema": "szl.restart-lineage-witness/v1",
        "state": state,
        "reason": reason,
        "replay_steps_conditional": replay,
    }


GATES_OK = {"decision": "RESTART_PERMITTED", "restart_permitted": True}


def admit(intent_bytes, registry_obj, **over):
    kwargs = dict(
        intent_bytes=intent_bytes,
        registry=registry_obj,
        now=NOW,
        lease_ok=True,
        witness_verdict=witness(),
        gates_decision=GATES_OK,
    )
    kwargs.update(over)
    return sac.admit(**kwargs)


class RegistryTests(unittest.TestCase):
    def test_valid_registry_passes(self):
        reg = sac.validate_registry(registry())
        self.assertEqual(reg["lanes"]["khipu-4b-v2"]["desired_state"], "STOPPED")

    def test_registry_drift_fails_closed(self):
        lone = registry()["lanes"]["khipu-4b-v2"]
        cases = [
            ("REGISTRY_SCHEMA_MISMATCH", dict(registry(), schema="other/v1")),
            ("REGISTRY_SCHEMA_MISMATCH", dict(registry(), extra_key=1)),
            ("REGISTRY_SCHEMA_MISMATCH", dict(registry(), lanes={})),
            ("REGISTRY_HEAD_INVALID", dict(registry(), last_intent_sha256="not-hex")),
            (
                "REGISTRY_LANE_SCHEMA_MISMATCH",
                dict(registry(), lanes={"khipu-4b-v2": {"group": "szl-train-example"}}),
            ),
            ("REGISTRY_GROUP_INVALID", registry(group="Bad_Group!")),
            ("REGISTRY_STATE_INVALID", registry(desired="PAUSED")),
            ("REGISTRY_ACTORS_INVALID", registry(actors=[])),
            ("REGISTRY_ACTORS_INVALID", registry(actors=["a", "a"])),
            ("REGISTRY_ACTORS_INVALID", registry(actors=["Bad Actor"])),
            ("REGISTRY_ACTORS_INVALID", registry(actors=[{}])),
            ("REGISTRY_ACTORS_INVALID", registry(actors=[["unexpected"]])),
            ("REGISTRY_ACTORS_INVALID", registry(actors=["vibe-cto", {}])),
            ("BUDGET_SCHEMA_MISMATCH", registry(budget={"remaining": 1})),
            ("BUDGET_VALUE_INVALID", registry(budget={**BUDGET, "remaining": -1})),
            ("BUDGET_VALUE_INVALID", registry(budget={**BUDGET, "remaining": 1.5})),
            ("BUDGET_VALUE_INVALID", registry(budget={**BUDGET, "remaining": True})),
            (
                "REGISTRY_LANE_ID_INVALID",
                dict(registry(), lanes={"UPPER": lone}),
            ),
        ]
        for reason, reg in cases:
            with self.subTest(reason):
                with self.assertRaises(ValueError) as ctx:
                    sac.validate_registry(reg)
                self.assertEqual(str(ctx.exception), reason)


    def test_unhashable_actor_registry_is_denied_without_exception(self):
        # Registry data is untrusted. Never let set(actors) crash admission.
        for actors in ([{}], [["nested"]], ["vibe-cto", {}], ["vibe-cto", ["nested"]]):
            with self.subTest(actors=repr(actors)):
                result = admit(intent(), registry(actors=actors))
                self.assertEqual(result["decision"], "DENY")
                self.assertEqual(result["reason"], "REGISTRY_INVALID")
                self.assertFalse(result["salad_mutation_authorized"])


class IntentParseTests(unittest.TestCase):
    def test_byte_budget_denies(self):
        self.assertEqual(admit(b"", registry())["reason"], "INTENT_BYTE_BUDGET")
        self.assertEqual(
            admit(b"x" * (sac.MAX_INTENT_BYTES + 1), registry())["reason"],
            "INTENT_BYTE_BUDGET",
        )

    def test_invalid_json_still_recepits_a_denial(self):
        for raw, reason in (
            (b"\xff\xfe\x00", "INTENT_PARSE_FAILED"),
            (b"{", "INTENT_PARSE_FAILED"),
            (
                b'{"schema": "szl.salad-admission-intent/v1", "op": NaN}',
                "INTENT_PARSE_FAILED",
            ),
            (
                b'{"schema": "szl.salad-admission-intent/v1", "op": "restart", '
                b'"op": "stop", "intent_id": "x", "actor": "a", "lane_id": "l", '
                b'"group": "g", "prev_intent_sha256": null}',
                "INTENT_PARSE_FAILED",
            ),
            (b"[1, 2]", "INTENT_SCHEMA_MISMATCH"),
            (b'{"schema": 1}', "INTENT_SCHEMA_MISMATCH"),
        ):
            with self.subTest(reason):
                out = admit(raw, registry())
                self.assertEqual(out["decision"], "DENY")
                self.assertEqual(out["reason"], reason)
                self.assertFalse(out["salad_mutation_authorized"])
                self.assertRegex(out["sha256"], r"^[0-9a-f]{64}$")

    def test_intent_value_noise_denies(self):
        bad = [
            intent(op="deploy"),
            intent(actor="Bad Actor"),
            intent(lane="UPPERCASE"),
            intent(prev="not-hex"),
            intent(intent_id="bad id!"),
            intent(op=1),
        ]
        for raw in bad:
            with self.subTest(raw):
                self.assertEqual(
                    admit(raw, registry())["reason"], "INTENT_VALUE_INVALID"
                )


class ChainHeadTests(unittest.TestCase):
    def test_only_the_registry_head_is_judged(self):
        ok = admit(intent(), registry(last=None))
        self.assertEqual(ok["decision"], "ADMIT")
        stale = admit(intent(), registry(last="a" * 64))
        self.assertEqual(stale["reason"], "INTENT_NOT_CHAIN_HEAD")
        orphan = admit(intent(prev="b" * 64), registry(last=None))
        self.assertEqual(orphan["reason"], "INTENT_NOT_CHAIN_HEAD")


class AuthorizationTests(unittest.TestCase):
    def test_unknown_lane_group_and_foreign_actor(self):
        self.assertEqual(
            admit(intent(lane="nope-4b"), registry())["reason"], "LANE_UNKNOWN"
        )
        self.assertEqual(
            admit(intent(), registry(group="szl-train-other"))["reason"],
            "GROUP_MISMATCH",
        )
        self.assertEqual(
            admit(intent(actor="stranger"), registry())["reason"],
            "ACTOR_NOT_AUTHORIZED",
        )

    def test_state_op_table_denies(self):
        cases = [
            ("restart", "RUNNING"),
            ("stop", "STOPPED"),
            ("create", "STOPPED"),
            ("restart", "NEW"),
        ]
        for op, state in cases:
            with self.subTest((op, state)):
                self.assertEqual(
                    admit(intent(op=op), registry(desired=state))["reason"],
                    "OP_INVALID_FOR_DESIRED_STATE",
                )
        for state, reason in (("DONE", "LANE_TERMINAL"), ("PARKED", "LANE_PARKED")):
            for op in sac.OPS:
                with self.subTest((state, op)):
                    self.assertEqual(
                        admit(intent(op=op), registry(desired=state))["reason"], reason
                    )


class RestartAdmissionTests(unittest.TestCase):
    def test_full_evidence_admits(self):
        out = admit(intent(), registry())
        self.assertEqual(out["decision"], "ADMIT")
        self.assertTrue(out["salad_mutation_authorized"])
        self.assertTrue(out["lease_verified"])
        self.assertEqual(out["desired_state_after"], "RUNNING")
        self.assertEqual(out["witness_state"], "REVIEW_CANDIDATE_ONLY")
        self.assertEqual(out["gates_decision"], "RESTART_PERMITTED")
        self.assertEqual(out["replay_steps_conditional"], 40)
        self.assertEqual(out["required_micro_usd"], 3_150_000)

    def test_replay_arithmetic_is_charged(self):
        out = admit(intent(), registry(), witness_verdict=witness(replay=0))
        self.assertEqual(out["required_micro_usd"], 2_750_000)

    def test_budget_shortfall_denies_exact_cover_admits(self):
        required = 3_150_000
        out = admit(intent(), registry(budget={**BUDGET, "remaining": required - 1}))
        self.assertEqual(out["reason"], "BUDGET_SHORTFALL")
        self.assertEqual(out["required_micro_usd"], required)
        exact = admit(intent(), registry(budget={**BUDGET, "remaining": required}))
        self.assertEqual(exact["decision"], "ADMIT")

    def test_regression_witness_hold_never_admits(self):
        # fleet_doctor_v2 style restart over a run that already recorded work
        out = admit(
            intent(),
            registry(),
            witness_verdict=witness(
                state="HOLD", reason="GENESIS_REPLAYS_RECORDED_WORK", replay=200
            ),
        )
        self.assertEqual(out["reason"], "WITNESS_HOLD")
        self.assertFalse(out["salad_mutation_authorized"])

    def test_missing_or_mismatched_witness_denies(self):
        self.assertEqual(
            admit(intent(), registry(), witness_verdict=None)["reason"],
            "WITNESS_UNPROVIDED",
        )
        self.assertEqual(
            admit(
                intent(),
                registry(),
                witness_verdict=witness(
                    reason="FRESH_NO_PRIOR_STEPS_UNATTESTED", replay=0
                ),
            )["reason"],
            "WITNESS_VERDICT_MISMATCH",
        )
        self.assertEqual(
            admit(intent(), registry(), witness_verdict=witness(replay=None))["reason"],
            "WITNESS_REPLAY_UNMEASURED",
        )

    def test_fleet_gate_not_permitted_denies(self):
        for gates_decision in (
            {"decision": "HOLD_QUARANTINED", "restart_permitted": False},
            {"decision": "RESTART_PERMITTED", "restart_permitted": False},
            None,
        ):
            with self.subTest(gates_decision):
                self.assertEqual(
                    admit(intent(), registry(), gates_decision=gates_decision)[
                        "reason"
                    ],
                    "FLEET_GATE_NOT_PERMITTED",
                )

    def test_no_lease_never_admits(self):
        out = admit(intent(), registry(), lease_ok=False)
        self.assertEqual(out["reason"], "OWNER_LEASE_UNVERIFIED")
        self.assertFalse(out["salad_mutation_authorized"])


class CreateStopParkTests(unittest.TestCase):
    def test_genesis_create_admits_with_fresh_witness(self):
        out = admit(
            intent(op="create"),
            registry(desired="NEW"),
            witness_verdict=witness(reason="FRESH_NO_PRIOR_STEPS_UNATTESTED", replay=0),
        )
        self.assertEqual(out["decision"], "ADMIT")
        self.assertEqual(out["required_micro_usd"], 2_750_000)
        self.assertEqual(out["desired_state_after"], "RUNNING")

    def test_create_rejects_resume_evidence(self):
        out = admit(intent(op="create"), registry(desired="NEW"))
        self.assertEqual(out["reason"], "WITNESS_VERDICT_MISMATCH")

    def test_stop_and_park_need_only_lease(self):
        stop = admit(
            intent(op="stop"),
            registry(desired="RUNNING"),
            witness_verdict=None,
            gates_decision=None,
        )
        self.assertEqual(stop["decision"], "ADMIT")
        self.assertEqual(stop["desired_state_after"], "STOPPED")
        self.assertIsNone(stop["required_micro_usd"])
        park = admit(
            intent(op="park"),
            registry(desired="STOPPED"),
            witness_verdict=None,
            gates_decision=None,
        )
        self.assertEqual(park["decision"], "ADMIT")
        self.assertEqual(park["desired_state_after"], "PARKED")


class ReceiptChainTests(unittest.TestCase):
    def test_receipts_hash_chain_and_detect_tampering(self):
        first = admit(intent(), registry())
        self.assertRegex(first["sha256"], r"^[0-9a-f]{64}$")
        second = admit(
            intent(intent_id="intent-0002"),
            registry(),
            prev_receipt_sha256=first["sha256"],
        )
        self.assertEqual(second["prev_sha256"], first["sha256"])
        lines = [json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)]
        from tools.salad import controller_gates as gates

        self.assertTrue(gates.verify_chain(lines))
        tamper = lines[0].replace(
            '"required_micro_usd": 3150000', '"required_micro_usd": 3150001'
        )
        self.assertFalse(gates.verify_chain([tamper, lines[1]]))

    def test_bad_chain_seed_fails_closed(self):
        out = admit(intent(), registry(), prev_receipt_sha256="zz")
        self.assertEqual(out["reason"], "RECEIPT_CHAIN_SEED_INVALID")
        self.assertFalse(out["salad_mutation_authorized"])


class HonestyTests(unittest.TestCase):
    def test_admit_is_authorization_not_attestation(self):
        out = admit(intent(), registry())
        self.assertEqual(out["schema"], sac.RECEIPT_SCHEMA)
        self.assertEqual(out["signature_status"], "NOT_SIGNED")
        self.assertTrue(out["salad_mutation_authorized"])

    def test_naive_clock_denies(self):
        out = sac.admit(
            intent_bytes=intent(),
            registry=registry(),
            now=dt.datetime(2026, 11, 20, 7, 30),
        )
        self.assertEqual(out["reason"], "CLOCK_UNAVAILABLE")
        self.assertFalse(out["salad_mutation_authorized"])


if __name__ == "__main__":
    unittest.main()
