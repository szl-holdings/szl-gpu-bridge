"""Adversarial controls for the single-writer admission runner.

Built with the 2026-11-20 review round (Mistral Large 4 + Mistral Nemo,
receipts sac_ops_mistral_out.json): every adversarial scenario named in
that round is a control here, plus the outbox ordering and offline chain
verification properties.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import unittest

from tools.salad import admission
from tools.salad import admission_runner as runner
from tools.salad import controller_gates as gates

NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)


def _budget(start=0):
    return {
        "remaining": 1000,
        "start": start,
        "replay_step": 0,
        "next_checkpoint": 0,
        "finalize": 0,
        "reserve": 0,
    }


def _registry(start=1):
    return {
        "schema": "szl.desired-state-registry/v1",
        "last_intent_sha256": None,
        "lanes": {
            "lane-a": {
                "group": "training",
                "desired_state": "NEW",
                "actors": ["doctor-1"],
                "budget": _budget(start),
            },
        },
    }


def _intent(op, prev, *, actor="doctor-1", name="intent-1"):
    doc = {
        "schema": "szl.salad-admission-intent/v1",
        "intent_id": name,
        "actor": actor,
        "op": op,
        "lane_id": "lane-a",
        "group": "training",
        "prev_intent_sha256": prev,
    }
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def _create_evidence(trigger=None):
    def evidence(intent_bytes, registry, now):
        if trigger is not None:
            trigger()
        return {
            "lease_ok": True,
            "witness_verdict": {
                "state": "REVIEW_CANDIDATE_ONLY",
                "reason": "FRESH_NO_PRIOR_STEPS_UNATTESTED",
                "replay_steps_conditional": 0,
            },
            "gates_decision": None,
        }

    return evidence


class _Applier:
    def __init__(self, ok=True, raises=False):
        self.calls = []
        self.ok = ok
        self.raises = raises

    def __call__(self, intent_bytes, context):
        self.calls.append(context["intent_sha256"])
        if self.raises:
            raise RuntimeError("applier exploded")
        return {"ok": self.ok, "observed": {"http_status": 200}}


def _kinds(out):
    return [entry["kind"] for entry in out["ledger"]]


class DryRunTests(unittest.TestCase):
    def test_dry_run_admit_records_pending(self):
        registry = _registry()
        intents = [_intent("create", None)]
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        self.assertEqual(out["receipts"][0]["decision"], "ADMIT")
        self.assertEqual(_kinds(out), [runner.KIND_PENDING])
        head = out["ledger"][0]
        self.assertEqual(head["intent_sha256"], registry["last_intent_sha256"])
        self.assertEqual(head["desired_state_before"], "NEW")
        self.assertEqual(head["desired_state_after"], "RUNNING")
        self.assertEqual(registry["lanes"]["lane-a"]["desired_state"], "NEW")
        self.assertEqual(registry["lanes"]["lane-a"]["budget"]["remaining"], 1000)
        self.assertIsNone(out["halted"])

    def test_dry_run_requeue_is_already_recorded(self):
        registry = _registry()
        intents = [_intent("create", None)]
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        ledger = out["ledger"]
        out2 = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=ledger,
            evidence=_create_evidence(),
        )
        self.assertEqual(_kinds(out2), [runner.KIND_PENDING, runner.KIND_RECORDED])
        self.assertEqual(out2["receipts"], [])

    def test_second_intent_chains_off_first_admit(self):
        registry = _registry()
        first = _intent("create", None)
        second = _intent("stop", admission.intent_sha256(first), name="intent-2")
        out = runner.run_queue(
            registry,
            [first, second],
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        self.assertIsNone(out["halted"])
        self.assertEqual(out["receipts"][0]["decision"], "ADMIT")
        self.assertEqual(out["receipts"][1]["decision"], "DENY")
        self.assertEqual(out["receipts"][1]["reason"], "OP_INVALID_FOR_DESIRED_STATE")
        self.assertEqual(_kinds(out), [runner.KIND_PENDING])


class ExecutionTests(unittest.TestCase):
    def test_pending_executes_once_and_advances_state(self):
        registry = _registry(start=50)
        intents = [_intent("create", None)]
        dry = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        applier = _Applier()
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=dry["ledger"],
            evidence=_create_evidence(),
            applier=applier,
        )
        self.assertEqual(
            _kinds(out),
            [runner.KIND_PENDING, runner.KIND_EXECUTING, runner.KIND_EXECUTED],
        )
        self.assertEqual(len(applier.calls), 1)
        lane = registry["lanes"]["lane-a"]
        self.assertEqual(lane["desired_state"], "RUNNING")
        self.assertEqual(lane["budget"]["remaining"], 950)
        self.assertIsNone(out["halted"])

    def test_executed_requeue_never_reexecutes(self):
        registry = _registry(start=50)
        intents = [_intent("create", None)]
        dry = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        first = _Applier()
        runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=dry["ledger"],
            evidence=_create_evidence(),
            applier=first,
        )
        rival = _Applier()
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=dry["ledger"],
            evidence=_create_evidence(),
            applier=rival,
        )
        self.assertEqual(rival.calls, [])
        self.assertEqual(_kinds(out)[-1], runner.KIND_RECORDED)
        self.assertIsNone(out["halted"])

    def test_applier_not_ok_halts_fail_closed(self):
        registry = _registry()
        intents = [_intent("create", None)]
        applier = _Applier(ok=False)
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
            applier=applier,
        )
        self.assertEqual(out["halted"], "EXECUTION_FAILED")
        self.assertEqual(
            _kinds(out),
            [runner.KIND_EXECUTING, runner.KIND_FAILED],
        )
        note = out["ledger"][-1]["note"]
        self.assertEqual(note, "APPLIER_NOT_OK")

    def test_applier_raise_halts_and_blocks_rest_of_queue(self):
        registry = _registry()
        first = _intent("create", None)
        second = _intent("create", admission.intent_sha256(first), name="intent-2")
        applier = _Applier(raises=True)
        out = runner.run_queue(
            registry,
            [first, second],
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
            applier=applier,
        )
        self.assertEqual(out["halted"], "EXECUTION_FAILED")
        self.assertEqual(len(applier.calls), 1)
        self.assertEqual(len(out["receipts"]), 1)

    def test_budget_shortfall_at_execution_denies(self):
        registry = _registry(start=50)
        intents = [_intent("create", None)]
        dry = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        registry["lanes"]["lane-a"]["budget"]["remaining"] = 30
        applier = _Applier()
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=dry["ledger"],
            evidence=_create_evidence(),
            applier=applier,
        )
        self.assertEqual(applier.calls, [])
        self.assertEqual(out["halted"], "EXECUTION_FAILED")
        self.assertEqual(out["ledger"][-1]["note"], "BUDGET_SHORTFALL_AT_EXECUTION")

    def test_state_drift_at_execution_denies(self):
        registry = _registry()
        intents = [_intent("create", None)]
        dry = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        registry["lanes"]["lane-a"]["desired_state"] = "PARKED"
        applier = _Applier()
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=dry["ledger"],
            evidence=_create_evidence(),
            applier=applier,
        )
        self.assertEqual(applier.calls, [])
        self.assertEqual(out["ledger"][-1]["note"], "STATE_DRIFT_AT_EXECUTION")

    def test_sink_outbox_order_is_persist_before_apply(self):
        registry = _registry()
        intents = [_intent("create", None)]
        order = []

        def sink(entry):
            order.append("sink:" + entry["kind"])

        class Ordered(_Applier):
            def __call__(self, intent_bytes, context):
                order.append("apply")
                return super().__call__(intent_bytes, context)

        applier = Ordered()
        runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
            applier=applier,
            sink=sink,
        )
        self.assertEqual(
            order,
            ["sink:" + runner.KIND_EXECUTING, "apply", "sink:" + runner.KIND_EXECUTED],
        )


class LedgerIntegrityTests(unittest.TestCase):
    def _executed_ledger(self):
        registry = _registry()
        intents = [_intent("create", None)]
        dry = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=dry["ledger"],
            evidence=_create_evidence(),
            applier=_Applier(),
        )
        return registry, intents, dry["ledger"]

    def test_crash_ambiguous_ledger_refuses_to_run(self):
        registry = _registry()
        intents = [_intent("create", None)]
        dry = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        ledger = dry["ledger"]
        sha = admission.intent_sha256(intents[0])
        runner._chain_append(
            ledger,
            runner._entry(NOW, runner.KIND_EXECUTING, intent_sha=sha),
        )
        with self.assertRaises(ValueError) as ctx:
            runner.scan_ledger(ledger)
        self.assertIn("CRASH_AMBIGUOUS", str(ctx.exception))
        with self.assertRaises(ValueError):
            runner.run_queue(
                _registry(),
                intents,
                now=NOW,
                ledger=ledger,
                evidence=_create_evidence(),
                applier=_Applier(),
            )

    def test_duplicate_execution_fails_closed(self):
        _, intents, ledger = self._executed_ledger()
        sha = admission.intent_sha256(intents[0])
        runner._chain_append(
            ledger,
            runner._entry(NOW, runner.KIND_EXECUTING, intent_sha=sha),
        )
        runner._chain_append(
            ledger,
            runner._entry(NOW, runner.KIND_EXECUTED, intent_sha=sha),
        )
        with self.assertRaises(ValueError):
            runner.scan_ledger(ledger)

    def test_tampered_ledger_fails_closed(self):
        _, intents, ledger = self._executed_ledger()
        ledger[2]["observed"] = {"http_status": 500}
        with self.assertRaises(ValueError):
            runner.scan_ledger(ledger)

    def test_ledger_is_offline_verifiable(self):
        _, intents, ledger = self._executed_ledger()
        self.assertTrue(gates.verify_chain(runner.ledger_lines(ledger)))


class QueueIntegrityTests(unittest.TestCase):
    def test_deny_does_not_advance_head(self):
        registry = _registry()
        bad = _intent("create", None, actor="intruder", name="bad-1")
        good = _intent("create", None, name="good-1")
        out = runner.run_queue(
            registry,
            [bad, good],
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
        )
        self.assertEqual(out["receipts"][0]["reason"], "ACTOR_NOT_AUTHORIZED")
        self.assertEqual(out["receipts"][1]["decision"], "ADMIT")
        self.assertIsNone(out["halted"])

    def test_registry_edited_mid_queue_halts(self):
        registry = _registry(start=50)
        first = _intent("create", None)
        second = _intent("create", admission.intent_sha256(first), name="intent-2")
        calls = []

        def trigger():
            calls.append(1)
            if len(calls) >= 2:
                registry["schema"] = "drifted"

        out = runner.run_queue(
            registry,
            [first, second],
            now=NOW,
            ledger=[],
            evidence=_create_evidence(trigger=trigger),
        )
        self.assertEqual(out["receipts"][0]["decision"], "ADMIT")
        self.assertEqual(out["halted"], "REGISTRY_INVALID")

    def test_invalid_receipt_seed_halts(self):
        registry = _registry()
        intents = [_intent("create", None)]
        out = runner.run_queue(
            registry,
            intents,
            now=NOW,
            ledger=[],
            evidence=_create_evidence(),
            prev_receipt_sha256="not-a-sha",
        )
        self.assertEqual(out["receipts"][0]["reason"], "RECEIPT_CHAIN_SEED_INVALID")

    def test_default_evidence_denies_everything(self):
        registry = _registry()
        intents = [_intent("create", None)]
        out = runner.run_queue(registry, intents, now=NOW, ledger=[])
        self.assertEqual(out["receipts"][0]["reason"], "OWNER_LEASE_UNVERIFIED")
        self.assertEqual(_kinds(out), [])

    def test_naive_clock_refuses_to_run(self):
        registry = _registry()
        with self.assertRaises(ValueError):
            runner.run_queue(
                registry,
                [_intent("create", None)],
                now=NOW.replace(tzinfo=None),
                ledger=[],
                evidence=_create_evidence(),
            )


if __name__ == "__main__":
    unittest.main()
