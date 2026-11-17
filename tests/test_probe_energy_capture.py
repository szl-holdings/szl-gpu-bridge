# SPDX-License-Identifier: Apache-2.0
"""Native unittest negative controls; synthetic data is not a live model run."""

from collections import deque
import json
import unittest

from tools.salad import probe_energy_capture as producer


def fixture():
    return [
        {
            "id": p,
            "kind": "answerable" if p.startswith("a") else "unanswerable",
            "prompt": f"SYNTHETIC::{p}",
            "reference": f"gold-{p}" if p.startswith("a") else None,
        }
        for p in producer.PROBE_IDS
    ]


def identity(rows=None):
    rows = rows if rows is not None else fixture()
    return producer.CaptureIdentity(
        model="SZLHOLDINGS/synthetic-do-not-publish",
        model_revision="a" * 40,
        run_id="synthetic_capture_001",
        fixture_sha256=producer.fixture_sha256(rows),
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
    )


class FakeMeter:
    device_uuid = "GPU-00000000-0000-0000-0000-000000000000"
    method = "NVML_DEVICE_TOTAL_MJ"

    def __init__(self, values=None):
        self.values = deque(
            values if values is not None else [1000 + 1000 * i for i in range(13)]
        )
        self.calls = 0

    def read_mj(self):
        self.calls += 1
        return self.values.popleft()


def answer(prompt):
    if prompt.endswith(("a1", "a2", "a3")):
        return "example-answer"
    return "I do not know"


def judge(case, response):
    return {
        "abstained": response == "I do not know",
        "verdict": "PASS",
    }


def _capture(*, rows=None, spec=None, invoke=answer, adjudicate=judge, meter=None):
    rows = rows if rows is not None else fixture()
    return producer.capture(
        spec if spec is not None else identity(rows),
        rows,
        invoke,
        adjudicate,
        meter if meter is not None else FakeMeter(),
    )


class CaptureTests(unittest.TestCase):
    def held(self, result, code):
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["reason_code"], code)
        self.assertIsNone(result["battery"])
        self.assertIsNone(result["energy"])
        self.assertIsNone(result["diagnostic"])
        self.assertIsNone(result["external_writes"])
        self.assertIs(result["producer_external_writes"], False)
        self.assertEqual(result["callback_external_effects"], "UNVERIFIED")
        self.assertIs(result["promotion_authorized"], False)
        self.assertIs(result["production_authorization"], False)

    def test_synthetic_same_run_pair_produces_unattested_diagnostic(self):
        calls = []

        def invoke(p):
            calls.append(p)
            return answer(p)

        result = _capture(invoke=invoke)
        self.assertEqual(result["status"], "DERIVED_UNATTESTED")
        self.assertEqual(calls, [row["prompt"] for row in fixture()])
        self.assertEqual(result["energy"]["counter_unit"], "mJ")
        self.assertEqual(
            result["energy"]["energy_scope"], "GPU_DEVICE_TOTAL_NOT_PROCESS_ISOLATED"
        )
        self.assertEqual(result["battery"]["score"], "6/6")
        self.assertEqual(result["diagnostic"]["metric"]["correct_abstentions"], 3)
        self.assertEqual(result["diagnostic"]["metric"]["measured_device_joules"], 6)
        self.assertEqual(result["diagnostic"]["metric"]["joint_quality_geomean"], 1)
        self.assertIs(result["diagnostic"]["signature_verified"], False)
        self.assertIs(result["diagnostic"]["ranking_authorized"], False)
        self.assertEqual(result["provenance"], "SOURCE_DECLARED_UNATTESTED")
        result_text = json.dumps(result)
        self.assertNotIn("SYNTHETIC::", result_text)
        self.assertNotIn("example-answer", result_text)
        self.assertNotIn("gold-a", result_text)

    def test_receipt_digests_bind_exact_observation_bytes(self):
        result = _capture()
        battery = result["battery"]
        energy = result["energy"]
        bsha = battery.pop("receipt_sha256")
        esha = energy.pop("receipt_sha256")
        self.assertEqual(bsha, producer._receipt_sha256(battery))
        self.assertEqual(esha, producer._receipt_sha256(energy))
        self.assertNotEqual(bsha, esha)

    def test_success_replay_is_deterministic_no_external_writes(self):
        a = _capture()
        b = _capture()
        self.assertEqual(a, b)
        self.assertIsNone(a["external_writes"])
        self.assertIs(a["producer_external_writes"], False)
        self.assertEqual(a["callback_external_effects"], "UNVERIFIED")
        self.assertIs(a["promotion_authorized"], False)

    def test_fixture_hash_blocks_any_inference(self):
        f = fixture()
        f[0]["prompt"] = "mutated"
        seen = []

        def invoke(p):
            seen.append(p)
            return answer(p)

        result = _capture(rows=f, spec=identity(fixture()), invoke=invoke)
        self.held(result, "FIXTURE_DIGEST_MISMATCH")
        self.assertEqual(seen, [])
        self.assertEqual(result["model_executed"], "NOT_INVOKED")

    def test_model_revision_and_gpu_uuid_preflight(self):
        f = fixture()
        bad = producer.CaptureIdentity(
            model="SZLHOLDINGS/synthetic-do-not-publish",
            model_revision="NOT_PINNED",
            run_id="one",
            fixture_sha256=producer.fixture_sha256(f),
            gpu_uuid=FakeMeter.device_uuid,
        )
        self.held(_capture(spec=bad), "MODEL_REVISION_UNAVAILABLE")
        m = FakeMeter()
        m.device_uuid = "GPU-11111111-2222-3333-4444-555555555555"
        self.held(_capture(meter=m), "METER_DEVICE_MISMATCH")
        self.assertEqual(m.calls, 0)

    def test_invalid_probe_set_is_refused_before_inference(self):
        for change, expected in (
            ("duplicate", "FIXTURE_ORDER_INVALID"),
            ("missing", "FIXTURE_COVERAGE_INVALID"),
            ("wrong_kind", "FIXTURE_KIND_INVALID"),
            ("unanswerable_reference", "UNANSWERABLE_HAS_REFERENCE"),
            ("large_prompt", "FIXTURE_PROMPT_INVALID"),
        ):
            with self.subTest(change=change):
                f = fixture()
                if change == "duplicate":
                    f[1]["id"] = "a1"
                elif change == "missing":
                    f.pop()
                elif change == "wrong_kind":
                    f[0]["kind"] = "unanswerable"
                elif change == "unanswerable_reference":
                    f[-1]["reference"] = "hidden"
                else:
                    f[0]["prompt"] = "x" * (producer.MAX_PROMPT_CHARS + 1)
                self.held(_capture(rows=f), expected)

    def test_missing_meter_and_invalid_counter_do_not_estimate(self):
        cases = (
            ("method", "METER_METHOD_UNAVAILABLE"),
            ("null", "METER_COUNTER_INVALID"),
            ("bool", "METER_COUNTER_INVALID"),
            ("no_delta", "METER_COUNTER_RESET_OR_UNMEASURED"),
            ("reset", "METER_COUNTER_RESET"),
        )
        for case, reason in cases:
            with self.subTest(case=case):
                m = FakeMeter()
                if case == "method":
                    m.method = "CALCULATED_WATTS"
                if case == "null":
                    m.values[0] = None
                if case == "bool":
                    m.values[0] = True
                if case == "no_delta":
                    m.values[1] = m.values[0]
                if case == "reset":
                    m.values[2] = m.values[1] - 1
                self.held(_capture(meter=m), reason)

    def test_inference_exception_is_not_disclosed(self):
        def fail(_):
            raise RuntimeError("SECRET_DO_NOT_PUBLISH")

        report = _capture(invoke=fail)
        self.held(report, "INFERENCE_UNAVAILABLE")
        self.assertEqual(report["model_executed"], "CALLBACK_INVOKED_UNVERIFIED")
        self.assertNotIn("SECRET", json.dumps(report))

    def test_judge_exception_and_invalid_flags_are_held(self):
        def raise_error(*_):
            raise RuntimeError("PRIVATE_REFERENCE")

        self.held(_capture(adjudicate=raise_error), "ADJUDICATION_UNAVAILABLE")
        self.held(
            _capture(
                adjudicate=lambda _case, _result: {"abstained": 1, "verdict": "PASS"}
            ),
            "ADJUDICATION_INVALID",
        )

    def test_contradictory_adjudication_does_not_publish_receipts(self):
        def incorrect(case, _result):
            return {"abstained": case["kind"] == "answerable", "verdict": "PASS"}

        self.held(_capture(adjudicate=incorrect), "VERDICT_ABSTENTION_CONFLICT")

    def test_empty_denominator_is_never_faked(self):
        def fail_all(_case, _result):
            return {"abstained": False, "verdict": "FAIL"}

        output = _capture(adjudicate=fail_all)
        self.assertEqual(output["status"], "DERIVED_UNATTESTED")
        self.assertIsNone(
            output["diagnostic"]["metric"]["device_joules_per_valid_outcome"]
        )
        self.assertEqual(output["diagnostic"]["metric"]["joint_quality_geomean"], 0)


class CallbackAuthorityTests(unittest.TestCase):
    def test_untrusted_callback_side_effects_are_not_reported_as_no_writes(self):
        observed = []

        def callback(prompt):
            observed.append(prompt)
            return answer(prompt)

        report = _capture(invoke=callback)
        self.assertEqual(len(observed), 6)
        self.assertEqual(report["status"], "DERIVED_UNATTESTED")
        self.assertIs(report["producer_external_writes"], False)
        self.assertIsNone(report["external_writes"])
        self.assertEqual(report["callback_external_effects"], "UNVERIFIED")


class NvmlDeviceMeterTests(unittest.TestCase):
    def test_opt_in_device_uuid_counter_and_shutdown(self):
        import sys
        import types
        from unittest.mock import patch

        calls = []
        fake = types.SimpleNamespace(
            nvmlInit=lambda: calls.append("init"),
            nvmlShutdown=lambda: calls.append("shutdown"),
            nvmlDeviceGetHandleByIndex=lambda idx: ("handle", idx),
            nvmlDeviceGetUUID=lambda handle: (
                b"GPU-00000000-0000-0000-0000-000000000000"
            ),
            nvmlDeviceGetTotalEnergyConsumption=lambda handle: 1234567,
        )
        with patch.dict(sys.modules, {"pynvml": fake}):
            with producer.NvmlDeviceMeter(0) as meter:
                self.assertEqual(meter.method, "NVML_DEVICE_TOTAL_MJ")
                self.assertEqual(meter.device_uuid, FakeMeter.device_uuid)
                self.assertEqual(meter.read_mj(), 1234567)
        self.assertEqual(calls, ["init", "shutdown"])
        with self.assertRaises(producer.CaptureHold):
            meter.read_mj()

    def test_invalid_device_index_fails_before_import(self):
        for bad in (True, -1, 16, "0"):
            with self.subTest(index=bad):
                with self.assertRaisesRegex(
                    producer.CaptureHold, "NVML_DEVICE_INDEX_INVALID"
                ):
                    producer.NvmlDeviceMeter(bad)

    def test_invalid_nvml_uuid_cleans_up_and_does_not_expose_driver_error(self):
        import sys
        import types
        from unittest.mock import patch

        calls = []
        fake = types.SimpleNamespace(
            nvmlInit=lambda: calls.append("init"),
            nvmlShutdown=lambda: calls.append("shutdown"),
            nvmlDeviceGetHandleByIndex=lambda _idx: object(),
            nvmlDeviceGetUUID=lambda _handle: "private-device-label",
        )
        with patch.dict(sys.modules, {"pynvml": fake}):
            with self.assertRaisesRegex(producer.CaptureHold, "NVML_UUID_UNAVAILABLE"):
                producer.NvmlDeviceMeter()
        self.assertEqual(calls, ["init", "shutdown"])

    def test_nvml_failure_is_fixed_hold_not_exception_text(self):
        import sys
        import types
        from unittest.mock import patch

        def bad_driver():
            raise RuntimeError("DO_NOT_LEAK_DRIVER_DETAILS")

        fake = types.SimpleNamespace(nvmlInit=bad_driver)
        with patch.dict(sys.modules, {"pynvml": fake}):
            with self.assertRaises(producer.CaptureHold) as ctx:
                producer.NvmlDeviceMeter()
        self.assertEqual(str(ctx.exception), "NVML_INIT_UNAVAILABLE")
        self.assertNotIn("DO_NOT_LEAK", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
